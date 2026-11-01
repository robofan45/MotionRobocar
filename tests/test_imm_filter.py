"""
IMM filter: three motion models together track a turning vehicle better than
the best-tuned single filter, and the model probabilities follow the true
motion. Mixing, combining and the model likelihood are checked on cases with
a known answer.
"""
import numpy as np
import pytest
from scipy.stats import multivariate_normal

import conftest  # noqa: F401  (adds the project root to sys.path)
from Localization.imm_filter import imm_filter as m


@pytest.fixture(scope="module")
def run():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m, "show_animation", False)
        return m.main()


def position_rmse(estimates, truth):
    settled = slice(round(m.WARM_UP / m.DT), None)
    return np.sqrt(np.mean(np.sum(
        (estimates[settled, :2] - truth[settled, :2]) ** 2, axis=1)))


def test_imm_beats_the_best_single_filter(run):
    truth, _, measurements, imm, cv, _ = run
    imm_error = position_rmse(imm, truth)  # observed 0.454 m
    assert imm_error < 0.7 * position_rmse(cv, truth)  # observed 0.740 m
    assert imm_error < 0.4 * position_rmse(measurements, truth)  # 1.416 m


def test_model_probabilities_follow_the_true_motion(run):
    _, modes, _, _, _, probs = run
    settled = slice(round(m.WARM_UP / m.DT), None)
    modes, probs = modes[settled], probs[settled]
    # the lag right after each switch is the rest (observed 0.83)
    assert np.mean(probs.argmax(axis=1) == modes) > 0.75
    for j in range(len(m.MODEL_NAMES)):
        assert probs[modes == j, j].mean() > 0.6  # observed 0.73 to 0.78


def test_switch_and_mixing_probabilities_are_distributions():
    switch = m.switch_matrix(3, 0.05)
    np.testing.assert_allclose(switch.sum(axis=1), 1.0)
    predicted, mix = m.mixing_probabilities(np.array([0.7, 0.2, 0.1]),
                                            switch)
    assert predicted.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(mix.sum(axis=0), 1.0)


def test_identical_models_reduce_to_one_kalman_filter():
    """With two copies of one model, mixing and combining must change
    nothing, whatever the switching probability."""
    rng = np.random.default_rng(0)
    f, q = m.transition_matrix(0.0), m.process_noise(1.0)
    mean, cov = m.initial_estimate(np.zeros(2))
    means, covs = np.array([mean, mean]), np.array([cov, cov])
    probs, switch = np.array([0.5, 0.5]), m.switch_matrix(2, 0.3)
    kf_mean, kf_cov = mean, cov
    for z in rng.normal(0.0, 3.0, size=(30, 2)):
        means, covs, probs, imm_mean, imm_cov = m.imm_step(
            means, covs, probs, z, [f, f], [q, q], switch)
        kf_mean, kf_cov, _ = m.kalman_step(kf_mean, kf_cov, z, f, q)
        np.testing.assert_allclose(imm_mean, kf_mean, atol=1e-9)
        np.testing.assert_allclose(imm_cov, kf_cov, atol=1e-9)


def test_coordinated_turn_keeps_the_speed_and_turns_at_the_set_rate():
    state = np.array([0.0, 0.0, 3.0, 4.0])
    after = m.transition_matrix(m.TURN_RATE) @ state
    assert np.hypot(after[2], after[3]) == pytest.approx(5.0)
    turned = np.arctan2(after[3], after[2]) - np.arctan2(state[3], state[2])
    assert turned == pytest.approx(m.TURN_RATE * m.DT)
    # a vanishing turn rate gives the straight-line model
    np.testing.assert_allclose(m.transition_matrix(1e-8),
                               m.transition_matrix(0.0), atol=1e-7)


def test_likelihood_is_the_gaussian_density_of_the_measurement():
    f, q = m.transition_matrix(m.TURN_RATE), m.process_noise(0.5)
    mean = np.array([1.0, 2.0, 9.0, 1.0])
    cov = np.diag([2.0, 3.0, 4.0, 5.0])
    z = np.array([2.5, 1.0])
    predicted_mean, predicted_cov = f @ mean, f @ cov @ f.T + q
    expected = multivariate_normal(m.H @ predicted_mean,
                                   m.H @ predicted_cov @ m.H.T + m.R).pdf(z)
    _, _, likelihood = m.kalman_step(mean, cov, z, f, q)
    assert likelihood == pytest.approx(expected)


def test_same_result_twice(run, monkeypatch):
    monkeypatch.setattr(m, "show_animation", False)
    for first, second in zip(run, m.main()):
        np.testing.assert_array_equal(first, second)


if __name__ == '__main__':
    conftest.run_this_test(__file__)
