import numpy as np

import conftest  # Add root path to sys.path
from Localization.ensemble_kalman_filter import ensemble_kalman_filter as m


def position_rmse(estimate, truth):
    return np.sqrt(np.mean(np.sum((estimate[:2] - truth[:2]) ** 2, axis=0)))


def test_estimate_beats_dead_reckoning(monkeypatch):
    monkeypatch.setattr(m, "show_animation", False)
    first = m.main()
    x_est, x_true, x_dr = first
    est_rmse = position_rmse(x_est, x_true)
    assert est_rmse < 0.32  # seeded run: 0.16 m
    assert est_rmse < position_rmse(x_dr, x_true) / 10  # dead reckoning: 5.9 m

    # A second run must give exactly the same result.
    for a, b in zip(first, m.main()):
        np.testing.assert_array_equal(a, b)


def test_covariance_is_ensemble_covariance():
    rng = np.random.default_rng(0)
    px = rng.normal(size=(4, m.NP))
    x_est = px.mean(axis=1, keepdims=True)
    deviation = px[:3] - x_est[:3]
    np.testing.assert_allclose(m.calc_covariance(x_est, px),
                               deviation @ deviation.T / m.NP, atol=1e-12)


if __name__ == '__main__':
    conftest.run_this_test(__file__)
