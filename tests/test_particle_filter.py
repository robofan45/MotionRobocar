import numpy as np

import conftest  # Add root path to sys.path
from Localization.particle_filter import particle_filter as m


def position_rmse(estimate, truth):
    return np.sqrt(np.mean(np.sum((estimate[:2] - truth[:2]) ** 2, axis=0)))


def test_estimate_beats_dead_reckoning(monkeypatch):
    monkeypatch.setattr(m, "show_animation", False)
    first = m.main()
    x_est, x_true, x_dr = first
    est_rmse = position_rmse(x_est, x_true)
    assert est_rmse < 0.35  # seeded run: 0.16 m
    assert est_rmse < position_rmse(x_dr, x_true) / 10  # dead reckoning: 7.5 m

    # A second run must give exactly the same result.
    for a, b in zip(first, m.main()):
        np.testing.assert_array_equal(a, b)


def test_low_variance_resampling_keeps_particles_in_proportion_to_weights():
    np.random.seed(0)
    weights = np.arange(1.0, m.NP + 1.0)
    weights /= weights.sum()
    px = np.tile(np.arange(m.NP, dtype=float), (4, 1))  # column i holds i
    new_px, new_pw = m.re_sampling(px, weights.reshape(1, -1))
    counts = np.bincount(new_px[0].astype(int), minlength=m.NP)
    # Low variance resampling keeps floor(NP * w) or ceil(NP * w) copies.
    assert np.all(np.abs(counts - m.NP * weights) < 1.0)
    np.testing.assert_allclose(new_pw, 1.0 / m.NP)


if __name__ == '__main__':
    conftest.run_this_test(__file__)
