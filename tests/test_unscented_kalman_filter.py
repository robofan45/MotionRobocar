import numpy as np

import conftest  # Add root path to sys.path
from Localization.unscented_kalman_filter import unscented_kalman_filter as m


def position_rmse(estimate, truth):
    return np.sqrt(np.mean(np.sum((estimate[:2] - truth[:2]) ** 2, axis=0)))


def test_estimate_beats_gps_and_dead_reckoning(monkeypatch):
    monkeypatch.setattr(m, "show_animation", False)
    x_est, x_true, x_dr, z = m.main()
    est_rmse = position_rmse(x_est, x_true)
    assert est_rmse < 0.45  # seeded run: 0.22 m
    assert est_rmse < position_rmse(z, x_true)  # raw GPS: 0.34 m
    assert est_rmse < position_rmse(x_dr, x_true) / 10  # dead reckoning: 3.9 m


def test_two_runs_give_identical_results(monkeypatch):
    monkeypatch.setattr(m, "show_animation", False)
    for a, b in zip(m.main(), m.main()):
        np.testing.assert_array_equal(a, b)


# With ALPHA = 0.001 the sigma point weights are about 1e6, so these exact
# identities hold only up to roundoff (observed about 1e-10).
TOL = 1e-8


def test_sigma_points_reproduce_mean_and_covariance():
    wm, wc, gamma = m.setup_ukf(4)
    x = np.array([[1.0], [2.0], [0.3], [0.8]])
    p = np.array([[0.5, 0.1, 0.0, 0.0],
                  [0.1, 0.4, 0.0, 0.0],
                  [0.0, 0.0, 0.2, 0.05],
                  [0.0, 0.0, 0.05, 0.3]])
    sigma = m.generate_sigma_points(x, p, gamma)
    assert abs(wm.sum() - 1.0) < TOL
    np.testing.assert_allclose(sigma @ wm.T, x, atol=TOL)
    np.testing.assert_allclose(
        m.calc_sigma_covariance(x, sigma, wc, np.zeros((4, 4))), p, atol=TOL)


def test_matches_kalman_filter_when_motion_is_linear():
    # With (almost) no yaw uncertainty the motion model is affine in the
    # state, so one UKF step must equal one exact Kalman filter step.
    wm, wc, gamma = m.setup_ukf(4)
    x = np.array([[1.0], [2.0], [0.3], [0.8]])
    p = np.diag([0.5, 0.4, 1e-12, 0.3])
    p[0, 1] = p[1, 0] = 0.1
    z = np.array([[1.4], [1.7]])
    u = np.array([[1.1], [0.05]])
    x_est, p_est = m.ukf_estimation(x, p, z, u, wm, wc, gamma)

    f = np.diag([1.0, 1.0, 1.0, 0.0])  # velocity comes from the input
    h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    x_pred = m.motion_model(x, u)
    p_pred = f @ p @ f.T + m.Q
    gain = p_pred @ h.T @ np.linalg.inv(h @ p_pred @ h.T + m.R)
    np.testing.assert_allclose(x_est, x_pred + gain @ (z - h @ x_pred),
                               atol=TOL)
    np.testing.assert_allclose(p_est, (np.eye(4) - gain @ h) @ p_pred,
                               atol=TOL)


if __name__ == '__main__':
    conftest.run_this_test(__file__)
