import numpy as np
import pytest

import conftest  # Add root path to sys.path
from Localization.extended_kalman_filter import extended_kalman_filter as m
from Localization.extended_kalman_filter import \
    ekf_with_velocity_correction as mv


def position_rmse(estimate, truth):
    return np.sqrt(np.mean(np.sum((estimate[:2] - truth[:2]) ** 2, axis=0)))


def test_estimate_beats_gps_and_dead_reckoning(monkeypatch):
    monkeypatch.setattr(m, "show_animation", False)
    x_est, x_true, x_dr, z = m.main()
    est_rmse = position_rmse(x_est, x_true)
    assert est_rmse < 0.35  # seeded run: 0.17 m
    assert est_rmse < position_rmse(z, x_true)  # raw GPS: 0.34 m
    assert est_rmse < position_rmse(x_dr, x_true) / 10  # dead reckoning: 3.9 m


def test_two_runs_give_identical_results(monkeypatch):
    monkeypatch.setattr(m, "show_animation", False)
    for a, b in zip(m.main(), m.main()):
        np.testing.assert_array_equal(a, b)


def test_velocity_correction_estimates_scale_factor(monkeypatch):
    monkeypatch.setattr(mv, "show_animation", False)
    x_est, x_true, x_dr, z = mv.main()
    # The true velocity scale factor is 0.9 and the filter starts from 1.0.
    assert abs(x_est[4, -1] - 0.9) < 0.008  # seeded run: 0.896
    assert np.max(np.abs(x_est[4, -100:] - 0.9)) < 0.01  # seeded run: 0.005
    est_rmse = position_rmse(x_est, x_true)
    assert est_rmse < 0.005  # seeded run: 0.0025 m
    assert est_rmse < position_rmse(z, x_true)  # raw GPS: 0.0034 m


def test_velocity_correction_two_runs_give_identical_results(monkeypatch):
    monkeypatch.setattr(mv, "show_animation", False)
    for a, b in zip(mv.main(), mv.main()):
        np.testing.assert_array_equal(a, b)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "jacob_f() treats velocity as a persistent state (dx/dv = dt*cos(yaw), "
    "dv/dv = 1), but motion_model() takes the velocity from the input "
    "(F[3, 3] = 0), so the Jacobian does not match the motion model"))
@pytest.mark.parametrize("module, x", [
    pytest.param(m, [1.0, 2.0, 0.4, 0.7], id="ekf"),
    pytest.param(mv, [1.0, 2.0, 0.4, 0.7, 0.95], id="velocity_correction"),
])
def test_jacobian_matches_motion_model(module, x):
    x = np.array(x).reshape(-1, 1)
    u = np.array([[1.2], [0.1]])
    eps = 1e-6
    numeric = np.hstack([
        (module.motion_model(x + eps * e, u)
         - module.motion_model(x - eps * e, u)) / (2 * eps)
        for e in np.eye(len(x))[:, :, None]])
    np.testing.assert_allclose(module.jacob_f(x, u), numeric, atol=1e-6)


if __name__ == '__main__':
    conftest.run_this_test(__file__)
