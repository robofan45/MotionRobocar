import math

import numpy as np
import pytest

import conftest  # Add root path to sys.path
from Localization.cubature_kalman_filter import cubature_kalman_filter as m

X_0 = m.x_0.copy()  # pristine initial state, saved before any run


def run_demo(monkeypatch):
    monkeypatch.setattr(m, "show_final", False)
    monkeypatch.setattr(m, "show_animation", False)
    monkeypatch.setattr(m, "show_ellipse", False)
    # main() changes the module-level x_0 in place (see the xfail test
    # below), so every run starts from its own copy of the initial state.
    monkeypatch.setattr(m, "x_0", X_0.copy())
    return m.main()


def position_rmse(a, b):
    return np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1)))


def test_estimate_is_closer_to_truth_than_measurements(monkeypatch):
    x_true, x_est, z = run_demo(monkeypatch)
    est_rmse = position_rmse(x_est, x_true)
    assert est_rmse < 0.2  # seeded run: 0.097 m
    assert est_rmse < position_rmse(z, x_true)  # raw GPS: 0.128 m


def test_two_runs_give_identical_results(monkeypatch):
    first = run_demo(monkeypatch)
    second = run_demo(monkeypatch)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def cubature_points(x, p):
    """2n points x +- sqrt(n) * columns of sqrt(P), each with weight 1/2n."""
    n = x.shape[0]
    eig_val, eig_vec = np.linalg.eigh(p)
    sd = eig_vec @ np.diag(np.sqrt(eig_val)) @ eig_vec.T  # symmetric root
    return [x[:, 0] + s * math.sqrt(n) * sd[:, i]
            for i in range(n) for s in (1.0, -1.0)]


X = np.array([[1.0], [2.0], [0.3], [1.5], [0.2]])
P = np.diag([0.04, 0.03, 0.01, 0.02, 0.005])


def test_prediction_mean_matches_cubature_rule():
    x_pred, _ = m.cubature_prediction(X.copy(), P.copy())
    mean = np.mean([m.f(pt.copy()) for pt in cubature_points(X, P)], axis=0)
    np.testing.assert_allclose(x_pred[:, 0], mean, atol=1e-12)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "cubature_prediction() propagates the cubature points twice: f() updates "
    "its argument in place, so the covariance loop applies the motion model "
    "again and the predicted covariance is too large"))
def test_prediction_covariance_matches_cubature_rule():
    _, p_pred = m.cubature_prediction(X.copy(), P.copy())
    points = np.array([m.f(pt.copy()) for pt in cubature_points(X, P)])
    deviation = points - points.mean(axis=0)
    expected = deviation.T @ deviation / len(points) + m.q
    np.testing.assert_allclose(p_pred, expected, atol=1e-12)


def test_update_matches_kalman_filter_for_linear_measurement():
    # h() is linear, so the cubature update must equal the Kalman update.
    z = np.array([[1.2], [1.8], [1.4], [0.25]])
    x_upd, p_upd = m.cubature_update(X.copy(), P.copy(), z)
    gain = P @ m.hx.T @ np.linalg.inv(m.hx @ P @ m.hx.T + m.r)
    np.testing.assert_allclose(x_upd, X + gain @ (z - m.hx @ X), atol=1e-12)
    np.testing.assert_allclose(p_upd, P - gain @ m.hx @ P, atol=1e-12)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "main() advances the module-level initial state x_0 in place (f() "
    "updates its argument), so a second run in the same process starts "
    "from a different state"))
def test_main_leaves_initial_state_unchanged(monkeypatch):
    run_demo(monkeypatch)
    np.testing.assert_array_equal(m.x_0, X_0)


if __name__ == '__main__':
    conftest.run_this_test(__file__)
