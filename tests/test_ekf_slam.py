import math

import numpy as np

import conftest  # Add root path to sys.path
from SLAM.EKFSLAM import ekf_slam as m


def run_demo(monkeypatch):
    monkeypatch.setattr(m, "show_animation", False)
    return m.main()


def position_rmse(estimate, truth):
    return np.sqrt(np.mean(np.sum((estimate[:2] - truth[:2]) ** 2, axis=0)))


def test_trajectory_and_landmarks_match_ground_truth(monkeypatch):
    first = run_demo(monkeypatch)
    x_est, x_true, x_dr, x_final, rfid = first
    est_rmse = position_rmse(x_est, x_true)
    assert est_rmse < 0.7  # seeded run: 0.35 m
    assert est_rmse < position_rmse(x_dr, x_true) / 5  # dead reckoning: 4.2 m

    # Each landmark is added to the state exactly once, close to the truth.
    landmarks = x_final[m.STATE_SIZE:, 0].reshape(-1, m.LM_SIZE)
    assert len(landmarks) == len(rfid)
    nearest = [int(np.argmin(np.hypot(*(rfid - lm).T))) for lm in landmarks]
    assert sorted(nearest) == list(range(len(rfid)))
    errors = np.hypot(*(landmarks - rfid[nearest]).T)
    assert errors.max() < 0.8  # seeded run: 0.42 m

    # A second run must give exactly the same result.
    for a, b in zip(first, run_demo(monkeypatch)):
        np.testing.assert_array_equal(a, b)


X = np.array([[1.0], [2.0], [0.4], [6.0], [3.0], [-2.0], [7.0]])  # pose + 2 LMs
U = np.array([[1.0], [0.1]])
EPS = 1e-6


def numeric_jacobian(func, x):
    return np.column_stack([(func(x + EPS * e) - func(x - EPS * e)).ravel()
                            / (2 * EPS) for e in np.eye(len(x))[:, :, None]])


def test_motion_jacobian_matches_motion_model():
    g, _ = m.jacob_motion(X, U)

    def motion(x):
        moved = x.copy()
        moved[:m.STATE_SIZE] = m.motion_model(x[:m.STATE_SIZE], U)
        return moved

    np.testing.assert_allclose(g, numeric_jacobian(motion, X), atol=1e-8)


def test_observation_jacobian_matches_range_bearing_model():
    for i in range(m.calc_n_lm(X)):
        lm = m.get_landmark_position_from_state(X, i)
        _, _, h = m.calc_innovation(lm, X, np.eye(len(X)), np.zeros(2), i)

        def observe(x, i=i):
            delta = m.get_landmark_position_from_state(x, i) - x[:2]
            return np.array([math.hypot(delta[0, 0], delta[1, 0]),
                             math.atan2(delta[1, 0], delta[0, 0]) - x[2, 0]])

        np.testing.assert_allclose(h, numeric_jacobian(observe, X), atol=1e-8)


def test_prediction_propagates_covariance_with_motion_jacobian():
    x = X[:m.STATE_SIZE].copy()
    p = np.array([[0.1, 0.0, 0.02], [0.0, 0.2, 0.03], [0.02, 0.03, 0.05]])
    g, fx = m.jacob_motion(x, U)
    _, p_pred = m.ekf_slam(x.copy(), p.copy(), U, np.zeros((0, 3)))
    np.testing.assert_allclose(p_pred, g @ p @ g.T + fx.T @ m.Cx @ fx,
                               atol=1e-12)


if __name__ == '__main__':
    conftest.run_this_test(__file__)
