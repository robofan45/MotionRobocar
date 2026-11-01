"""
iLQR: the optimised trajectory reaches the goal pose, the cost falls every
iteration, and the result is a true local minimum of the cost, checked
without trusting the backward pass.
"""
import math

import numpy as np
import pytest

import conftest  # noqa: F401  (adds the project root to sys.path)
from PathTracking.ilqr import ilqr as m


@pytest.fixture(scope="module")
def run():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m, "show_animation", False)
        return m.main()


def test_car_stops_at_the_goal_pose(run):
    xs, _, _ = run
    error = xs[-1] - m.GOAL
    assert math.hypot(error[0], error[1]) < 0.05  # observed 0.006 m
    assert abs(math.degrees(error[2])) < 1.0  # observed 0.26 deg
    assert abs(error[3]) < 0.05  # observed 0.021 m/s


def test_every_accepted_iteration_lowers_the_cost(run):
    costs = run[2]
    assert len(costs) > 5  # observed 11 iterations
    assert np.all(np.diff(costs) < 0.0)
    assert costs[-1] < costs[0] / 100.0  # observed 8684 -> 7.38


def test_states_are_the_controls_replayed_through_the_model(run):
    xs, us, _ = run
    np.testing.assert_allclose(m.rollout(m.START, us), xs, rtol=0.0,
                               atol=1e-12)


def test_steering_stays_within_its_limit(run):
    _, us, _ = run
    assert np.abs(us[:, 1]).max() <= m.MAX_STEER  # observed 25.5 deg


def test_jacobians_match_finite_differences():
    rng = np.random.default_rng(0)
    eps = 1e-6
    for _ in range(20):
        x = rng.uniform([-5.0, -5.0, -np.pi, -3.0], [5.0, 5.0, np.pi, 8.0])
        u = rng.uniform([-2.0, -0.5], [2.0, 0.5])
        fx, fu = m.jacobians(x, u)
        num_fx = np.column_stack([
            (m.motion_model(x + eps * e, u) - m.motion_model(x - eps * e, u))
            / (2 * eps) for e in np.eye(4)])
        num_fu = np.column_stack([
            (m.motion_model(x, u + eps * e) - m.motion_model(x, u - eps * e))
            / (2 * eps) for e in np.eye(2)])
        np.testing.assert_allclose(fx, num_fx, atol=1e-7)
        np.testing.assert_allclose(fu, num_fu, atol=1e-7)


def cost_slope(us, direction, eps=1e-5):
    """Directional derivative of the total cost, by central differences."""
    up, down = us + eps * direction, us - eps * direction
    return (m.total_cost(m.rollout(m.START, up), up)
            - m.total_cost(m.rollout(m.START, down), down)) / (2 * eps)


def test_result_is_a_local_minimum(run):
    """The cost no longer changes to first order in any direction, while it
    did steeply at the starting guess."""
    _, us, _ = run
    rng = np.random.default_rng(1)
    for _ in range(10):
        direction = rng.normal(size=us.shape)
        direction /= np.linalg.norm(direction)
        assert abs(cost_slope(np.zeros_like(us), direction)) > 1.0  # >= 14
        assert abs(cost_slope(us, direction)) < 1e-2  # observed <= 2.4e-4


if __name__ == '__main__':
    conftest.run_this_test(__file__)
