"""
Answer checks for the dynamic window approach demo.

Each scenario runs once (the stuck one takes about 30 s) on a fresh Config,
so its settings do not leak into the module. The robot must reach the goal
without a collision and drive like the robot it simulates: speed and turn
rate inside their limits, changing no faster than the acceleration limits
allow within one time step, and every state following from the previous
one by the unicycle motion model (re-implemented here).
"""
import math

import numpy as np
import pytest

import conftest
from PathPlanning.DynamicWindowApproach import dynamic_window_approach as m

# obstacles for the stuck condition
STUCK_OBSTACLES = -1 * np.array([[-1.0, -1.0],
                                 [0.0, 2.0],
                                 [2.0, 6.0],
                                 [2.0, 8.0],
                                 [3.0, 9.27],
                                 [3.79, 9.39],
                                 [7.25, 8.97],
                                 [7.0, 2.0],
                                 [3.0, 4.0],
                                 [6.0, 5.0],
                                 [3.5, 5.8],
                                 [6.0, 9.0],
                                 [8.8, 9.0],
                                 [5.0, 9.0],
                                 [7.5, 3.0],
                                 [9.0, 8.0],
                                 [5.8, 4.4],
                                 [12.0, 12.0],
                                 [3.0, 2.0],
                                 [13.0, 13.0]
                                 ])
SCENARIOS = ["main1", "main2", "stuck_main"]


def run(goal, robot_type=m.RobotType.circle, **settings):
    """Run main() on a fresh Config; return (trajectory, config, goal)."""
    config = m.Config()
    for name, value in settings.items():
        setattr(config, name, value)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m, "config", config)
        mp.setattr(m, "show_animation", False)
        trajectory = m.main(gx=goal[0], gy=goal[1], robot_type=robot_type)
    return trajectory, config, goal


@pytest.fixture(scope="module")
def main1():
    return run((1.0, 1.0))


@pytest.fixture(scope="module")
def main2():
    return run((1.0, 1.0), robot_type=m.RobotType.rectangle)


@pytest.fixture(scope="module")
def stuck_main():
    # adjusted cost gains, obstacles and goal for the stuck condition
    return run((-5.0, -7.0), to_goal_cost_gain=0.2, obstacle_cost_gain=2.0,
               ob=STUCK_OBSTACLES)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_robot_reaches_goal(scenario, request):
    trajectory, config, goal = request.getfixturevalue(scenario)
    assert np.array_equal(trajectory[0], [0.0, 0.0, math.pi / 8.0, 0.0, 0.0])
    assert math.dist(trajectory[-1, :2], goal) <= config.robot_radius


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_robot_never_collides(scenario, request):
    trajectory, config, _ = request.getfixturevalue(scenario)
    x, y, yaw = (trajectory[:, i, None] for i in range(3))
    dx, dy = config.ob[None, :, 0] - x, config.ob[None, :, 1] - y
    if config.robot_type == m.RobotType.circle:
        assert np.hypot(dx, dy).min() > config.robot_radius
    else:  # how far each obstacle lies outside the rectangle robot
        ahead = dx * np.cos(yaw) + dy * np.sin(yaw)
        left = -dx * np.sin(yaw) + dy * np.cos(yaw)
        outside = np.maximum(np.abs(ahead) - config.robot_length / 2,
                             np.abs(left) - config.robot_width / 2)
        assert outside.min() > 0.0


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_speed_and_turn_rate_stay_within_limits(scenario, request):
    trajectory, config, _ = request.getfixturevalue(scenario)
    v, omega = trajectory[:, 3], trajectory[:, 4]
    assert np.all((config.min_speed <= v) & (v <= config.max_speed))
    assert np.all(np.abs(omega) <= config.max_yaw_rate + 1e-9)
    # speed changes by at most max_accel * dt per step
    assert np.all(np.abs(np.diff(v)) <= config.max_accel * config.dt + 1e-9)


@pytest.mark.parametrize("scenario", [
    "main1",
    "main2",
    pytest.param("stuck_main", marks=pytest.mark.xfail(
        strict=True, raises=AssertionError,
        reason="when the robot stops, the unstick rule sets the turn rate "
               "to -max_delta_yaw_rate, an acceleration [rad/s^2] used as a "
               "rate: a jump ten times what one time step allows"))])
def test_turn_rate_changes_no_faster_than_allowed(scenario, request):
    trajectory, config, _ = request.getfixturevalue(scenario)
    omega = trajectory[:, 4]
    assert np.all(np.abs(np.diff(omega))
                  <= config.max_delta_yaw_rate * config.dt + 1e-9)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_states_follow_the_motion_model(scenario, request):
    trajectory, config, _ = request.getfixturevalue(scenario)
    previous, state = trajectory[:-1], trajectory[1:]
    v, omega = state[:, 3], state[:, 4]  # command applied in each step
    yaw = previous[:, 2] + omega * config.dt
    assert np.allclose(state[:, 2], yaw)
    assert np.allclose(state[:, 0], previous[:, 0] + v * np.cos(yaw) * config.dt)
    assert np.allclose(state[:, 1], previous[:, 1] + v * np.sin(yaw) * config.dt)


if __name__ == '__main__':
    conftest.run_this_test(__file__)
