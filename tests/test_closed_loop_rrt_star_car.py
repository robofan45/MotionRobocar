import math

import numpy as np
import pytest

import conftest
import path_checks as pc
from PathPlanning.ClosedLoopRRTStar import closed_loop_rrt_star_car as m
from PathPlanning.ClosedLoopRRTStar import pure_pursuit
from PathPlanning.ClosedLoopRRTStar import unicycle_model

# the demo's own goal (100 iterations) takes about 26 s, so plan to a goal
# 1 m ahead instead; with the demo's seed, 3 iterations give a trajectory
# the car can track (with 5, every candidate fails the final heading check)
SCENARIO = dict(gx=1.0, gy=0.0, gyaw=0.0, max_iter=3)


def run_demo():
    """Run main() without animation; return its result and the planner."""
    planners = []
    planning = m.ClosedLoopRRTStar.planning

    def recording_planning(self, *args, **kwargs):
        planners.append(self)
        return planning(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m, "show_animation", False)
        mp.setattr(m.ClosedLoopRRTStar, "planning", recording_planning)
        result = m.main(**SCENARIO)
    assert len(planners) == 1
    return result, planners[0]


@pytest.fixture(scope="module")
def demo():
    return run_demo()


def test_1(demo):
    (found, x, y, yaw, v, t, a, d), planner = demo
    start, goal = planner.start, planner.end
    assert found, "no trackable path found"

    # planning() appends the goal pose to the simulated car states
    assert (x[-1], y[-1], yaw[-1]) == (goal.x, goal.y, goal.yaw)
    x, y, yaw = np.array(x[:-1]), np.array(y[:-1]), np.array(yaw[:-1])
    v, t, a, d = np.array(v), np.array(t), np.array(a), np.array(d)
    assert len(x) == len(y) == len(yaw) == len(v) == len(t) == len(a) == len(d)
    assert (x[0], y[0], yaw[0], v[0]) == (start.x, start.y, start.yaw, 0.0)

    # the simulation stops when the car comes within goal_dis of the goal;
    # that last step (|v| * dt long) is not stored
    dt = unicycle_model.dt
    assert math.hypot(x[-1] - goal.x, y[-1] - goal.y) <= \
        pure_pursuit.goal_dis + abs(v[-1]) * dt
    # the planner accepts final headings up to 10 * yaw_th off
    assert abs(pc.angle_diff(yaw[-1], goal.yaw)) < 10 * planner.yaw_th

    np.testing.assert_allclose(np.diff(t), dt)
    assert np.max(np.abs(v)) <= planner.target_speed
    assert np.max(np.abs(a)) <= unicycle_model.accel_max
    assert np.max(np.abs(d)) <= unicycle_model.steer_max

    # each step follows the car model: move |v| dt along the heading, turn
    # by v / L * tan(steer) * dt, then change speed by a * dt
    np.testing.assert_allclose(np.diff(x), v[:-1] * np.cos(yaw[:-1]) * dt,
                               atol=1e-12)
    np.testing.assert_allclose(np.diff(y), v[:-1] * np.sin(yaw[:-1]) * dt,
                               atol=1e-12)
    turn = np.angle(np.exp(1j * np.diff(yaw)))
    np.testing.assert_allclose(
        turn, v[:-1] / unicycle_model.L * np.tan(d[1:]) * dt, atol=1e-12)
    np.testing.assert_allclose(np.diff(v), a[1:] * dt, atol=1e-12)

    # the tracked trajectory itself is collision checked point by point
    trajectory = list(zip(x, y))
    pc.assert_collision_free(trajectory, planner.obstacle_list,
                             planner.robot_radius,
                             check_spacing=pc.max_step(trajectory))


def test_1_same_result_every_run(demo):
    again, _ = run_demo()
    assert demo[0][0]
    assert again == demo[0]


if __name__ == '__main__':
    conftest.run_this_test(__file__)
