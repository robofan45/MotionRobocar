"""
Answer checks for the Hybrid A* demo.

The path is a list of car poses (x, y, yaw) about 0.1 m apart. It must start
and end at the requested poses, be drivable (the car moves along its heading,
forward or backward as flagged, and never turns tighter than its minimum
turning radius), and the car rectangle must never touch an obstacle point;
that last check is a brute force one, independent of the planner's k-d tree.
"""
import math

import numpy as np
import pytest

import conftest
from PathPlanning.HybridAStar import car
from PathPlanning.HybridAStar import hybrid_a_star as m

START = (10.0, 10.0, np.deg2rad(90.0))  # from main()
GOAL = (50.0, 50.0, np.deg2rad(-90.0))
MAX_CURVATURE = math.tan(car.MAX_STEER) / car.WB  # [1/m]
# heading change over one sample at the tightest turn
MAX_TURN_PER_STEP = MAX_CURVATURE * m.MOTION_RESOLUTION  # [rad]


@pytest.fixture(scope="module")
def path():
    m.show_animation = False
    return m.main()


def demo_obstacles():
    """Obstacle points of main()."""
    ox = [*range(60), *[60.0] * 60, *range(61), *[0.0] * 61,
          *[20.0] * 40, *[40.0] * 40]
    oy = [*[0.0] * 60, *range(60), *[60.0] * 61, *range(61),
          *range(40), *[60.0 - i for i in range(40)]]
    return np.array(ox, dtype=float), np.array(oy, dtype=float)


def test_path_starts_and_ends_at_the_requested_poses(path):
    assert (path.x_list[0], path.y_list[0], path.yaw_list[0]) == \
        pytest.approx(START, abs=1e-9)
    assert (path.x_list[-1], path.y_list[-1], path.yaw_list[-1]) == \
        pytest.approx(GOAL, abs=1e-6)


def test_path_is_continuous_and_drivable(path):
    x, y, yaw = (np.array(v) for v in (path.x_list, path.y_list,
                                       path.yaw_list))
    forward = np.array(path.direction_list[1:])
    step = np.hypot(np.diff(x), np.diff(y))
    turn = np.abs((np.diff(yaw) + np.pi) % (2 * np.pi) - np.pi)
    assert step.max() <= m.MOTION_RESOLUTION + 1e-9
    # samples repeat where Reeds-Shepp segments join: the pose must not jump
    still = step < 1e-9
    assert np.all(turn[still] < 1e-9)
    moving = ~still
    # the car moves along its heading, the way the direction flag says
    along = (np.diff(x) * np.cos(yaw[:-1]) +
             np.diff(y) * np.sin(yaw[:-1]))[moving] / step[moving]
    assert np.all(np.abs(along) >= math.cos(2 * MAX_TURN_PER_STEP))
    assert np.array_equal(along > 0, forward[moving])
    # never tighter than the minimum turning radius (1 % allows for the
    # chord of a sampled arc being shorter than the arc)
    assert np.all(turn[moving] <= 1.01 * MAX_CURVATURE * step[moving])


def test_car_never_touches_an_obstacle(path):
    ox, oy = demo_obstacles()
    x, y, yaw = (np.array(v)[:, None] for v in (path.x_list, path.y_list,
                                               path.yaw_list))
    dx, dy = ox[None, :] - x, oy[None, :] - y
    ahead = dx * np.cos(yaw) + dy * np.sin(yaw)  # obstacle in car frame
    left = -dx * np.sin(yaw) + dy * np.cos(yaw)
    # how far each obstacle point lies outside the car rectangle
    outside = np.maximum.reduce([ahead - car.LF, -car.LB - ahead,
                                 np.abs(left) - car.W / 2.0])
    assert outside.min() > 0.5  # measured 0.91 m


def test_path_stays_inside_the_walls(path):
    assert 0.0 < min(path.x_list) and max(path.x_list) < 60.0
    assert 0.0 < min(path.y_list) and max(path.y_list) < 60.0


if __name__ == '__main__':
    conftest.run_this_test(__file__)
