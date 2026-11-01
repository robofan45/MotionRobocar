"""
Answer checks for the greedy best-first grid planner.

Greedy best-first search follows the heuristic only, so it is not optimal:
only the validity of the path is checked (it joins start and goal, moves one
cell at a time inside the grid and never enters a cell closer to an obstacle
than the robot radius).
"""
import math

import pytest

import conftest
import grid_planner_checks as chk
from PathPlanning.GreedyBestFirstSearch import greedy_best_first_search as m

ROBOT_RADIUS = 1.0  # [m] value used by main()


@pytest.fixture(autouse=True)
def no_animation():
    m.show_animation = False


class GridView:
    """The planner's grid under the attribute names grid_planner_checks uses."""

    def __init__(self, planner):
        self.resolution = planner.reso
        self.min_x, self.min_y = planner.minx, planner.miny
        self.x_width, self.y_width = planner.xwidth, planner.ywidth
        self.obstacle_map = planner.obmap


def drop_repeats(xs, ys):
    """Remove consecutive duplicate points (see the repeated goal test)."""
    points = [(x, y) for x, y in zip(xs, ys)]
    kept = points[:1] + [b for a, b in zip(points, points[1:]) if a != b]
    return [x for x, _ in kept], [y for _, y in kept]


def assert_clear_of_obstacles(xs, ys, ox, oy, radius):
    """Every point is farther than the robot radius from every obstacle."""
    for x, y in zip(xs, ys):
        nearest = min(math.hypot(x - o_x, y - o_y) for o_x, o_y in zip(ox, oy))
        assert nearest > radius, f"({x}, {y}) is {nearest:.2f} m from an obstacle"


def check_valid_path(planner, start, goal, path=None):
    """Plan (unless the path is given), then check the path is valid (inside
    the grid, one cell per move, no blocked cell), ignoring the repeated
    goal; return (rx, ry)."""
    rx, ry = drop_repeats(*(path or planner.planning(*start, *goal)))
    chk.assert_valid_path(GridView(planner), rx, ry, start, goal)
    return rx, ry


def odd_width_with_gap_obstacles():
    """73 m wide box (36.5 cells of 2 m, rounded down to 36) with a wall at
    y = 14 that leaves a gap only in the last grid column."""
    ox, oy = chk.odd_extent_obstacles()
    ox += [float(x) for x in range(69)]
    oy += [14.0] * 69
    return ox, oy


@pytest.fixture(scope="module")
def demo():
    """Planner and raw path of the demo scenario in main()."""
    m.show_animation = False
    ox, oy = chk.demo_obstacles(as_float=False)
    planner = m.BestFirstSearchPlanner(ox, oy, 2.0, ROBOT_RADIUS)
    return ox, oy, planner, planner.planning(10.0, 10.0, 50.0, 50.0)


def test_1():
    m.main()


def test_demo_path_is_valid(demo):
    ox, oy, planner, path = demo
    rx, ry = check_valid_path(planner, (10.0, 10.0), (50.0, 50.0), path)
    assert_clear_of_obstacles(rx, ry, ox, oy, ROBOT_RADIUS)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="the goal is listed twice at the start of the path")
def test_path_has_no_repeated_points(demo):
    rx, ry = demo[3]
    assert (rx, ry) == drop_repeats(rx, ry)


# chk.random_scenario(7) walls the goal off: see the unreachable goal test
@pytest.mark.parametrize("seed", [s for s in range(20) if s != 7])
def test_random_maps_give_valid_paths(seed):
    ox, oy, start, goal = chk.random_scenario(seed)
    planner = m.BestFirstSearchPlanner(ox, oy, 1.0, 0.5)
    rx, ry = check_valid_path(planner, start, goal)
    # the start itself may lie inside an inflated obstacle (seed 6)
    assert_clear_of_obstacles(rx[:-1], ry[:-1], ox, oy, 0.5)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="returns a path that jumps from the goal to the last "
                          "searched cell, through the walls, instead of an "
                          "empty path")
def test_unreachable_goal_returns_empty_path():
    ox, oy = chk.enclosed_goal_obstacles()
    planner = m.BestFirstSearchPlanner(ox, oy, 1.0, 0.5)
    assert planner.planning(3.0, 3.0, 15.0, 15.0) == ([], [])


def test_map_width_that_rounds_down():
    ox, oy = odd_width_with_gap_obstacles()
    planner = m.BestFirstSearchPlanner(ox, oy, 2.0, 1.0)
    check_valid_path(planner, (4.0, 4.0), (60.0, 26.0))


if __name__ == '__main__':
    conftest.run_this_test(__file__)
