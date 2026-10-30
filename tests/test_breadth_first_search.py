"""
Answer checks for the breadth-first grid planner.

Breadth-first search expands cells in order of their number of moves from
the start, so on the 8-connected grid it returns a path with the fewest moves
(not the shortest distance: a diagonal move counts the same as a straight
one). The reference is an independent move count on the planner's own grid.
"""
from collections import deque
import math

import pytest

import conftest
import grid_planner_checks as chk
from PathPlanning.BreadthFirstSearch import breadth_first_search as m

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


def fewest_moves(view, start, goal):
    """Independent breadth-first move count between two cells, or None."""
    source, target = chk.cell(view, *start), chk.cell(view, *goal)
    moves = {source: 0}
    queue = deque([source])
    while queue:
        current = queue.popleft()
        if current == target:
            return moves[current]
        for dx, dy, _ in chk.MOTIONS:
            nxt = (current[0] + dx, current[1] + dy)
            if nxt not in moves and chk.is_free(view, *nxt):
                moves[nxt] = moves[current] + 1
                queue.append(nxt)
    return None


def assert_clear_of_obstacles(xs, ys, ox, oy, radius):
    """Every point is farther than the robot radius from every obstacle."""
    for x, y in zip(xs, ys):
        nearest = min(math.hypot(x - o_x, y - o_y) for o_x, o_y in zip(ox, oy))
        assert nearest > radius, f"({x}, {y}) is {nearest:.2f} m from an obstacle"


def check_fewest_moves_path(planner, start, goal):
    """Plan, then check the path is valid (inside the grid, one cell per
    move, no blocked cell) and uses the fewest moves; return (rx, ry)."""
    rx, ry = planner.planning(*start, *goal)
    view = GridView(planner)
    chk.assert_valid_path(view, rx, ry, start, goal)
    assert len(rx) - 1 == fewest_moves(view, start, goal)
    return rx, ry


def odd_width_with_gap_obstacles():
    """73 m wide box (36.5 cells of 2 m, rounded down to 36) with a wall at
    y = 14 that leaves a gap only in the last grid column."""
    ox, oy = chk.odd_extent_obstacles()
    ox += [float(x) for x in range(69)]
    oy += [14.0] * 69
    return ox, oy


def test_1():
    m.main()


def test_demo_path_is_valid_and_has_fewest_moves():
    ox, oy = chk.demo_obstacles()
    planner = m.BreadthFirstSearchPlanner(ox, oy, 2.0, ROBOT_RADIUS)
    rx, ry = check_fewest_moves_path(planner, (10.0, 10.0), (50.0, 50.0))
    assert_clear_of_obstacles(rx, ry, ox, oy, ROBOT_RADIUS)


# chk.random_scenario(7) walls the goal off: see the unreachable goal test
@pytest.mark.parametrize("seed", [s for s in range(20) if s != 7])
def test_random_maps_give_valid_paths_with_fewest_moves(seed):
    ox, oy, start, goal = chk.random_scenario(seed)
    planner = m.BreadthFirstSearchPlanner(ox, oy, 1.0, 0.5)
    rx, ry = check_fewest_moves_path(planner, start, goal)
    # the start itself may lie inside an inflated obstacle (seed 6)
    assert_clear_of_obstacles(rx[:-1], ry[:-1], ox, oy, 0.5)


@pytest.mark.xfail(strict=True, raises=KeyError,
                   reason="raises KeyError(-1) instead of returning an "
                          "empty path when the goal cannot be reached")
def test_unreachable_goal_returns_empty_path():
    ox, oy = chk.enclosed_goal_obstacles()
    planner = m.BreadthFirstSearchPlanner(ox, oy, 1.0, 0.5)
    assert planner.planning(3.0, 3.0, 15.0, 15.0) == ([], [])


@pytest.mark.xfail(strict=True, raises=KeyError,
                   reason="raises KeyError(-1) when start and goal are the "
                          "same cell")
def test_start_equals_goal():
    ox, oy = chk.boundary(20)
    planner = m.BreadthFirstSearchPlanner(ox, oy, 1.0, 0.5)
    assert planner.planning(7.0, 7.0, 7.0, 7.0) == ([7.0], [7.0])


@pytest.mark.xfail(strict=True, raises=IndexError,
                   reason="when the map width in cells rounds down, the "
                          "search steps one column past the obstacle map")
def test_map_width_that_rounds_down():
    ox, oy = odd_width_with_gap_obstacles()
    planner = m.BreadthFirstSearchPlanner(ox, oy, 2.0, 1.0)
    check_fewest_moves_path(planner, (4.0, 4.0), (60.0, 26.0))


if __name__ == '__main__':
    conftest.run_this_test(__file__)
