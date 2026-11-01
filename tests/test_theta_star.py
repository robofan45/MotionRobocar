"""
Answer checks for the Theta* any-angle grid planner.

Theta* lets a cell take its grandparent as parent when the two can see each
other, so the path is a list of waypoints joined by straight segments. The
checks: the waypoints are free cells, every segment passes the planner's
line-of-sight rule (a Bresenham line through free cells, re-implemented
below), and the path is never longer than the shortest 8-connected grid path
from the independent reference solver. That bound holds because a straight
segment from the grandparent is never longer than the two moves it replaces
(triangle inequality), so the usual A* argument still applies.
"""
import math

import pytest

import conftest
import grid_planner_checks as chk
from PathPlanning.ThetaStar import theta_star as m

ROBOT_RADIUS = 1.0  # [m] value used by main()

DEMO_PATH = [(50.0, 50.0), (42.0, 20.0), (40.0, 18.0), (20.0, 42.0),
             (18.0, 40.0), (10.0, 10.0)]


@pytest.fixture(autouse=True)
def no_animation():
    m.show_animation = False


def line_of_sight_cells(a, b):
    """Cells on the planner's Bresenham line from cell a to cell b."""
    (x0, y0), (x1, y1) = a, b
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    cells = [(x0, y0)]
    while (x0, y0) != (x1, y1):
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
        cells.append((x0, y0))
    return cells


def assert_clear_of_obstacles(xs, ys, ox, oy, radius):
    """Every point is farther than the robot radius from every obstacle."""
    for x, y in zip(xs, ys):
        nearest = min(math.hypot(x - o_x, y - o_y) for o_x, o_y in zip(ox, oy))
        assert nearest > radius, f"({x}, {y}) is {nearest:.2f} m from an obstacle"


def check_theta_star_path(planner, start, goal):
    """Plan and check the any-angle path; return (rx, ry) (goal first)."""
    rx, ry = planner.planning(*start, *goal)
    cells = [chk.cell(planner, x, y) for x, y in zip(rx, ry)]
    assert cells[0] == chk.cell(planner, *goal), "path must begin at goal"
    assert cells[-1] == chk.cell(planner, *start), "path must end at start"
    for x, y in zip(rx, ry):  # waypoints are grid cell positions
        ix, iy = chk.cell(planner, x, y)
        assert x == pytest.approx(planner.min_x + ix * planner.resolution)
        assert y == pytest.approx(planner.min_y + iy * planner.resolution)
    # the path is listed goal first; each link was checked parent -> child
    for child, parent in zip(cells, cells[1:]):
        assert child != parent, "repeated waypoint"
        for c in line_of_sight_cells(parent, child):
            # the start cell itself is not checked (it may be inside an
            # inflated obstacle on random maps)
            assert c == cells[-1] or chk.is_free(planner, *c), \
                f"segment {parent} -> {child} crosses blocked cell {c}"
    length = chk.path_cost(planner, rx, ry)  # in cells
    assert length <= chk.reference_cost(planner, *start, *goal) + 1e-9
    assert length >= math.dist(cells[0], cells[-1]) - 1e-9
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


def test_demo_path_is_valid_short_and_unchanged():
    ox, oy = chk.demo_obstacles(as_float=False)
    planner = m.ThetaStarPlanner(ox, oy, 2.0, ROBOT_RADIUS)
    rx, ry = check_theta_star_path(planner, (10.0, 10.0), (50.0, 50.0))
    assert_clear_of_obstacles(rx, ry, ox, oy, ROBOT_RADIUS)
    assert list(zip(rx, ry)) == DEMO_PATH


# chk.random_scenario(7) walls the goal off: see the unreachable goal test
@pytest.mark.parametrize("seed", [s for s in range(20) if s != 7])
def test_random_maps_give_valid_paths_no_longer_than_grid_paths(seed):
    ox, oy, start, goal = chk.random_scenario(seed)
    planner = m.ThetaStarPlanner(ox, oy, 1.0, 0.5)
    rx, ry = check_theta_star_path(planner, start, goal)
    # the start itself may lie inside an inflated obstacle (seed 6)
    assert_clear_of_obstacles(rx[:-1], ry[:-1], ox, oy, 0.5)


@pytest.mark.parametrize("seed", range(6))
def test_without_theta_star_it_finds_shortest_grid_paths(seed, monkeypatch):
    monkeypatch.setattr(m, "use_theta_star", False)  # plain A*
    ox, oy, start, goal = chk.random_scenario(seed)
    planner = m.ThetaStarPlanner(ox, oy, 1.0, 0.5)
    chk.check_planner(planner, start, goal)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="returns a path made of the goal alone instead of "
                          "an empty path when the goal cannot be reached")
def test_unreachable_goal_returns_empty_path():
    ox, oy = chk.enclosed_goal_obstacles()
    planner = m.ThetaStarPlanner(ox, oy, 1.0, 0.5)
    assert planner.planning(3.0, 3.0, 15.0, 15.0) == ([], [])


def test_start_equals_goal():
    ox, oy = chk.boundary(20)
    planner = m.ThetaStarPlanner(ox, oy, 1.0, 0.5)
    assert planner.planning(7.0, 7.0, 7.0, 7.0) == ([7.0], [7.0])


def test_map_width_that_rounds_down():
    ox, oy = odd_width_with_gap_obstacles()
    planner = m.ThetaStarPlanner(ox, oy, 2.0, 1.0)
    check_theta_star_path(planner, (4.0, 4.0), (60.0, 26.0))


if __name__ == '__main__':
    conftest.run_this_test(__file__)
