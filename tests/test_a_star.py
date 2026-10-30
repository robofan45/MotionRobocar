import pytest

import conftest
import grid_planner_checks as chk
from PathPlanning.AStar import a_star as m

DEMO_PATH = [
    (50.0, 50.0), (48.0, 48.0), (48.0, 46.0), (48.0, 44.0), (46.0, 42.0),
    (46.0, 40.0), (44.0, 38.0), (44.0, 36.0), (42.0, 34.0), (42.0, 32.0),
    (42.0, 30.0), (42.0, 28.0), (42.0, 26.0), (42.0, 24.0), (42.0, 22.0),
    (42.0, 20.0), (40.0, 18.0), (38.0, 20.0), (38.0, 22.0), (38.0, 24.0),
    (36.0, 26.0), (34.0, 28.0), (32.0, 30.0), (30.0, 32.0), (28.0, 34.0),
    (26.0, 36.0), (24.0, 38.0), (22.0, 40.0), (20.0, 42.0), (18.0, 40.0),
    (18.0, 38.0), (18.0, 36.0), (18.0, 34.0), (18.0, 32.0), (18.0, 30.0),
    (18.0, 28.0), (18.0, 26.0), (18.0, 24.0), (18.0, 22.0), (18.0, 20.0),
    (16.0, 18.0), (14.0, 16.0), (12.0, 14.0), (12.0, 12.0), (10.0, 10.0)]


@pytest.fixture(autouse=True)
def no_animation():
    m.show_animation = False


def test_1():
    m.main()


def test_demo_path_is_valid_shortest_and_unchanged():
    ox, oy = chk.demo_obstacles(as_float=False)
    planner = m.AStarPlanner(ox, oy, 2.0, 1.0)
    rx, ry = chk.check_planner(planner, (10.0, 10.0), (50.0, 50.0))
    assert list(zip(rx, ry)) == DEMO_PATH


@pytest.mark.parametrize("seed", range(30))
def test_random_maps_give_shortest_valid_paths(seed):
    ox, oy, start, goal = chk.random_scenario(seed, shift="random")
    planner = m.AStarPlanner(ox, oy, 1.0, 0.5)
    chk.check_planner(planner, start, goal, obstacles=(ox, oy))


def test_coarse_grid_and_bigger_robot():
    ox, oy, start, goal = chk.random_scenario(1234, size=40)
    planner = m.AStarPlanner(ox, oy, 2.0, 1.5)
    chk.check_planner(planner, start, goal)


def test_unreachable_goal_returns_empty_path():
    ox, oy = chk.enclosed_goal_obstacles()
    planner = m.AStarPlanner(ox, oy, 1.0, 0.5)
    assert planner.planning(3.0, 3.0, 15.0, 15.0) == ([], [])


def test_goal_inside_wall_returns_empty_path():
    ox, oy = chk.boundary(20)
    planner = m.AStarPlanner(ox, oy, 1.0, 0.5)
    assert planner.planning(5.0, 5.0, 0.0, 10.0) == ([], [])


def test_start_equals_goal():
    ox, oy = chk.boundary(20)
    planner = m.AStarPlanner(ox, oy, 1.0, 0.5)
    assert planner.planning(7.0, 7.0, 7.0, 7.0) == ([7.0], [7.0])


def test_map_width_that_rounds_down_does_not_crash():
    ox, oy = chk.odd_extent_obstacles()
    planner = m.AStarPlanner(ox, oy, 2.0, 1.0)
    chk.check_planner(planner, (4.0, 4.0), (72.0, 26.0))


def test_last_partial_column_is_reachable():
    # 73 m / 2 m = 36.5 cells: the last half cell must still exist,
    # and (72, 26) is 1 m from the nearest wall, more than the 0.5 m radius
    ox, oy = chk.odd_extent_obstacles()
    planner = m.AStarPlanner(ox, oy, 2.0, 0.5)
    rx, ry = chk.check_planner(planner, (4.0, 4.0), (72.0, 26.0),
                               obstacles=(ox, oy))
    assert rx, "goal next to the far wall must be reachable"


def test_map_away_from_origin_keeps_the_start():
    # the map's corner is at (1, 1), not at the origin
    planner = m.AStarPlanner([1.0, 11.0], [1.0, 11.0], 1.0, 0.0)
    rx, ry = chk.check_planner(planner, (1.0, 2.0), (5.0, 2.0))
    assert (rx[-1], ry[-1]) == (1.0, 2.0)


def test_start_or_goal_off_the_map_returns_empty_path():
    ox, oy = chk.boundary(20)
    planner = m.AStarPlanner(ox, oy, 1.0, 0.5)
    assert planner.planning(-5.0, 5.0, 10.0, 10.0) == ([], [])
    assert planner.planning(5.0, 5.0, 25.0, 10.0) == ([], [])


@pytest.mark.parametrize("ox, oy, resolution", [
    ([0.0, 10.0], [0.0, 10.0], 0.0),
    ([0.0, 10.0], [0.0, 10.0], -1.0),
    ([0.0, 10.0], [0.0, 10.0], float("nan")),
    ([], [], 1.0),
    ([0.0, 10.0], [0.0], 1.0),
    ([0.0, float("nan"), 10.0], [0.0, 5.0, 10.0], 1.0),
    ([0.0, float("inf")], [0.0, 10.0], 1.0),
])
def test_invalid_map_inputs_raise(ox, oy, resolution):
    with pytest.raises(ValueError):
        m.AStarPlanner(ox, oy, resolution, 0.5)


def test_non_finite_start_or_goal_raises():
    ox, oy = chk.boundary(20)
    planner = m.AStarPlanner(ox, oy, 1.0, 0.5)
    with pytest.raises(ValueError):
        planner.planning(float("nan"), 5.0, 10.0, 10.0)
    with pytest.raises(ValueError):
        planner.planning(5.0, 5.0, 10.0, float("inf"))


def test_animation_has_a_legend(monkeypatch):
    labels = chk.run_with_animation(m, monkeypatch)
    assert {"Obstacle", "Start", "Goal", "Searched cell", "Path"} <= labels


if __name__ == '__main__':
    conftest.run_this_test(__file__)
