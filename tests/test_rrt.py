import math

import pytest

import conftest  # noqa: F401  (adds the project root to sys.path)
import path_checks as pc
from PathPlanning.RRT import rrt as m
from PathPlanning.RRT import rrt_with_pathsmoothing as m1


def run_demo(module):
    """Run module.main() without animation; return its result and planner."""
    planners = []
    planning = module.RRT.planning

    def recording_planning(self, *args, **kwargs):
        planners.append(self)
        return planning(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(module, "show_animation", False)
        mp.setattr(module.RRT, "planning", recording_planning)
        result = module.main()
    assert len(planners) == 1
    return result, planners[0]


def check_rrt_path(path, planner):
    start = [planner.start.x, planner.start.y]
    goal = [planner.end.x, planner.end.y]
    assert path is not None, "no path found"
    # the path is listed from the goal back to the start
    assert path[0] == goal
    assert path[-1] == start
    # collisions are only checked every path_resolution metres
    pc.assert_collision_free(path, planner.obstacle_list, planner.robot_radius,
                             check_spacing=planner.path_resolution)
    # steer() may snap one path_resolution beyond expand_dis onto its target
    max_step = planner.expand_dis + planner.path_resolution
    for a, b in zip(path[:-1], path[1:]):
        assert math.dist(a, b) <= max_step + 1e-9
    assert pc.path_length(path) >= math.dist(start, goal)


def test1():
    path, planner = run_demo(m)
    check_rrt_path(path, planner)


def test1_same_result_every_run():
    first, _ = run_demo(m)
    second, _ = run_demo(m)
    assert first is not None
    assert first == second


def test2():
    (path, smoothed), planner = run_demo(m1)
    check_rrt_path(path, planner)

    assert smoothed[0] == path[0]
    assert smoothed[-1] == path[-1]
    # line_collision_check() samples each shortcut every 0.2 m
    pc.assert_collision_free(smoothed, planner.obstacle_list,
                             planner.robot_radius, check_spacing=0.2)
    # a shortcut never makes the path longer
    assert pc.path_length(smoothed) <= pc.path_length(path) + 1e-9
    assert pc.path_length(smoothed) >= math.dist(path[0], path[-1])


def test2_same_result_every_run():
    first, _ = run_demo(m1)
    second, _ = run_demo(m1)
    assert first[0] is not None
    assert first == second


if __name__ == '__main__':
    conftest.run_this_test(__file__)
