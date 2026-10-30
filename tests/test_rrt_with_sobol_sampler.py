import math

import pytest

import conftest  # noqa: F401  (adds the project root to sys.path)
import path_checks as pc
from PathPlanning.RRT import rrt_with_sobol_sampler as m


def run_demo():
    """Run main() without animation; return the path and the planner."""
    planners = []
    planning = m.RRTSobol.planning

    def recording_planning(self, *args, **kwargs):
        planners.append(self)
        return planning(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m, "show_animation", False)
        mp.setattr(m.RRTSobol, "planning", recording_planning)
        path = m.main()
    assert len(planners) == 1
    return path, planners[0]


def test1():
    path, planner = run_demo()
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


def test1_same_result_every_run():
    first, _ = run_demo()
    second, _ = run_demo()
    assert first is not None
    assert first == second


if __name__ == '__main__':
    conftest.run_this_test(__file__)
