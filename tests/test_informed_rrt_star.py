import pytest

import conftest  # noqa: F401  (adds the project root to sys.path)
import path_checks as pc
from PathPlanning.InformedRRTStar import informed_rrt_star as m


def run_demo():
    """Run main() without animation; return the path and the planner."""
    planners = []
    search = m.InformedRRTStar.informed_rrt_star_search

    def recording_search(self, *args, **kwargs):
        planners.append(self)
        return search(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m, "show_animation", False)
        mp.setattr(m.InformedRRTStar, "informed_rrt_star_search",
                   recording_search)
        path = m.main()
    assert len(planners) == 1
    return path, planners[0]


@pytest.fixture(scope="module")
def demo():
    return run_demo()


def test1(demo):
    path, planner = demo
    start = [planner.start.x, planner.start.y]
    goal = [planner.goal.x, planner.goal.y]

    assert path is not None, "no path found"
    # the path is listed from the goal back to the start
    assert path[0] == goal
    assert path[-1] == start
    # tree edges are checked exactly against the circles, so no allowance
    pc.assert_collision_free(path, planner.obstacle_list)

    # the straight line runs through obstacle (3, 6, 1); no collision free
    # path can be shorter than the way around that circle (11.367 m)
    detour = pc.shortest_detour(start, goal, (3, 6, 1))
    length = pc.path_length(path)
    assert length >= detour
    # informed sampling gets close to that bound: the seeded run is
    # 0.088 m above it, so allow twice that
    assert length <= detour + 0.18


def test1_same_result_every_run(demo):
    again, _ = run_demo()
    assert demo[0] is not None
    assert again == demo[0]


if __name__ == '__main__':
    conftest.run_this_test(__file__)
