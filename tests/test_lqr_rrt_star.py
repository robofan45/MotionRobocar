import math

import pytest

import conftest  # Add root path to sys.path
import path_checks as pc
from PathPlanning.LQRRRTStar import lqr_rrt_star as m


@pytest.fixture(scope="module")
def demo():
    """Run main() without animation; return the path and the planner.

    main() runs the planner's 200 iterations (about 26 s), so its result is
    not compared with a second run."""
    planners = []
    planning = m.LQRRRTStar.planning

    def recording_planning(self, *args, **kwargs):
        planners.append(self)
        return planning(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m, "show_animation", False)
        mp.setattr(m.LQRRRTStar, "planning", recording_planning)
        path = m.main()
    assert len(planners) == 1
    return path, planners[0]


def test1(demo):
    path, planner = demo
    start = [planner.start.x, planner.start.y]
    goal = [planner.end.x, planner.end.y]

    assert path is not None, "no path found"
    # the path is listed from the goal back to the start
    assert path[0] == goal
    assert path[-1] == start
    # the tree node that reached the goal is inside the goal region
    assert math.dist(path[1], goal) <= planner.goal_xy_th
    # every sampled point of the LQR trajectories is collision checked
    pc.assert_collision_free(path, planner.obstacle_list, planner.robot_radius,
                             check_spacing=pc.max_step(path))
    assert pc.path_length(path) >= math.dist(start, goal)


@pytest.mark.xfail(
    strict=True, raises=AssertionError,
    reason="rewire() swaps a node for the end of an LQR trajectory steered "
           "towards it, which stops short of it, but the node's children keep "
           "trajectories that start at the old position, so the path jumps "
           "0.13 m at one junction")
def test_path_trajectories_start_at_their_parent(demo):
    _, planner = demo
    node = planner.node_list[planner.search_best_goal_node()]
    while node.parent:
        gap = math.hypot(node.path_x[0] - node.parent.x,
                         node.path_y[0] - node.parent.y)
        assert gap < 1e-9
        node = node.parent


def test_main_passes_max_iter_to_the_planner():
    max_iters = []

    def planning_stub(self, animation=True, search_until_max_iter=True):
        max_iters.append(self.max_iter)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m, "show_animation", False)
        mp.setattr(m.LQRRRTStar, "planning", planning_stub)
        m.main(maxIter=5)
    assert max_iters == [5]


if __name__ == '__main__':
    conftest.run_this_test(__file__)
