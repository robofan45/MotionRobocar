import random

import pytest

import conftest  # noqa: F401  (adds the project root to sys.path)
import path_checks as pc
from PathPlanning.BatchInformedRRTStar import batch_informed_rrt_star as m

# connect() checks int(10 * d) evenly spaced points, ends included, on an
# edge of length d: at most 0.3 m apart once d >= 0.2 m
CHECK_SPACING = 0.3


def run_demo(max_iter):
    """Run main() without animation; return the path and the planner."""
    planners = []
    plan = m.BITStar.plan

    def recording_plan(self, *args, **kwargs):
        planners.append(self)
        return plan(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m, "show_animation", False)
        mp.setattr(m.BITStar, "plan", recording_plan)
        path = m.main(maxIter=max_iter)
    assert len(planners) == 1
    return path, planners[0]


def tree_edges(planner):
    to_xy = planner.tree.node_id_to_real_world_coord
    return [[to_xy(v), to_xy(x)] for v, x in planner.tree.edges]


def test_1():
    # the demo needs 51 iterations (about 1 s) before its first path, and
    # the full 80 take about 6 s, so check the tree it grows in 10
    path, planner = run_demo(max_iter=10)
    edges = tree_edges(planner)
    assert edges, "the tree did not grow"
    for edge in edges:
        pc.assert_collision_free(edge, planner.obstacleList,
                                 check_spacing=CHECK_SPACING)

    again, planner_again = run_demo(max_iter=10)
    assert again == path
    assert planner_again.samples == planner.samples
    assert tree_edges(planner_again) == edges


def test_path_around_one_obstacle():
    start, goal, obstacle = [5, 5], [9, 5], (7, 5, 0.5)
    random.seed(42)
    planner = m.BITStar(start=start, goal=goal, obstacleList=[obstacle],
                        randArea=[-2, 15], maxIter=15)
    path = planner.plan(animation=False)

    assert path, "no path found"
    assert path[0] == start
    assert path[-1] == goal
    pc.assert_collision_free(path, [obstacle], check_spacing=CHECK_SPACING)
    # no collision free path is shorter than the way around the circle
    assert pc.path_length(path) >= pc.shortest_detour(start, goal, obstacle)


if __name__ == '__main__':
    conftest.run_this_test(__file__)
