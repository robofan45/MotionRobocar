import math

import numpy as np
import pytest
import scipy.linalg as la

import conftest  # Add root path to sys.path
from PathPlanning.LQRPlanner import lqr_planner as m


def run_demo():
    """Run main() without animation; return (start, goal, rx, ry) for every
    goal it plans to."""
    runs = []
    planning = m.LQRPlanner.lqr_planning

    def recording_planning(self, sx, sy, gx, gy, show_animation=True):
        rx, ry = planning(self, sx, sy, gx, gy, show_animation=show_animation)
        runs.append(((sx, sy), (gx, gy), rx, ry))
        return rx, ry

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m, "SHOW_ANIMATION", False)
        mp.setattr(m.LQRPlanner, "lqr_planning", recording_planning)
        m.main()
    return runs


def test_1():
    runs = run_demo()
    planner = m.LQRPlanner()
    assert len(runs) == 10
    for (sx, sy), (gx, gy), rx, ry in runs:
        assert rx and len(rx) == len(ry), "no path found"
        assert (rx[0], ry[0]) == (sx, sy)
        # stops at the first point within GOAL_DIST of the goal
        dists = [math.hypot(x - gx, y - gy) for x, y in zip(rx, ry)]
        assert dists[-1] <= planner.GOAL_DIST
        assert min(dists[:-1]) > planner.GOAL_DIST
        assert sum(math.hypot(x1 - x0, y1 - y0) for x0, y0, x1, y1
                   in zip(rx, ry, rx[1:], ry[1:])) >= dists[0] - dists[-1]


def test_1_same_result_every_run():
    first = run_demo()
    assert first == run_demo()


def test_closed_loop_is_stable():
    planner = m.LQRPlanner()
    A, B = planner.get_system_model()
    _, _, eig_values = planner.dlqr(A, B, np.eye(2), np.eye(1))
    assert np.all(np.abs(eig_values) < 1.0)


def test_solve_dare_matches_scipy():
    planner = m.LQRPlanner()
    A, B = planner.get_system_model()
    Q, R = np.eye(2), np.eye(1)
    # the iteration stops once an update changes X by less than EPS
    np.testing.assert_allclose(planner.solve_dare(A, B, Q, R),
                               la.solve_discrete_are(A, B, Q, R),
                               atol=2 * planner.EPS)


if __name__ == '__main__':
    conftest.run_this_test(__file__)
