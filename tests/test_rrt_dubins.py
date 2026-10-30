import pytest

import conftest  # noqa: F401  (adds the project root to sys.path)
import path_checks as pc
from PathPlanning.RRTDubins import rrt_dubins as m


def run_demo():
    """Run main() without animation; return the path and the planner."""
    planners = []
    planning = m.RRTDubins.planning

    def recording_planning(self, *args, **kwargs):
        planners.append(self)
        return planning(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m, "show_animation", False)
        mp.setattr(m.RRTDubins, "planning", recording_planning)
        path = m.main()
    assert len(planners) == 1
    return path, planners[0]


def test1():
    path, planner = run_demo()
    pc.assert_dubins_path(path, planner)


def test1_same_result_every_run():
    first, _ = run_demo()
    second, _ = run_demo()
    assert first is not None
    assert first == second


if __name__ == '__main__':
    conftest.run_this_test(__file__)
