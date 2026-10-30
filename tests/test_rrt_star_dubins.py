import pytest

import conftest  # noqa: F401  (adds the project root to sys.path)
import path_checks as pc
from PathPlanning.RRTStarDubins import rrt_star_dubins as m


def test1():
    planners = []
    planning = m.RRTStarDubins.planning

    def recording_planning(self, *args, **kwargs):
        planners.append(self)
        return planning(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m, "show_animation", False)
        mp.setattr(m.RRTStarDubins, "planning", recording_planning)
        # one run takes about 14 s, so it is not compared with a second run
        path = m.main()
    assert len(planners) == 1
    pc.assert_dubins_path(path, planners[0])


if __name__ == '__main__':
    conftest.run_this_test(__file__)
