"""
Answer checks for the Bug 0, Bug 1 and Bug 2 planners on the demo map.

main() runs the three planners once; BugPlanner is wrapped so the planner
objects (and their paths) can be read back. Each path must start at the
start, end at the goal, never step onto an obstacle, and follow the bug
rules: one cell per move, and a move that starts away from the obstacle
boundary heads straight for the goal. Bug algorithms are not optimal, so
path lengths are not compared.
"""
import numpy as np
import pytest

import conftest
from PathPlanning.BugPlanning import bug as m

START, GOAL = (0.0, 0.0), (167.0, 50.0)  # from main()
NAMES = ["bug0", "bug1", "bug2"]


@pytest.fixture(scope="module")
def planners():
    """The planners main() ran, by name (one run of the slow demo)."""
    runs = []

    class RecordingBugPlanner(m.BugPlanner):
        def __init__(self, *args):
            super().__init__(*args)
            runs.append(self)

    m.show_animation = False
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m, "BugPlanner", RecordingBugPlanner)
        m.main(bug_0=True, bug_1=True, bug_2=True)
    assert len(runs) == 3
    return dict(zip(NAMES, runs))


def path_of(bug):
    return list(zip(bug.r_x, bug.r_y))


@pytest.mark.parametrize("name", NAMES)
def test_path_runs_from_start_to_goal(planners, name):
    path = path_of(planners[name])
    assert path[0] == START
    assert path[-1] == GOAL


@pytest.mark.parametrize("name", NAMES)
def test_path_never_enters_an_obstacle(planners, name):
    bug = planners[name]
    obstacles = set(zip(bug.obs_x, bug.obs_y))
    assert not obstacles & set(path_of(bug))


@pytest.mark.parametrize("name", NAMES)
def test_path_follows_the_bug_rules(planners, name):
    bug = planners[name]
    obstacles = set(zip(bug.obs_x, bug.obs_y))
    # cells next to an obstacle (computed here, not taken from the planner)
    boundary = {(x + dx, y + dy) for x, y in obstacles
                for dx in (-1, 0, 1) for dy in (-1, 0, 1)} - obstacles
    path = path_of(bug)
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        assert max(abs(x1 - x0), abs(y1 - y0)) == 1, \
            f"({x0}, {y0}) -> ({x1}, {y1}) is not a one-cell move"
        if (x0, y0) not in boundary:  # moving freely: straight for the goal
            assert (x1 - x0, y1 - y0) == (np.sign(GOAL[0] - x0),
                                          np.sign(GOAL[1] - y0)), \
                f"({x0}, {y0}) -> ({x1}, {y1}) does not head for the goal"


if __name__ == '__main__':
    conftest.run_this_test(__file__)
