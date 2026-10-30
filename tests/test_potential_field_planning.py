"""
Answer checks for the potential field planner demo.

The planner walks downhill on a grid of potential values: attraction to the
goal plus repulsion near obstacles. The potential is recomputed here from the
module's formulas, and every step must go to the neighbour with the lowest
potential, so the potential falls all the way to the goal. The robot must
reach the goal one cell at a time without visiting a cell twice (no
oscillation) and without running into an obstacle.
"""
import math

import pytest

import conftest
from PathPlanning.PotentialFieldPlanning import potential_field_planning as m

START, GOAL = (0.0, 10.0), (30.0, 30.0)  # from main()
GRID_SIZE = 0.5  # [m] from main()
# main() calls this the robot radius; it is the range of the repulsion
REPULSION_RANGE = 5.0  # [m]
OX, OY = [15.0, 5.0, 20.0, 25.0], [25.0, 15.0, 26.0, 25.0]


@pytest.fixture(scope="module")
def path():
    m.show_animation = False
    rx, ry = m.main()
    return list(zip(rx, ry))


def potential(x, y):
    """Attractive plus repulsive potential (the formulas of the module)."""
    attractive = 0.5 * m.KP * math.hypot(x - GOAL[0], y - GOAL[1])
    nearest = min(math.hypot(x - ox, y - oy) for ox, oy in zip(OX, OY))
    if nearest > REPULSION_RANGE:
        return attractive
    nearest = max(nearest, 0.1)
    return attractive + 0.5 * m.ETA * (1.0 / nearest
                                       - 1.0 / REPULSION_RANGE) ** 2


def test_robot_reaches_goal_one_cell_at_a_time(path):
    assert path[0] == START
    assert math.dist(path[-1], GOAL) < GRID_SIZE  # the planner's stop rule
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        steps = (round((x1 - x0) / GRID_SIZE), round((y1 - y0) / GRID_SIZE))
        assert max(map(abs, steps)) == 1, "one cell per step"
        assert (x1 - x0, y1 - y0) == pytest.approx(
            (steps[0] * GRID_SIZE, steps[1] * GRID_SIZE))


def test_robot_never_oscillates(path):
    assert len(set(path)) == len(path), "a cell is visited twice"


def test_every_step_goes_down_to_the_lowest_neighbour(path):
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        lowest = min(potential(x0 + dx * GRID_SIZE, y0 + dy * GRID_SIZE)
                     for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                     if (dx, dy) != (0, 0))
        assert potential(x1, y1) == pytest.approx(lowest, abs=1e-9)
        assert potential(x1, y1) < potential(x0, y0)


def test_robot_keeps_away_from_obstacles(path):
    nearest = min(math.hypot(x - ox, y - oy)
                  for x, y in path for ox, oy in zip(OX, OY))
    assert nearest > 2 * GRID_SIZE  # measured 2.24 m


if __name__ == '__main__':
    conftest.run_this_test(__file__)
