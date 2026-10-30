"""
Answer checks for the bidirectional A* demo, which places random obstacles
and picks a random start and goal.

main() seeds numpy, so every run gives the same map and path. The path is
checked move by move against the module's motion model (8 neighbours, never
an obstacle, no diagonal squeeze between two obstacles), and an independent
breadth-first search on the same map decides whether a path exists at all.
The search is not optimal (Manhattan heuristic, two fronts), so its cost is
not compared.
"""
from collections import deque

import numpy as np
import pytest

import conftest
from PathPlanning.AStar import a_star_searching_from_two_side as m

TOP, BOTTOM = [60, 60], [0, 0]  # boundary corners used by main()


@pytest.fixture(autouse=True)
def no_animation():
    m.show_animation = False


@pytest.fixture(scope="module")
def demo():
    """(start, goal, obstacles incl. boundary, path) of main(800)."""
    m.show_animation = False
    return m.main(800)


def drop_repeats(cells):
    """Remove consecutive duplicate cells (see the repeated point test)."""
    return cells[:1] + [b for a, b in zip(cells, cells[1:]) if a != b]


def legal_moves(cell, blocked):
    """The module's motion model: the 8 neighbours that are not blocked,
    except a diagonal step between two blocked orthogonal neighbours."""
    x, y = cell
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            nxt = (x + dx, y + dy)
            if (dx, dy) == (0, 0) or nxt in blocked:
                continue
            if dx and dy and (x + dx, y) in blocked and (x, y + dy) in blocked:
                continue
            yield nxt


def reachable(start, goal, blocked):
    """Independent breadth-first search: can the goal be reached?"""
    seen = {start}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        if current == goal:
            return True
        for nxt in legal_moves(current, blocked):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


def check_answer(start, goal, bound, path):
    """A path exactly when one exists, and every move of it is legal."""
    start, goal = tuple(start), tuple(goal)
    blocked = {tuple(cell) for cell in bound.tolist()}
    assert start not in blocked and goal not in blocked
    if path is None:
        assert not reachable(start, goal, blocked), "a path exists"
        return
    cells = drop_repeats([tuple(cell) for cell in path.tolist()])
    assert cells[0] == start, "path must begin at the start"
    assert cells[-1] == goal, "path must end at the goal"
    for a, b in zip(cells, cells[1:]):
        assert b in set(legal_moves(a, blocked)), f"illegal move {a} -> {b}"
    for x, y in cells:  # strictly inside the boundary
        assert BOTTOM[0] < x < TOP[0] and BOTTOM[1] < y < TOP[1]


def test_demo_finds_a_valid_path(demo):
    start, goal, bound, path = demo
    assert path is not None  # seed 42 gives a solvable map
    check_answer(start, goal, bound, path)


def test_crowded_demo_reports_no_path():
    start, goal, bound, path = m.main(5000)  # obstacles block the path
    assert path is None
    check_answer(start, goal, bound, path)


def test_demo_is_reproducible(demo):
    start, goal, bound, path = m.main(800)
    assert (start, goal) == (demo[0], demo[1])
    assert np.array_equal(bound, demo[2])
    assert np.array_equal(path, demo[3])


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="the cell where the two searches meet is listed "
                          "twice in the path")
def test_path_has_no_repeated_points(demo):
    cells = [tuple(cell) for cell in demo[3].tolist()]
    assert cells == drop_repeats(cells)


# the first two maps have a path, the last two do not
@pytest.mark.parametrize("seed, obstacle_number",
                         [(0, 800), (2, 1500), (6, 2500), (7, 3000)])
def test_random_maps(seed, obstacle_number):
    np.random.seed(seed)  # same steps as main(), other maps
    start = m.random_coordinate(BOTTOM, TOP)
    goal = m.random_coordinate(BOTTOM, TOP)
    bound, obstacle = m.boundary_and_obstacles(start, goal, TOP, BOTTOM,
                                               obstacle_number)
    path = m.searching_control(start, goal, bound, obstacle)
    check_answer(start, goal, bound, path)


if __name__ == '__main__':
    conftest.run_this_test(__file__)
