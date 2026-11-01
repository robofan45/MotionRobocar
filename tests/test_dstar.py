"""
Answer checks for the D* demo.

The robot plans on the known map. Right after that a new wall appears
(AddNewObstacle) which the robot only notices when the next cell of its plan
is blocked; D* then repairs the plan. So the robot first follows a shortest
path of the original map and, from the cell where its plan runs into the new
wall, a shortest path of the changed map. Both are checked with an
independent Dijkstra search.
"""
import heapq
import math

import pytest

import conftest
import grid_planner_checks as chk
from PathPlanning.DStar import dstar as m

START, GOAL = (10, 10), (50, 50)  # from main()
SIZE = 100  # main() uses a 100 x 100 map
NEW_WALL = {(x, 40) for x in range(5, 21)}  # added by AddNewObstacle()


@pytest.fixture(scope="module")
def path():
    m.show_animation = False
    rx, ry = m.main()
    return list(zip(rx, ry))


def drop_repeats(points):
    """Remove consecutive duplicate points (see the repeated point test)."""
    return points[:1] + [b for a, b in zip(points, points[1:]) if a != b]


def static_obstacles():
    """The obstacles of main() that lie on the map (it ignores the rest)."""
    ox, oy = chk.demo_obstacles(as_float=False)
    return {(int(x), int(y)) for x, y in zip(ox, oy)
            if 0 <= x < SIZE and 0 <= y < SIZE}


def distances_to_goal(blocked):
    """Independent Dijkstra from the goal (8-connected, 1 or sqrt(2))."""
    dist = {GOAL: 0.0}
    queue = [(0.0, GOAL)]
    while queue:
        d, current = heapq.heappop(queue)
        if d > dist[current]:
            continue
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nxt = (current[0] + dx, current[1] + dy)
                if (dx, dy) == (0, 0) or nxt in blocked \
                        or not (0 <= nxt[0] < SIZE and 0 <= nxt[1] < SIZE):
                    continue
                nd = d + math.hypot(dx, dy)
                if nd < dist.get(nxt, math.inf):
                    dist[nxt] = nd
                    heapq.heappush(queue, (nd, nxt))
    return dist


def test_path_avoids_obstacles_and_moves_one_cell_at_a_time(path):
    cells = path
    assert cells[0] == START
    for (x0, y0), (x1, y1) in zip(cells, cells[1:]):
        assert max(abs(x1 - x0), abs(y1 - y0)) == 1, "steps must be 1 cell"
    blocked = static_obstacles() | NEW_WALL
    for x, y in cells:
        assert 0 <= x < SIZE and 0 <= y < SIZE
        assert (x, y) not in blocked, f"({x}, {y}) is an obstacle"


def test_robot_follows_shortest_paths_of_the_map_it_knows(path):
    cells = path
    old_map = distances_to_goal(static_obstacles())
    new_map = distances_to_goal(static_obstacles() | NEW_WALL)

    def on_shortest_path(dist, a, b):
        return dist[a] == pytest.approx(math.dist(a, b) + dist[b])

    moves = list(zip(cells, cells[1:]))
    # the first move that leaves the original map's shortest path
    k = next(i for i, (a, b) in enumerate(moves)
             if not on_shortest_path(old_map, a, b))
    # it happens where the shortest path the robot was on runs into the wall
    here = cells[k]
    assert any(on_shortest_path(old_map, here, w) for w in NEW_WALL
               if math.dist(here, w) < 1.5)
    # and from there the robot follows a shortest path of the changed map
    for a, b in moves[k:]:
        assert on_shortest_path(new_map, a, b), f"{a} -> {b} is a detour"


def test_path_ends_at_goal(path):
    assert path[-1] == GOAL


def test_path_has_no_repeated_points(path):
    assert path == drop_repeats(path)


if __name__ == '__main__':
    conftest.run_this_test(__file__)
