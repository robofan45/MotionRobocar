"""
Answer checks for the D* Lite demo.

The robot plans on the known map and starts moving; after its fourth move it
discovers a new wall (the spoofed obstacles of main()) that blocks its route
and replans. D* Lite keeps the robot on a shortest path of the map it knows
at each moment, so the first four moves follow a shortest path of the
original map and every later move a shortest path of the map with the new
wall. Both are checked with an independent Dijkstra search.
"""
import heapq
import math
import random

import pytest

import conftest
import grid_planner_checks as chk
from PathPlanning.DStarLite import d_star_lite as m

START, GOAL = (10, 10), (50, 50)  # from main()
GRID = range(-10, 60)  # x and y of the planner's cells (world coordinates)
# wall main() reveals after the 4th move (its spoofed obstacles)
NEW_WALL = {(x, 20) for x in range(0, 21)} | {(0, y) for y in range(0, 20)}
MOVES_BEFORE_NEW_WALL = 4


@pytest.fixture(scope="module")
def demo():
    """Result of main() and the state of the random generator after it."""
    m.show_animation = False
    result = m.main()
    return result, random.getstate()


def static_obstacles():
    ox, oy = chk.demo_obstacles(as_float=False)
    return {(int(x), int(y)) for x, y in zip(ox, oy)}


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
                        or nxt[0] not in GRID or nxt[1] not in GRID:
                    continue
                nd = d + math.hypot(dx, dy)
                if nd < dist.get(nxt, math.inf):
                    dist[nxt] = nd
                    heapq.heappush(queue, (nd, nxt))
    return dist


def test_demo_reaches_goal_with_legal_moves(demo):
    found, path_x, path_y = demo[0]
    assert found
    path = list(zip(path_x, path_y))
    assert path[0] == START and path[-1] == GOAL
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        assert max(abs(x1 - x0), abs(y1 - y0)) == 1, "steps must be 1 cell"
    blocked = static_obstacles() | NEW_WALL
    for x, y in path:
        assert x in GRID and y in GRID
        assert (x, y) not in blocked, f"({x}, {y}) is an obstacle"


def test_robot_follows_shortest_paths_of_the_map_it_knows(demo):
    _, path_x, path_y = demo[0]
    path = list(zip(path_x, path_y))
    old_map = distances_to_goal(static_obstacles())
    new_map = distances_to_goal(static_obstacles() | NEW_WALL)
    for i, (a, b) in enumerate(zip(path, path[1:])):
        known = old_map if i < MOVES_BEFORE_NEW_WALL else new_map
        assert known[a] == pytest.approx(math.dist(a, b) + known[b]), \
            f"move {i} {a} -> {b} leaves the shortest path"
    # so the whole trip is 4 straight moves plus the replanned shortest path
    length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
    assert length == pytest.approx(
        MOVES_BEFORE_NEW_WALL + new_map[path[MOVES_BEFORE_NEW_WALL]])


def test_demo_is_reproducible(demo):
    result, state = demo
    assert m.main() == result
    # every random draw of the demo comes from the generator main() seeds
    assert random.getstate() == state



def test_legend_fits_in_the_window(monkeypatch):
    """The 'Key:' legend sits right of the map; it used to be cut off at
    the window's edge."""
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "pause", lambda *args, **kwargs: None)
    monkeypatch.setattr(m, "show_animation", True)
    try:
        m.main()
        fig = plt.gcf()
        fig.canvas.draw()
        legend = fig.axes[0].get_legend().get_window_extent()
        window = fig.bbox
    finally:
        plt.close("all")
    assert legend.x0 >= window.x0 and legend.x1 <= window.x1
    assert legend.y0 >= window.y0 and legend.y1 <= window.y1

if __name__ == '__main__':
    conftest.run_this_test(__file__)
