"""
Shared answer checks for the grid based A* and Dijkstra planners.

The reference solver below is an independent textbook Dijkstra search on the
planner's own occupancy grid, so it tells us the true shortest path cost.

author: Mahim Trivedi
"""
import heapq
import math
import random

import pytest

SQRT2 = math.sqrt(2.0)
MOTIONS = [(1, 0, 1.0), (0, 1, 1.0), (-1, 0, 1.0), (0, -1, 1.0),
           (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]


def demo_obstacles(as_float=True):
    """Obstacle layout used by the A* and Dijkstra demos."""
    conv = float if as_float else (lambda v: v)
    ox, oy = [], []
    for i in range(-10, 60):
        ox.append(conv(i))
        oy.append(-10.0)
    for i in range(-10, 60):
        ox.append(60.0)
        oy.append(conv(i))
    for i in range(-10, 61):
        ox.append(conv(i))
        oy.append(60.0)
    for i in range(-10, 61):
        ox.append(-10.0)
        oy.append(conv(i))
    for i in range(-10, 40):
        ox.append(20.0)
        oy.append(conv(i))
    for i in range(0, 40):
        ox.append(40.0)
        oy.append(60.0 - i)
    return ox, oy


def boundary(size):
    """Square wall from 0 to size (inclusive) with 1 m spacing."""
    ox, oy = [], []
    for i in range(size + 1):
        ox += [float(i), float(i), 0.0, float(size)]
        oy += [0.0, float(size), float(i), float(i)]
    return ox, oy


def random_scenario(seed, size=None, shift=(0.0, 0.0)):
    """Random walls inside a square boundary plus a start and a goal.

    shift="random" moves the whole scene, so the map does not start at (0, 0).
    """
    rng = random.Random(seed)
    size = size or rng.randint(14, 30)
    ox, oy = boundary(size)
    for _ in range(rng.randint(1, 4)):
        if rng.random() < 0.5:
            x = float(rng.randint(3, size - 3))
            y0, y1 = sorted(rng.sample(range(0, size + 1), 2))
            for y in range(y0, y1 + 1):
                ox.append(x)
                oy.append(float(y))
        else:
            y = float(rng.randint(3, size - 3))
            x0, x1 = sorted(rng.sample(range(0, size + 1), 2))
            for x in range(x0, x1 + 1):
                ox.append(float(x))
                oy.append(y)
    start = (float(rng.randint(2, size - 2)), float(rng.randint(2, size - 2)))
    goal = (float(rng.randint(2, size - 2)), float(rng.randint(2, size - 2)))
    if shift == "random":  # drawn last, so the layout matches shift=(0, 0)
        shift = (rng.choice([0.0, 3.0, 17.0, -6.0]),
                 rng.choice([0.0, 2.0, 11.0, -9.0]))
    dx, dy = shift
    ox = [x + dx for x in ox]
    oy = [y + dy for y in oy]
    return ox, oy, (start[0] + dx, start[1] + dy), (goal[0] + dx, goal[1] + dy)


def cell(planner, x, y):
    res = planner.resolution
    return (round((x - planner.min_x) / res), round((y - planner.min_y) / res))


def is_free(planner, ix, iy):
    return (0 <= ix < planner.x_width and 0 <= iy < planner.y_width
            and not planner.obstacle_map[ix][iy])


def reference_cost(planner, sx, sy, gx, gy):
    """Shortest path cost in grid steps, or None if the goal is unreachable."""
    start, goal = cell(planner, sx, sy), cell(planner, gx, gy)
    if start == goal:
        return 0.0
    if not is_free(planner, *goal):
        return None
    dist = {start: 0.0}
    queue = [(0.0, start)]
    while queue:
        d, current = heapq.heappop(queue)
        if current == goal:
            return d
        if d > dist[current]:
            continue
        for dx, dy, w in MOTIONS:
            nxt = (current[0] + dx, current[1] + dy)
            if not is_free(planner, *nxt):
                continue
            nd = d + w
            if nd < dist.get(nxt, math.inf):
                dist[nxt] = nd
                heapq.heappush(queue, (nd, nxt))
    return None


def path_cost(planner, rx, ry):
    res = planner.resolution
    return sum(math.hypot(x1 - x0, y1 - y0) / res
               for x0, y0, x1, y1 in zip(rx, ry, rx[1:], ry[1:]))


def assert_valid_path(planner, rx, ry, start, goal):
    """Path runs goal -> start, moves one cell at a time, never hits a wall."""
    assert len(rx) == len(ry) > 0
    res = planner.resolution
    gx, gy = cell(planner, *goal)
    sx, sy = cell(planner, *start)
    assert cell(planner, rx[0], ry[0]) == (gx, gy), "path must begin at goal"
    assert cell(planner, rx[-1], ry[-1]) == (sx, sy), "path must end at start"
    cells = [cell(planner, x, y) for x, y in zip(rx, ry)]
    for (x0, y0), (x1, y1) in zip(cells, cells[1:]):
        assert max(abs(x1 - x0), abs(y1 - y0)) == 1, "steps must be 1 cell"
    for ix, iy in cells[:-1]:  # the start cell itself is not checked
        assert is_free(planner, ix, iy), f"cell {(ix, iy)} is blocked"
    for x, y in zip(rx, ry):
        assert x == pytest.approx(planner.min_x + cell(planner, x, y)[0] * res)


def robot_radius(planner):
    return planner.rr if hasattr(planner, "rr") else planner.robot_radius


def assert_clear_of_raw_obstacles(planner, rx, ry, ox, oy):
    """Every path point except the start keeps more than the robot radius
    from every raw obstacle point (independent of the planner's grid)."""
    rr = robot_radius(planner)
    for x, y in zip(rx[:-1], ry[:-1]):
        nearest = min(math.hypot(x - px, y - py) for px, py in zip(ox, oy))
        assert nearest > rr, f"({x}, {y}) is {nearest:.3f} m from an obstacle"


def check_planner(planner, start, goal, obstacles=None):
    """Plan and compare against the reference; return (rx, ry)."""
    rx, ry = planner.planning(*start, *goal)
    expected = reference_cost(planner, *start, *goal)
    if expected is None:
        assert (rx, ry) == ([], []), "unreachable goal must give an empty path"
    else:
        assert_valid_path(planner, rx, ry, start, goal)
        assert path_cost(planner, rx, ry) == pytest.approx(expected, abs=1e-9)
        if obstacles is not None:
            assert_clear_of_raw_obstacles(planner, rx, ry, *obstacles)
    return rx, ry


def enclosed_goal_obstacles():
    """Boundary 0..20 with a closed box around (15, 15)."""
    ox, oy = boundary(20)
    for i in range(11, 20):
        ox += [11.0, 19.0, float(i), float(i)]
        oy += [float(i), float(i), 11.0, 19.0]
    return ox, oy


def odd_extent_obstacles():
    """Boundary whose width in cells rounds down (73 m / 2 m = 36.5)."""
    ox, oy = [], []
    for i in range(0, 74):
        ox += [float(i), float(i)]
        oy += [0.0, 30.0]
    for i in range(0, 31):
        ox += [0.0, 73.0]
        oy += [float(i), float(i)]
    return ox, oy


def run_with_animation(module, monkeypatch):
    """Run module.main() with plotting on, without opening windows."""
    monkeypatch.setattr(module.plt, "pause", lambda *a, **k: None)
    monkeypatch.setattr(module.plt, "show", lambda *a, **k: None)
    module.show_animation = True
    try:
        module.main()
        legend = module.plt.gca().get_legend()
        labels = set() if legend is None else {
            t.get_text() for t in legend.get_texts()}
    finally:
        module.show_animation = False
        module.plt.close("all")
    return labels
