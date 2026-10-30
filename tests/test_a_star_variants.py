"""
Answer checks for the A* variants demo.

Every variant runs on the demo maze from main(). Its path (read back from the
predecessor links) must join start and goal and never cross a wall cell.
Plain A* (no variant switched on) uses an admissible octile heuristic, so its
path cost must equal an independent Dijkstra search with the module's move
costs (10 straight, 14 diagonal). The variants give up optimality for speed,
so for them only the validity of the path is checked.
"""
import heapq
import math

import pytest

import conftest
import grid_planner_checks as chk
import PathPlanning.AStar.a_star_variants as a_star

START, GOAL = (5, 5), (35, 45)  # from main()
VARIANT_FLAGS = ["use_beam_search", "use_iterative_deepening",
                 "use_dynamic_weighting", "use_theta_star", "use_jump_point"]


@pytest.fixture(autouse=True)
def plain_a_star_without_animation(monkeypatch):
    monkeypatch.setattr(a_star, "show_animation", False)
    for flag in VARIANT_FLAGS:
        monkeypatch.setattr(a_star, flag, False)


def path_from(search):
    """Follow the predecessor links from the goal back to the start."""
    node = search.all_nodes[tuple(search.goal_pt)]
    path = [tuple(int(v) for v in node["pos"])]
    while node["pred"] is not None:
        assert len(path) <= len(search.all_nodes), "predecessor links loop"
        node = search.all_nodes[tuple(node["pred"])]
        path.append(tuple(int(v) for v in node["pos"]))
    return path[::-1]


def move_cost(a, b):
    """Octile cost of a move: 10 per straight cell, 14 per diagonal cell."""
    dx, dy = abs(b[0] - a[0]), abs(b[1] - a[1])
    return 10 * max(dx, dy) + 4 * min(dx, dy)


def shortest_cost(grid, start, goal):
    """Independent Dijkstra over the free cells of the grid, or None."""
    dist = {start: 0}
    queue = [(0, start)]
    while queue:
        d, current = heapq.heappop(queue)
        if current == goal:
            return d
        if d > dist[current]:
            continue
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nxt = (current[0] + dx, current[1] + dy)
                if (dx, dy) == (0, 0) or grid.get(nxt, True):
                    continue  # outside the map or a wall
                nd = d + move_cost(current, nxt)
                if nd < dist.get(nxt, math.inf):
                    dist[nxt] = nd
                    heapq.heappush(queue, (nd, nxt))
    return None


def nodes_on_move(a, b):
    """Grid nodes on a move along one of the 8 directions (what the grid
    variants and the theta variant check)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = max(abs(dx), abs(dy))
    assert dx in (0, n, -n) and dy in (0, n, -n), \
        f"{a} -> {b} does not follow one of the 8 directions"
    return [(a[0] + k * dx // n, a[1] + k * dy // n) for k in range(n + 1)]


def cells_crossed(a, b, step=0.005):
    """Cells a straight segment passes through, the line of sight test of
    the jump point variant: dense samples, cell = (int(x), int(y))."""
    n = max(1, math.ceil(math.dist(a, b) / step))
    return {(math.floor(a[0] + (b[0] - a[0]) * k / n),
             math.floor(a[1] + (b[1] - a[1]) * k / n)) for k in range(n + 1)}


def assert_valid_path(path, grid, start, goal, any_angle=False):
    assert path[0] == start, "path must begin at the start"
    assert path[-1] == goal, "path must end at the goal"
    for a, b in zip(path, path[1:]):
        assert a != b, "repeated point"
        cells = cells_crossed(a, b) if any_angle else nodes_on_move(a, b)
        for c in cells:  # grid.get is None outside the map
            assert grid.get(c) is False, f"{a} -> {b} crosses wall cell {c}"


def run_main():
    search = a_star.main()
    return search, path_from(search)


def test_plain_a_star_finds_the_shortest_path():
    search, path = run_main()
    assert_valid_path(path, search.obs_grid, START, GOAL)
    assert all(max(abs(b[0] - a[0]), abs(b[1] - a[1])) == 1
               for a, b in zip(path, path[1:])), "steps must be 1 cell"
    cost = sum(move_cost(a, b) for a, b in zip(path, path[1:]))
    assert cost == shortest_cost(search.obs_grid, START, GOAL)


@pytest.mark.parametrize("flag", ["use_beam_search", "use_iterative_deepening",
                                  "use_dynamic_weighting"])
def test_grid_variants_give_valid_paths(flag, monkeypatch):
    monkeypatch.setattr(a_star, flag, True)
    search, path = run_main()
    assert_valid_path(path, search.obs_grid, START, GOAL)
    assert all(max(abs(b[0] - a[0]), abs(b[1] - a[1])) == 1
               for a, b in zip(path, path[1:])), "steps must be 1 cell"
    cost = sum(move_cost(a, b) for a, b in zip(path, path[1:]))
    assert cost >= shortest_cost(search.obs_grid, START, GOAL)


def test_theta_star_variant_gives_valid_straight_jumps(monkeypatch):
    monkeypatch.setattr(a_star, "use_theta_star", True)
    search, path = run_main()
    # jumps follow one of the 8 directions over free nodes
    assert_valid_path(path, search.obs_grid, START, GOAL)
    assert all(max(abs(b[0] - a[0]), abs(b[1] - a[1])) <= a_star.max_theta
               for a, b in zip(path, path[1:]))


def test_jump_point_variant_gives_valid_corner_to_corner_path(monkeypatch):
    monkeypatch.setattr(a_star, "use_jump_point", True)
    search, path = run_main()
    assert_valid_path(path, search.obs_grid, START, GOAL, any_angle=True)
    assert all(math.dist(a, b) <= a_star.max_corner
               for a, b in zip(path, path[1:]))


def random_maze(seed):
    """chk.random_scenario's walls as the module's cell dictionary."""
    ox, oy, start, goal = chk.random_scenario(seed)
    size = round(max(ox))
    grid = {(i, j): False for i in range(size + 1) for j in range(size + 1)}
    for x, y in zip(ox, oy):
        grid[(round(x), round(y))] = True
    return grid, size, (round(start[0]), round(start[1])), \
        (round(goal[0]), round(goal[1]))


# seeds whose start and goal are free and connected
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_plain_a_star_on_random_mazes(seed):
    grid, size, start, goal = random_maze(seed)
    expected = shortest_cost(grid, start, goal)
    assert expected is not None and not grid[start] and not grid[goal]
    search = a_star.SearchAlgo(grid, *goal, *start, size + 1, size + 1)
    search.a_star()
    path = path_from(search)
    assert_valid_path(path, grid, start, goal)
    assert sum(move_cost(a, b) for a, b in zip(path, path[1:])) == expected


if __name__ == '__main__':
    conftest.run_this_test(__file__)
