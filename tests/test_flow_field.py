"""
Answer checks for the flow field demo.

The integration field must hold, for every cell, the cheapest cost to reach
the goal: entering a cell costs its terrain energy (free 1, medium 7, hard 20)
plus 10 for a straight or 14 for a diagonal move. It is compared with an
independent Dijkstra search. Every vector must point to a neighbour that is
cheaper, so following the vectors from the start always reaches the goal.
"""
import heapq
import math

import pytest

import conftest
import PathPlanning.FlowField.flowfield as flow_field

ENERGY = {"free": 1, "medium": 7, "hard": 20}  # the module's cost field


@pytest.fixture(scope="module")
def flow():
    flow_field.show_animation = False
    return flow_field.main()


def as_cell(point):
    return round(point[0]), round(point[1])


def neighbours(cell):
    x, y = cell
    return [(x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            if (dx, dy) != (0, 0)]


def move_cost(a, b):
    return 10 if a[0] == b[0] or a[1] == b[1] else 14


def cheapest_cost_to_goal(flow):
    """Independent Dijkstra from the goal over the non-obstacle cells."""
    goal = as_cell(flow.goal_pt)
    terrain = flow.obs_grid
    dist = {goal: 0}
    queue = [(0, goal)]
    while queue:
        d, current = heapq.heappop(queue)
        if d > dist[current]:
            continue
        for nxt in neighbours(current):
            if nxt == goal or not (0 <= nxt[0] < flow.limit_x and
                                   0 <= nxt[1] < flow.limit_y) \
                    or terrain[nxt] == "obs":
                continue
            nd = d + ENERGY[terrain[nxt]] + move_cost(current, nxt)
            if nd < dist.get(nxt, math.inf):
                dist[nxt] = nd
                heapq.heappush(queue, (nd, nxt))
    return dist


def test_integration_field_is_the_cheapest_cost_to_goal(flow):
    expected = cheapest_cost_to_goal(flow)
    assert len(expected) > 1000  # every free cell of the maze is reachable
    for cell, cost in flow.integration_field.items():
        assert cost == expected.get(cell, math.inf), cell
    assert set(expected) <= set(flow.integration_field)


def test_vectors_point_to_a_cheaper_neighbour(flow):
    field = flow.integration_field
    assert flow.vector_field[as_cell(flow.goal_pt)] == (None, None)
    for cell, cost in field.items():
        if cost in (0, math.inf):  # the goal, or walled off cells
            continue
        nxt = flow.vector_field[cell]
        assert nxt in neighbours(cell)
        assert flow.obs_grid[nxt] != "obs"
        assert field[nxt] < cost


def test_following_the_vectors_from_the_start_reaches_the_goal(flow):
    path = [as_cell(flow.start_pt)]
    while flow.vector_field[path[-1]] != (None, None):
        path.append(flow.vector_field[path[-1]])
        assert len(path) <= len(flow.vector_field), "vectors loop"
    assert path[-1] == as_cell(flow.goal_pt)
    for a, b in zip(path, path[1:]):
        assert b in neighbours(a) and flow.obs_grid[b] != "obs"
    # The vectors point to the cheapest neighbour without adding the move
    # cost, so the route is not always a cheapest one; on the demo map it is.
    cost = sum(ENERGY[flow.obs_grid[a]] + move_cost(a, b)
               for a, b in zip(path, path[1:]))
    assert cost == flow.integration_field[path[0]]


def test_animated_walk_ends_at_the_goal(monkeypatch):
    """With the animation on, the walk used to plot the goal's (None, None)
    vector, and matplotlib raised a ValueError."""
    import matplotlib.pyplot as plt
    walked = []
    real_plot = plt.plot

    def recording_plot(*args, **kwargs):
        if args[2:] == ("b*",):
            walked.append(args[:2])
        return real_plot(*args, **kwargs)

    monkeypatch.setattr(plt, "plot", recording_plot)
    monkeypatch.setattr(plt, "pause", lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    monkeypatch.setattr(flow_field, "show_animation", True)
    try:
        flow = flow_field.main()
    finally:
        plt.close("all")
    assert walked[0] == tuple(flow.start_pt)
    assert walked[-1] == tuple(flow.goal_pt)


if __name__ == '__main__':
    conftest.run_this_test(__file__)
