"""
Answer checks for the Voronoi road map planner demo.

The road map nodes are the Voronoi vertices of the obstacle points plus the
start and the goal; each node is linked to up to N_KNN near nodes by edges
shorter than MAX_EDGE_LEN that the planner found collision free. The path
must join start and goal along road map edges, keep the robot (radius
robot_size) clear of every obstacle point along every segment, and be the
shortest path through the road map, checked with an independent Dijkstra
search on the same road map.
"""
import heapq
import math

import numpy as np
import pytest

import conftest  # Add root path to sys.path
from PathPlanning.VoronoiRoadMap import voronoi_road_map as m

START, GOAL = (10.0, 10.0), (50.0, 50.0)  # from main()
ROBOT_SIZE = 5.0  # [m]


@pytest.fixture(autouse=True)
def no_animation():
    m.show_animation = False


def demo_obstacles():
    """Obstacle points of main()."""
    ox = [*map(float, range(60)), *[60.0] * 60, *map(float, range(61)),
          *[0.0] * 61, *[20.0] * 40, *[40.0] * 40]
    oy = [*[0.0] * 60, *map(float, range(60)), *[60.0] * 61,
          *map(float, range(61)), *map(float, range(40)),
          *[60.0 - i for i in range(40)]]
    return ox, oy


@pytest.fixture(scope="module")
def demo():
    """Obstacles, road map (nodes, edges) and path of main()'s scenario."""
    road = {}
    build = m.VoronoiRoadMapPlanner.generate_road_map_info

    def recording_build(self, node_x, node_y, rr, obstacle_tree):
        road["nodes"] = list(zip(node_x, node_y))
        road["edges"] = build(self, node_x, node_y, rr, obstacle_tree)
        return road["edges"]

    m.show_animation = False
    ox, oy = demo_obstacles()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m.VoronoiRoadMapPlanner, "generate_road_map_info",
                   recording_build)
        rx, ry = m.VoronoiRoadMapPlanner().planning(*START, *GOAL, ox, oy,
                                                    ROBOT_SIZE)
    return np.c_[ox, oy], road["nodes"], road["edges"], list(zip(rx, ry))


def clearance(a, b, obstacles):
    """Smallest distance from the segment a-b to an obstacle point."""
    a, b = np.asarray(a), np.asarray(b)
    d = b - a
    t = np.clip((obstacles - a) @ d / (d @ d), 0.0, 1.0)
    return np.hypot(*(obstacles - (a + t[:, None] * d)).T).min()


def shortest_length(nodes, edges, start, goal):
    """Independent Dijkstra over the (directed) road map."""
    dist = {start: 0.0}
    queue = [(0.0, start)]
    while queue:
        d, i = heapq.heappop(queue)
        if i == goal:
            return d
        if d > dist[i]:
            continue
        for j in edges[i]:
            nd = d + math.dist(nodes[i], nodes[j])
            if nd < dist.get(j, math.inf):
                dist[j] = nd
                heapq.heappush(queue, (nd, j))
    return None


def test1():
    m.main()


def test_path_joins_start_and_goal_along_road_map_edges(demo):
    _, nodes, edges, path = demo
    assert path[0] == START and path[-1] == GOAL
    for a, b in zip(path, path[1:]):
        assert nodes.index(b) in edges[nodes.index(a)], f"no edge {a} -> {b}"
        assert 0 < math.dist(a, b) < m.VoronoiRoadMapPlanner().MAX_EDGE_LEN


def test_robot_keeps_clear_of_obstacles_along_the_path(demo):
    obstacles, _, _, path = demo
    for a, b in zip(path, path[1:]):
        assert clearance(a, b, obstacles) > ROBOT_SIZE  # measured 9.9 m


def test_path_is_the_shortest_through_the_road_map(demo):
    _, nodes, edges, path = demo
    length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
    expected = shortest_length(nodes, edges, nodes.index(START),
                               nodes.index(GOAL))
    assert length == pytest.approx(expected, abs=1e-9)


if __name__ == '__main__':
    conftest.run_this_test(__file__)
