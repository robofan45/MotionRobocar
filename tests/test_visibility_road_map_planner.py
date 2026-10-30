"""
Answer checks for the visibility road map planner demo.

The road map links the start, the goal and the obstacle corners pushed out
by expand_distance whenever the straight edge between two of them crosses
no obstacle edge. The path must join start and goal without crossing an
obstacle (an independent segment intersection test), and its length must
equal the shortest path through the same nodes, found by an independent
visibility graph and Dijkstra search.
"""
import heapq
import math

import pytest

import conftest  # Add root path to sys.path
from PathPlanning.VisibilityRoadMap import visibility_road_map as m

START, GOAL = (10.0, 10.0), (50.0, 50.0)  # from main()
EXPAND_DISTANCE = 5.0  # [m]
POLYGONS = [([20.0, 30.0, 15.0], [20.0, 20.0, 30.0]),  # obstacles of main()
            ([40.0, 45.0, 50.0, 40.0], [50.0, 40.0, 20.0, 40.0]),
            ([20.0, 30.0, 30.0, 20.0], [40.0, 45.0, 60.0, 50.0])]


@pytest.fixture(autouse=True)
def no_animation():
    m.show_animation = False


@pytest.fixture
def demo():
    """Planner, obstacles and path of the demo scenario in main()."""
    planner = m.VisibilityRoadMap(EXPAND_DISTANCE)
    obstacles = [m.ObstaclePolygon(list(xs), list(ys)) for xs, ys in POLYGONS]
    rx, ry = planner.planning(*START, *GOAL, obstacles)
    return planner, obstacles, list(zip(rx, ry))


def corners(obstacle):
    """Polygon corners without the closing repeat of the first one."""
    return list(zip(obstacle.x_list, obstacle.y_list))[:-1]


def orientation(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def on_segment(p, a, b):
    return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))


def segments_touch(p1, p2, q1, q2):
    """Closed segments p1-p2 and q1-q2 cross or touch."""
    d1, d2 = orientation(q1, q2, p1), orientation(q1, q2, p2)
    d3, d4 = orientation(p1, p2, q1), orientation(p1, p2, q2)
    if d1 * d2 < 0 and d3 * d4 < 0:
        return True
    return ((d1 == 0 and on_segment(p1, q1, q2))
            or (d2 == 0 and on_segment(p2, q1, q2))
            or (d3 == 0 and on_segment(q1, p1, p2))
            or (d4 == 0 and on_segment(q2, p1, p2)))


def inside(point, polygon):
    """Ray casting point in polygon test."""
    x, y = point
    result = False
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            result = not result
    return result


def blocked(a, b, obstacles):
    """The segment a-b crosses or touches an obstacle, or lies inside one."""
    middle = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    for obstacle in obstacles:
        polygon = corners(obstacle)
        edges = zip(polygon, polygon[1:] + polygon[:1])
        if any(segments_touch(a, b, c, d) for c, d in edges) \
                or inside(middle, polygon):
            return True
    return False


def shortest_length(nodes, obstacles):
    """Independent Dijkstra over the visibility graph of the nodes; node 0
    is the start and node 1 the goal."""
    dist = {0: 0.0}
    queue = [(0.0, 0)]
    while queue:
        d, i = heapq.heappop(queue)
        if i == 1:
            return d
        if d > dist[i]:
            continue
        for j, node in enumerate(nodes):
            if j == i or blocked(nodes[i], node, obstacles):
                continue
            nd = d + math.dist(nodes[i], node)
            if nd < dist.get(j, math.inf):
                dist[j] = nd
                heapq.heappush(queue, (nd, j))
    return None


def test1():
    m.main()


def test_path_joins_start_and_goal_without_crossing_obstacles(demo):
    _, obstacles, path = demo
    assert path[0] == START and path[-1] == GOAL
    for a, b in zip(path, path[1:]):
        assert not blocked(a, b, obstacles), f"{a} -> {b} hits an obstacle"


def test_road_map_nodes_keep_away_from_the_obstacles(demo):
    planner, obstacles, _ = demo
    for obstacle in obstacles:
        cvx, cvy = planner.calc_vertexes_in_configuration_space(
            obstacle.x_list, obstacle.y_list)
        for corner, node in zip(corners(obstacle), zip(cvx, cvy)):
            assert math.dist(corner, node) == pytest.approx(EXPAND_DISTANCE)
            assert not any(inside(node, corners(o)) for o in obstacles)


def test_path_is_the_shortest_through_the_road_map_nodes(demo):
    planner, obstacles, path = demo
    nodes = [START, GOAL]
    for obstacle in obstacles:
        nodes += zip(*planner.calc_vertexes_in_configuration_space(
            obstacle.x_list, obstacle.y_list))
    assert all(point in nodes for point in path)
    length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
    assert length == pytest.approx(shortest_length(nodes, obstacles),
                                   abs=1e-9)


if __name__ == '__main__':
    conftest.run_this_test(__file__)
