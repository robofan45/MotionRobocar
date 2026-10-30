import math

import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

import conftest  # Add root path to sys.path
import path_checks as pc
from PathPlanning.ProbabilisticRoadMap import probabilistic_road_map as m


def run_demo():
    """Run main() without animation; return the path and a record of the
    planning problem and the road map that was searched."""
    record = {}
    prm_planning = m.prm_planning
    generate_road_map = m.generate_road_map

    def recording_prm_planning(sx, sy, gx, gy, ox, oy, rr, *, rng=None):
        record.update(start=(sx, sy), goal=(gx, gy), rr=rr,
                      obstacles=[(x, y, 0.0) for x, y in zip(ox, oy)])
        return prm_planning(sx, sy, gx, gy, ox, oy, rr, rng=rng)

    def recording_generate_road_map(sample_x, sample_y, rr, obstacle_kd_tree):
        road_map = generate_road_map(sample_x, sample_y, rr, obstacle_kd_tree)
        record.update(sample_x=list(sample_x), sample_y=list(sample_y),
                      road_map=road_map)
        return road_map

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m, "show_animation", False)
        mp.setattr(m, "prm_planning", recording_prm_planning)
        mp.setattr(m, "generate_road_map", recording_generate_road_map)
        rx, ry = m.main()
    return list(zip(rx, ry)), record


def road_map_shortest_distance(record):
    """Independent Dijkstra (scipy) from the start to the goal sample over
    the same directed road map."""
    xs, ys, road_map = record["sample_x"], record["sample_y"], record["road_map"]
    rows, cols, weights = [], [], []
    for i, neighbours in enumerate(road_map):
        for j in neighbours:
            rows.append(i)
            cols.append(j)
            weights.append(math.hypot(xs[j] - xs[i], ys[j] - ys[i]))
    n = len(road_map)
    graph = csr_matrix((weights, (rows, cols)), shape=(n, n))
    # sample_points() appends the start and then the goal to the samples
    return dijkstra(graph, indices=n - 2)[n - 1]


@pytest.fixture(scope="module")
def demo():
    return run_demo()


def test1(demo):
    path, record = demo
    assert path, "no path found"
    # the path is listed from the goal back to the start
    assert path[0] == record["goal"]
    assert path[-1] == record["start"]
    # every road map vertex keeps the robot radius from the obstacles
    for x, y in path:
        assert min(math.hypot(x - ox, y - oy)
                   for ox, oy, _ in record["obstacles"]) >= record["rr"]
    # edges are only checked every robot radius (5 m) along their length;
    # the path cuts 0.074 m into the radius (see the xfail test below),
    # allow twice that
    assert pc.min_clearance(path, record["obstacles"], record["rr"]) >= -0.15
    # the path is a shortest path on the road map it searched
    assert pc.path_length(path) == pytest.approx(
        road_map_shortest_distance(record), rel=1e-12)
    assert pc.path_length(path) >= math.dist(record["start"], record["goal"])


def test1_same_result_every_run(demo):
    path, _ = run_demo()
    assert path
    assert path == demo[0]


@pytest.mark.xfail(
    strict=True, raises=AssertionError,
    reason="is_collision() checks an edge only every robot radius (5 m) "
           "along its length, so road map edges can pass closer than the "
           "robot radius to an obstacle (0.074 m closer in this run)")
def test_path_keeps_robot_radius_clearance(demo):
    path, record = demo
    assert pc.min_clearance(path, record["obstacles"], record["rr"]) >= 0.0


if __name__ == '__main__':
    conftest.run_this_test(__file__)
