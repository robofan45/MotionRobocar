"""
Shared answer checks for the sampling based and curve path planner tests.

Paths are lists of points whose first two entries are x and y. Obstacles are
circles given as (x, y, radius).
"""
import math

import numpy as np


def xy(path):
    """Return the path as an (n, 2) float array of x, y."""
    return np.array([[p[0], p[1]] for p in path], dtype=float)


def path_length(path):
    steps = np.diff(xy(path), axis=0)
    return float(np.sum(np.hypot(steps[:, 0], steps[:, 1])))


def max_step(path):
    steps = np.diff(xy(path), axis=0)
    return float(np.max(np.hypot(steps[:, 0], steps[:, 1])))


def distance_to_path(path, point):
    """Exact distance from a point to the polyline (closest point on any
    segment)."""
    points = xy(path)
    a, b = points[:-1], points[1:]
    ab = b - a
    ab2 = np.sum(ab * ab, axis=1)
    t = np.zeros(len(a))
    nonzero = ab2 > 0.0
    t[nonzero] = (np.sum((np.asarray(point) - a) * ab, axis=1)[nonzero]
                  / ab2[nonzero])
    closest = a + np.clip(t, 0.0, 1.0)[:, None] * ab
    return float(np.min(np.hypot(*(closest - point).T)))


def min_clearance(path, obstacles, robot_radius=0.0):
    """Smallest gap between the robot disc and any circle obstacle along the
    whole path (vertices and segments); negative means the path cuts into an
    obstacle."""
    return min(distance_to_path(path, (ox, oy)) - size - robot_radius
               for (ox, oy, size) in obstacles)


def shortest_detour(a, b, circle):
    """Length of the shortest way from a to b around one circle obstacle:
    the straight line if it misses the circle, otherwise tangent, arc,
    tangent. Any collision free path from a to b is at least this long."""
    cx, cy, r = circle
    straight = math.dist(a, b)
    if distance_to_path([a, b], (cx, cy)) >= r:
        return straight
    d1, d2 = math.dist(a, (cx, cy)), math.dist(b, (cx, cy))
    angle_acb = abs(math.remainder(math.atan2(a[1] - cy, a[0] - cx)
                                   - math.atan2(b[1] - cy, b[0] - cx),
                                   2.0 * math.pi))
    arc = angle_acb - math.acos(r / d1) - math.acos(r / d2)
    return math.sqrt(d1 ** 2 - r ** 2) + math.sqrt(d2 ** 2 - r ** 2) + r * arc


def chord_depth(radius, spacing):
    """How deep a straight chord between two points that are just outside a
    circle of `radius`, `spacing` apart, can reach into the circle.

    Planners that check collisions only at points `spacing` apart guarantee
    no more clearance than minus this value between those points.
    """
    half = spacing / 2.0
    if half >= radius:
        return radius  # the chord can pass through the centre
    return radius - math.sqrt(radius ** 2 - half ** 2)


def assert_collision_free(path, obstacles, robot_radius=0.0, check_spacing=0.0):
    """Assert that the robot never cuts into an obstacle, allowing only the
    chord depth that the planner's own check spacing cannot see."""
    allowance = max((chord_depth(size + robot_radius, check_spacing)
                     for (_, _, size) in obstacles), default=0.0)
    clearance = min_clearance(path, obstacles, robot_radius)
    assert clearance > -allowance, (
        f"path cuts {-clearance:.4f} m into an obstacle "
        f"(allowed {allowance:.4f} m by the check spacing)")


def drop_repeated_points(path, eps=1e-9):
    points = xy(path)
    keep = [0]
    for i in range(1, len(points)):
        if math.hypot(*(points[i] - points[keep[-1]])) > eps:
            keep.append(i)
    return points[keep]


def headings(path):
    """Direction of each non degenerate step of the path."""
    steps = np.diff(drop_repeated_points(path), axis=0)
    return np.arctan2(steps[:, 1], steps[:, 0])


def step_lengths(path):
    """Length of each non degenerate step of the path."""
    steps = np.diff(drop_repeated_points(path), axis=0)
    return np.hypot(steps[:, 0], steps[:, 1])


def turning_curvature(path):
    """Heading change per metre between consecutive steps of the path."""
    points = drop_repeated_points(path)
    steps = np.diff(points, axis=0)
    lengths = np.hypot(steps[:, 0], steps[:, 1])
    turn = np.angle(np.exp(1j * np.diff(np.arctan2(steps[:, 1], steps[:, 0]))))
    return np.abs(turn) / (0.5 * (lengths[:-1] + lengths[1:]))


def angle_diff(a, b):
    """Smallest signed difference a - b of two angles."""
    return math.remainder(a - b, 2.0 * math.pi)


def assert_dubins_path(path, planner):
    """Answer checks for the RRT planners that steer with Dubins paths."""
    start, goal = planner.start, planner.end
    assert path is not None, "no path found"
    # the path is listed from the goal back to the start; after the goal
    # itself come the Dubins path points, from the tree node that reached
    # the goal region back to the start
    assert path[0] == [goal.x, goal.y]
    assert path[-1] == [start.x, start.y]
    assert math.dist(path[1], path[0]) <= planner.goal_xy_th
    dubins_points = path[1:]

    # every Dubins path point is collision checked
    assert_collision_free(path, planner.obstacle_list, planner.robot_radius,
                          check_spacing=max_step(dubins_points))
    # Dubins path points are at most two 0.1 m steps apart: no jumps
    assert max_step(dubins_points) < 0.2
    assert path_length(path) >= math.dist(path[0], path[-1])

    # never turns tighter than the minimum turning radius 1 / curvature;
    # for curvature 1.0 the estimate on 0.1 m steps reads up to 1.0012
    assert turning_curvature(dubins_points).max() <= planner.curvature * 1.0025

    # leaves along the start heading and arrives along the goal heading:
    # a chord of length h on a curve of curvature k turns away from the
    # tangent at its ends by at most asin(k * h / 2)
    forward = dubins_points[::-1]
    steps, directions = step_lengths(forward), headings(forward)
    assert abs(angle_diff(directions[0], start.yaw)) <= \
        math.asin(planner.curvature * steps[0] / 2) + 1e-9
    assert abs(angle_diff(directions[-1], goal.yaw)) <= \
        planner.goal_yaw_th + math.asin(planner.curvature * steps[-1] / 2)
