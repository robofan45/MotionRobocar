import math

import numpy as np
import pytest

import conftest
import path_checks as pc
from PathPlanning.BezierPath import bezier_path as m


def run_demo(main):
    """Run a demo without animation; return every
    ((sx, sy, syaw, ex, ey, eyaw, offset), path, control_points) it built."""
    built = []
    calc_4points_bezier_path = m.calc_4points_bezier_path

    def recording(sx, sy, syaw, ex, ey, eyaw, offset):
        path, control_points = calc_4points_bezier_path(
            sx, sy, syaw, ex, ey, eyaw, offset)
        built.append(((sx, sy, syaw, ex, ey, eyaw, offset),
                      path, control_points))
        return path, control_points

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m, "show_animation", False)
        mp.setattr(m, "calc_4points_bezier_path", recording)
        main()
    return built


def de_casteljau(control_points, t):
    """Independent evaluation of a Bezier curve by repeated interpolation."""
    points = np.array(control_points, dtype=float)
    while len(points) > 1:
        points = (1 - t) * points[:-1] + t * points[1:]
    return points[0]


def cubic_derivatives(p, t):
    """First and second derivatives of a cubic Bezier curve, written out."""
    d1 = 3 * ((1 - t) ** 2 * (p[1] - p[0]) + 2 * (1 - t) * t * (p[2] - p[1])
              + t ** 2 * (p[3] - p[2]))
    d2 = 6 * ((1 - t) * (p[2] - 2 * p[1] + p[0]) + t * (p[3] - 2 * p[2] + p[1]))
    return d1, d2


def check_curve(args, path, control_points):
    sx, sy, syaw, ex, ey, eyaw, offset = args
    p = control_points
    # ends exactly at the given points
    assert (path[0][0], path[0][1]) == (sx, sy)
    assert (path[-1][0], path[-1][1]) == (ex, ey)
    # the inner control points sit |end - start| / offset along the start
    # heading and behind the end heading, so the curve leaves and arrives
    # along those headings
    arm = math.hypot(ex - sx, ey - sy) / offset
    for vector, yaw in ((p[1] - p[0], syaw), (p[3] - p[2], eyaw)):
        assert np.linalg.norm(vector) == pytest.approx(arm, rel=1e-12)
        assert pc.angle_diff(math.atan2(vector[1], vector[0]), yaw) == \
            pytest.approx(0.0, abs=1e-12)
    # 100 points of the cubic curve at evenly spaced parameters
    expected = [de_casteljau(p, t) for t in np.linspace(0.0, 1.0, len(path))]
    np.testing.assert_allclose(path, expected, rtol=0, atol=1e-12)
    assert pc.path_length(path) >= math.hypot(ex - sx, ey - sy)


def test_1():
    built = run_demo(m.main)
    assert len(built) == 1
    check_curve(*built[0])


def test_2():
    built = run_demo(m.main2)
    assert [args[-1] for args, _, _ in built] == [1.0, 2.0, 3.0, 4.0]
    for curve in built:
        check_curve(*curve)
    # a larger offset gives shorter control arms and a shorter curve
    lengths = [pc.path_length(path) for _, path, _ in built]
    assert lengths == sorted(lengths, reverse=True)


@pytest.mark.parametrize("t", [0.0, 0.25, 0.5, 0.86, 1.0])
def test_derivatives_and_curvature(t):
    # the curve of the demo, whose curvature main() draws at t = 0.86
    _, control_points = m.calc_4points_bezier_path(
        10.0, 1.0, np.radians(180.0), 0.0, -3.0, np.radians(-45.0), 3.0)
    derivative_points = m.bezier_derivatives_control_points(control_points, 2)
    d1, d2 = cubic_derivatives(control_points, t)
    np.testing.assert_allclose(m.bezier(t, derivative_points[1]), d1,
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(m.bezier(t, derivative_points[2]), d2,
                               rtol=1e-12, atol=1e-12)
    expected = (d1[0] * d2[1] - d1[1] * d2[0]) / np.linalg.norm(d1) ** 3
    assert m.curvature(d1[0], d1[1], d2[0], d2[1]) == \
        pytest.approx(expected, rel=1e-12)


if __name__ == '__main__':
    conftest.run_this_test(__file__)
