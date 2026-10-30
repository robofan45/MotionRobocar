import numpy as np
import pytest
from numpy.polynomial import polynomial

import conftest
import path_checks as pc
from PathPlanning.Eta3SplinePath import eta3_spline_path as m

# exact identities hold up to round-off (observed below 3e-13)
ROUND_OFF = 1e-9


def run_demo():
    """Run main() without animation; return every segment (with its eta and
    kappa) and every path it built, in order."""
    segments, paths = [], []

    class RecordingSegment(m.Eta3PathSegment):
        def __init__(self, start_pose, end_pose, eta=None, kappa=None):
            super().__init__(start_pose, end_pose, eta=eta, kappa=kappa)
            self.eta, self.kappa = eta, kappa
            segments.append(self)

    class RecordingPath(m.Eta3Path):
        def __init__(self, path_segments):
            super().__init__(path_segments)
            paths.append(self)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m, "show_animation", False)
        mp.setattr(m, "Eta3PathSegment", RecordingSegment)
        mp.setattr(m, "Eta3Path", RecordingPath)
        m.main()
    return segments, paths


@pytest.fixture(scope="module")
def demo():
    return run_demo()


def derivative(segment, u, order=0):
    """x and y (rows) of the segment's polynomial, or of its derivative of
    the given order, at u (a number or an array)."""
    return np.array([polynomial.polyval(u, polynomial.polyder(c, order))
                     for c in segment.coeffs])


def geometry(segment, u):
    """Heading, curvature and curvature rate d(kappa)/ds at u (a number or
    an array) where the segment's speed is not zero."""
    d1, d2, d3 = (derivative(segment, u, k) for k in (1, 2, 3))
    speed = np.hypot(d1[0], d1[1])
    cross12 = d1[0] * d2[1] - d1[1] * d2[0]
    cross13 = d1[0] * d3[1] - d1[1] * d3[0]
    dot12 = d1[0] * d2[0] + d1[1] * d2[1]
    kappa = cross12 / speed ** 3
    dkappa_du = cross13 / speed ** 3 - 3 * cross12 * dot12 / speed ** 5
    return np.arctan2(d1[1], d1[0]), kappa, dkappa_du / speed


def test_1(demo):
    segments, paths = demo
    # test1() and test2() build 10 one-segment paths each, test3() the
    # five-segment reference path
    assert [len(p.segments) for p in paths] == [1] * 20 + [5]
    assert len(segments) == 25

    for segment in segments:
        start, end = segment.start_pose, segment.end_pose
        np.testing.assert_allclose(segment.calc_point(0.0), start[:2],
                                   atol=ROUND_OFF)
        np.testing.assert_allclose(segment.calc_point(1.0), end[:2],
                                   atol=ROUND_OFF)
        # heading, curvature and curvature rate are set at an end whose
        # speed parameter eta is positive (zero speed leaves them undefined)
        for u, pose, eta, kappa, dkappa in (
                (0.0, start, segment.eta[0], segment.kappa[0], segment.kappa[1]),
                (1.0, end, segment.eta[1], segment.kappa[2], segment.kappa[3])):
            if eta > 0:
                heading, k, dk = geometry(segment, u)
                assert abs(pc.angle_diff(heading, pose[2])) < ROUND_OFF
                assert k == pytest.approx(kappa, abs=ROUND_OFF)
                assert dk == pytest.approx(dkappa, abs=ROUND_OFF)


def test_reference_path(demo):
    _, paths = demo
    path = paths[-1]
    lane_change, line, spiral, twirl, arc = path.segments

    # continuous position, heading, curvature and curvature rate where two
    # segments with positive end speeds meet
    for a, b in zip(path.segments[:-1], path.segments[1:]):
        np.testing.assert_allclose(a.calc_point(1.0), b.calc_point(0.0),
                                   atol=ROUND_OFF)
        if a.eta[1] > 0 and b.eta[0] > 0:
            ha, ka, dka = geometry(a, 1.0)
            hb, kb, dkb = geometry(b, 0.0)
            assert abs(pc.angle_diff(ha, hb)) < ROUND_OFF
            assert ka == pytest.approx(kb, abs=ROUND_OFF)
            assert dka == pytest.approx(dkb, abs=ROUND_OFF)

    u = np.linspace(0.0, 1.0, 1001)
    # the line segment stays on y = 1.5 and moves forward from x = 4 to 5.5
    points = derivative(line, u)
    np.testing.assert_allclose(points[1], 1.5, atol=ROUND_OFF)
    assert np.all(np.diff(points[0]) >= 0.0)
    # the lane change has symmetric end conditions, so the curve is point
    # symmetric about the middle of its chord
    chord_middle_twice = np.add(lane_change.start_pose[:2],
                                lane_change.end_pose[:2])[:, None]
    np.testing.assert_allclose(
        derivative(lane_change, u) + derivative(lane_change, 1.0 - u),
        np.broadcast_to(chord_middle_twice, (2, len(u))), atol=ROUND_OFF)
    # the circular arc segment approximates radius 2 m: its curvature stays
    # within 0.0034 of 0.5 and its length within 0.0028 m of the arc's;
    # allow twice those
    turn = arc.end_pose[2] - arc.start_pose[2]
    assert np.max(np.abs(geometry(arc, u)[1] - 0.5)) < 0.007
    assert arc.segment_length == pytest.approx(turn / 0.5, abs=0.006)

    # each segment's integrated length matches its densely sampled polyline
    for segment in path.segments:
        polyline = derivative(segment, u).T
        assert segment.segment_length == pytest.approx(
            pc.path_length(polyline), rel=1e-5)


if __name__ == '__main__':
    conftest.run_this_test(__file__)
