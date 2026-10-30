import conftest  # Add root path to sys.path
import math

import numpy as np
import pytest

from PathTracking.pure_pursuit import pure_pursuit as m

TARGET_SPEED = 10.0 / 3.6  # [m/s] target speed used by main()


def distance_to_polyline(px, py, cx, cy):
    """Shortest distance from every point (px[i], py[i]) to the polyline."""
    points = np.column_stack([px, py])[:, None, :]
    course = np.column_stack([cx, cy])
    start, end = course[:-1], course[1:]
    keep = np.any(start != end, axis=1)  # skip zero-length segments
    start, end = start[keep], end[keep]
    segment = end - start
    t = np.sum((points - start) * segment, axis=2) / np.sum(segment ** 2, axis=1)
    closest = start + np.clip(t, 0.0, 1.0)[..., None] * segment
    return np.linalg.norm(points - closest, axis=2).min(axis=1)


def run_and_check(monkeypatch, reverse, max_lateral, rms_lateral, end_distance):
    monkeypatch.setattr(m, "show_animation", False)
    monkeypatch.setattr(m, "is_reverse_mode", reverse)
    states, cx, cy = m.main()

    # Analytic course y = sin(x / 5) * x / 2, sampled much finer than cx.
    course_x = np.linspace(0.0, cx[-1], 2000)
    course_y = np.sin(course_x / 5.0) * course_x / 2.0
    # Pure pursuit steers the rear axle point onto the course.
    direction = np.array(states.direction)
    yaw = np.array(states.yaw)
    rear_x = np.array(states.x) - direction * m.WB / 2 * np.cos(yaw)
    rear_y = np.array(states.y) - direction * m.WB / 2 * np.sin(yaw)

    # The course end is reached before the 100 s time limit.
    assert states.t[-1] < 100.0
    assert math.hypot(rear_x[-1] - course_x[-1],
                      rear_y[-1] - course_y[-1]) < end_distance

    # Lateral error once the 3 m start offset is corrected (first 3 s).
    # Pure pursuit cuts the tightest bends, which are sharper than the
    # vehicle's minimum turning radius.
    lateral = distance_to_polyline(rear_x, rear_y, course_x, course_y)
    lateral = lateral[np.array(states.t) > 3.0]
    assert lateral.max() < max_lateral
    assert np.sqrt(np.mean(lateral ** 2)) < rms_lateral

    # The P speed controller holds the target speed.
    assert max(states.v) == pytest.approx(TARGET_SPEED, rel=0.01)
    assert states.v[-1] == pytest.approx(TARGET_SPEED, rel=0.01)
    assert set(states.direction) == {-1 if reverse else 1}


def test1(monkeypatch):
    # observed: end 4.26 m, lateral max 2.37 m, rms 0.71 m
    run_and_check(monkeypatch, reverse=False,
                  max_lateral=4.8, rms_lateral=1.4, end_distance=8.5)


def test_backward(monkeypatch):
    # observed: end 4.20 m, lateral max 1.08 m, rms 0.32 m
    run_and_check(monkeypatch, reverse=True,
                  max_lateral=2.2, rms_lateral=0.65, end_distance=8.5)


if __name__ == '__main__':
    conftest.run_this_test(__file__)
