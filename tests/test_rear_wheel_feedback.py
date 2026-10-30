import conftest  # Add root path to sys.path
import math

import numpy as np
import pytest

from PathTracking.rear_wheel_feedback_control import rear_wheel_feedback_control as m

TARGET_SPEED = 10.0 / 3.6  # [m/s] target speed of calc_target_speed()


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


def test1(monkeypatch):
    monkeypatch.setattr(m, "show_animation", False)
    t, x, y, yaw, v, goal_flag, reference_path = m.main()

    s = np.linspace(0.0, reference_path.length, 2000)
    course_x, course_y = reference_path.X(s), reference_path.Y(s)

    # The run ends at the goal (end of the spline), not at the time limit.
    # It stops within 0.3 m of the goal; the last stored pose is one step
    # before that.
    assert goal_flag
    assert t[-1] < 500.0  # observed 30.7 s
    assert math.hypot(x[-1] - course_x[-1], y[-1] - course_y[-1]) < 0.86  # observed 0.43 m

    # Lateral tracking error w.r.t. the spline course.
    lateral = distance_to_polyline(x, y, course_x, course_y)
    assert lateral.max() < 2.3  # observed 1.15 m (sharp bends)
    assert np.sqrt(np.mean(lateral ** 2)) < 0.65  # observed 0.32 m

    # The P speed controller reaches the target speed.
    assert max(v) == pytest.approx(TARGET_SPEED, rel=0.01)


if __name__ == '__main__':
    conftest.run_this_test(__file__)
