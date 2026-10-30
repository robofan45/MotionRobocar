import conftest  # Add root path to sys.path
import math

import numpy as np
import pytest

from PathTracking.lqr_steer_control import lqr_steer_control as m

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


def test1(monkeypatch):
    monkeypatch.setattr(m, "show_animation", False)
    t, x, y, yaw, v, cx, cy = m.main()

    # The run ends at the goal (last waypoint), not at the 500 s time limit.
    # It stops within 0.3 m of the goal; the last stored pose is one step
    # before that.
    assert t[-1] < 500.0  # observed 17.6 s
    assert math.hypot(x[-1] - cx[-1], y[-1] - cy[-1]) < 0.6  # observed 0.29 m

    # Lateral tracking error w.r.t. the spline course.
    assert distance_to_polyline(x, y, cx, cy).max() < 0.5  # observed 0.247 m

    # The PID speed controller reaches the target speed.
    assert max(v) == pytest.approx(TARGET_SPEED, rel=0.01)


if __name__ == '__main__':
    conftest.run_this_test(__file__)
