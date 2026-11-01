import conftest  # Add root path to sys.path
import math

import numpy as np
import pytest

from PathTracking.lqr_speed_steer_control import lqr_speed_steer_control as m

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


def test_1(monkeypatch):
    monkeypatch.setattr(m, "show_animation", False)
    t, x, y, yaw, v, cx, cy = m.main()

    # The run ends at the goal (last waypoint), not at the 500 s time limit.
    # It stops within 0.3 m of the goal; the last stored pose is one step
    # before that.
    assert t[-1] < 500.0  # observed 19.4 s
    assert math.hypot(x[-1] - cx[-1], y[-1] - cy[-1]) < 0.5  # observed 0.258 m

    # Lateral tracking error w.r.t. the spline course.
    assert distance_to_polyline(x, y, cx, cy).max() < 0.45  # observed 0.224 m

    # Speed control: reaches the target speed, then slows down for the goal.
    assert max(v) == pytest.approx(TARGET_SPEED, rel=0.01)
    assert v[-1] < 0.65  # observed 0.33 m/s (profile ends at 1 km/h)


def test_speed_profile_slows_down_only_at_the_end():
    cyaw = [0.0] * 100  # straight course
    speed_profile = m.calc_speed_profile(cyaw, TARGET_SPEED)
    assert speed_profile[0] == pytest.approx(TARGET_SPEED)


if __name__ == '__main__':
    conftest.run_this_test(__file__)
