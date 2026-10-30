import conftest  # Add root path to sys.path
import math

import numpy as np
import pytest

from PathTracking.stanley_control import stanley_control as m

TARGET_SPEED = 30.0 / 3.6  # [m/s] target speed used by main()


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

    # The front axle (the point Stanley steers) reaches the course end
    # before the 100 s time limit.
    front_x = np.array(x) + m.L * np.cos(yaw)
    front_y = np.array(y) + m.L * np.sin(yaw)
    assert t[-1] < 100.0  # observed 27.3 s
    assert math.hypot(front_x[-1] - cx[-1], front_y[-1] - cy[-1]) < 2.4  # observed 1.20 m

    # The car starts 4.7 m beside the course; once that offset is corrected
    # (6 s) the rear axle stays close to the course (off-tracking in bends).
    lateral = distance_to_polyline(x, y, cx, cy)
    assert lateral[np.array(t) > 6.0].max() < 0.6  # observed 0.31 m

    # The P speed controller reaches the target speed.
    assert max(v) == pytest.approx(TARGET_SPEED, rel=0.01)


if __name__ == '__main__':
    conftest.run_this_test(__file__)
