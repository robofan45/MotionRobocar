import conftest  # Add root path to sys.path
import math

import numpy as np
import pytest

from PathTracking.model_predictive_speed_and_steer_control \
    import model_predictive_speed_and_steer_control as m


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


# Each simulation takes 10-30 s, so every scenario runs once per module and
# the tests share the result.
@pytest.fixture(scope="module")
def switch_back_run():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m, "show_animation", False)
        return m.main()


@pytest.fixture(scope="module")
def backward_run():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m, "show_animation", False)
        return m.main2()


def goal_of(course):
    """main() and main2() use a 1 m course tick; the goal is its last point."""
    cx, cy, _, _ = course(1.0)
    return cx[-1], cy[-1]


def check_inputs_within_limits(d, a):
    # hard constraints of the QP (1e-6: solver tolerance)
    assert np.abs(d).max() <= m.MAX_STEER + 1e-6
    assert np.abs(a).max() <= m.MAX_ACCEL + 1e-6


def test_1(switch_back_run):
    """main(): forward along the switch-back course, then reverse to the end."""
    t, x, y, yaw, v, d, a = switch_back_run
    cx, cy, _, _ = m.get_switch_back_course(0.1)  # same course, sampled densely
    goal_x, goal_y = goal_of(m.get_switch_back_course)

    # Stopped at the goal before the time limit.
    assert t[-1] < m.MAX_TIME  # observed 69.2 s
    assert math.hypot(x[-1] - goal_x, y[-1] - goal_y) < 0.3  # observed 0.135 m
    assert abs(v[-1]) <= m.STOP_SPEED

    # The car tracks the course closely except when cutting the hairpin and
    # turning around at the cusps.
    lateral = distance_to_polyline(x, y, cx, cy)
    assert np.median(lateral) < 0.025  # observed 0.012 m
    assert lateral.max() < 5.5  # observed 2.71 m

    # Reaches the target speed driving forward and driving backward.
    assert max(v) > 0.9 * m.TARGET_SPEED  # observed 4.31 m/s
    assert min(v) < -0.9 * m.TARGET_SPEED  # observed -3.76 m/s

    check_inputs_within_limits(d, a)


def test_2(backward_run):
    """main2(): drive the wavy straight course backwards."""
    t, x, y, yaw, v, d, a = backward_run
    cx, cy, _, _ = m.get_straight_course3(0.1)  # same course, sampled densely
    goal_x, goal_y = goal_of(m.get_straight_course3)

    # Stopped at the goal before the time limit.
    assert t[-1] < m.MAX_TIME  # observed 31.4 s
    assert math.hypot(x[-1] - goal_x, y[-1] - goal_y) < 0.25  # observed 0.121 m
    assert abs(v[-1]) <= m.STOP_SPEED

    lateral = distance_to_polyline(x, y, cx, cy)
    assert np.median(lateral) < 0.016  # observed 0.0077 m
    assert lateral.max() < 2.5  # observed 1.24 m (overshoot at the end)

    # The whole course is driven in reverse at the target speed.
    assert min(v) < -0.9 * m.TARGET_SPEED  # observed -3.05 m/s
    assert np.mean(np.array(v) < 0.0) > 0.8  # observed 0.88

    check_inputs_within_limits(d, a)


@pytest.mark.xfail(raises=AssertionError, strict=True, reason=(
    "MAX_DSTEER only limits steering changes inside the MPC horizon; the "
    "first input is not linked to the previously applied steering "
    "(State.predelta is never used), so applied steering jumps faster than "
    "the limit (observed 3.6 rad/s vs 0.52 rad/s)"))
def test_applied_steering_rate_within_limit(switch_back_run):
    t, x, y, yaw, v, d, a = switch_back_run
    steering_rate = np.abs(np.diff(d)) / m.DT
    assert steering_rate.max() <= m.MAX_DSTEER + 1e-6


if __name__ == '__main__':
    conftest.run_this_test(__file__)
