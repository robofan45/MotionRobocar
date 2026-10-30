import math

import numpy as np
import pytest
from numpy.polynomial import polynomial

import conftest  # Add root path to sys.path
from PathPlanning.QuinticPolynomialsPlanner import quintic_polynomials_planner as m

# exact identities hold up to round-off (observed below 1e-13)
ROUND_OFF = 1e-9


def run_demo():
    """Run main() without animation; return the planner's arguments and
    result."""
    calls = []
    planner = m.quintic_polynomials_planner

    def recording_planner(*args):
        result = planner(*args)
        calls.append((args, result))
        return result

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m, "show_animation", False)
        mp.setattr(m, "quintic_polynomials_planner", recording_planner)
        m.main()
    assert len(calls) == 1
    return calls[0]


def quintic(p0, v0, a0, p1, v1, a1, horizon):
    """Coefficients (lowest order first) of the quintic with the given
    position, velocity and acceleration at t = 0 and t = horizon, from the
    full 6 x 6 system."""
    rows = []
    for t in (0.0, horizon):
        rows.append([t ** k for k in range(6)])
        rows.append([k * t ** (k - 1) if k else 0.0 for k in range(6)])
        rows.append([k * (k - 1) * t ** (k - 2) if k > 1 else 0.0
                     for k in range(6)])
    return np.linalg.solve(np.array(rows), [p0, v0, a0, p1, v1, a1])


def trajectory(args, horizon, times):
    """Position, velocity, acceleration and jerk (x and y rows) at `times`
    of the quintic for the planner arguments `args` and this horizon."""
    sx, sy, syaw, sv, sa, gx, gy, gyaw, gv, ga = args[:10]
    cx = quintic(sx, sv * math.cos(syaw), sa * math.cos(syaw),
                 gx, gv * math.cos(gyaw), ga * math.cos(gyaw), horizon)
    cy = quintic(sy, sv * math.sin(syaw), sa * math.sin(syaw),
                 gy, gv * math.sin(gyaw), ga * math.sin(gyaw), horizon)
    return [np.array([polynomial.polyval(times, polynomial.polyder(c, k))
                      for c in (cx, cy)]) for k in range(4)]


@pytest.fixture(scope="module")
def demo():
    return run_demo()


def test1(demo):
    args, (time, x, y, yaw, v, a, j) = demo
    sx, sy, syaw, sv, sa, gx, gy, gyaw, gv, ga, max_accel, max_jerk, dt = args

    # start and goal states
    assert (x[0], y[0], v[0]) == (sx, sy, sv)
    assert yaw[0] == pytest.approx(syaw, abs=ROUND_OFF)
    assert a[0] == pytest.approx(sa, abs=ROUND_OFF)
    assert x[-1] == pytest.approx(gx, abs=ROUND_OFF)
    assert y[-1] == pytest.approx(gy, abs=ROUND_OFF)
    assert yaw[-1] == pytest.approx(gyaw, abs=ROUND_OFF)
    assert v[-1] == pytest.approx(gv, abs=ROUND_OFF)
    assert a[-1] == pytest.approx(ga, abs=ROUND_OFF)

    # within the limits
    assert max(abs(i) for i in a) <= max_accel
    assert max(abs(i) for i in j) <= max_jerk

    # the same trajectory from an independently solved quintic
    horizon = time[-1]
    np.testing.assert_allclose(np.diff(time), dt, atol=ROUND_OFF)
    position, velocity, acceleration, jerk = trajectory(args, horizon,
                                                        np.array(time))
    np.testing.assert_allclose(x, position[0], atol=ROUND_OFF)
    np.testing.assert_allclose(y, position[1], atol=ROUND_OFF)
    np.testing.assert_allclose(v, np.hypot(*velocity), atol=ROUND_OFF)
    np.testing.assert_allclose(np.abs(a), np.hypot(*acceleration),
                               atol=ROUND_OFF)
    np.testing.assert_allclose(np.abs(j), np.hypot(*jerk), atol=ROUND_OFF)

    # the planner tries horizons MIN_T, 2 MIN_T, ... and keeps the first
    # within the limits, so the one before breaks a limit
    assert horizon - m.MIN_T >= m.MIN_T
    times = np.arange(0.0, horizon - m.MIN_T + dt, dt)
    _, _, acceleration, jerk = trajectory(args, horizon - m.MIN_T, times)
    assert (np.max(np.hypot(*acceleration)) > max_accel
            or np.max(np.hypot(*jerk)) > max_jerk)


if __name__ == '__main__':
    conftest.run_this_test(__file__)
