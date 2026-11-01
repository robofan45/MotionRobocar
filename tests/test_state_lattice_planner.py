import math

import numpy as np
import pytest

import conftest  # Add root path to sys.path
import path_checks as pc
from PathPlanning.StateLatticePlanner import state_lattice_planner as m

motion_model = m.motion_model


def run_demo():
    """Run main() without animation; return every generate_path() call
    (targets, k0, results) and every trajectory optimisation
    (target, k0, x, y, yaw, parameters) it made."""
    path_calls, optimisations = [], []
    generate_path = m.generate_path
    optimize_trajectory = m.planner.optimize_trajectory

    def recording_generate_path(target_states, k0):
        results = generate_path(target_states, k0)
        path_calls.append((target_states, k0, results))
        return results

    def recording_optimize_trajectory(target, k0, p):
        x, y, yaw, p = optimize_trajectory(target, k0, p)
        optimisations.append(((target.x, target.y, target.yaw), k0,
                              x, y, yaw, None if p is None else p.copy()))
        return x, y, yaw, p

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m, "show_animation", False)
        mp.setattr(m.planner, "show_animation", False)
        mp.setattr(m, "generate_path", recording_generate_path)
        mp.setattr(m.planner, "optimize_trajectory",
                   recording_optimize_trajectory)
        m.main()
    return path_calls, optimisations


def test1():
    path_calls, optimisations = run_demo()

    # five sampling demos; every target state gets a trajectory
    assert len(path_calls) == 5
    for targets, _, results in path_calls:
        assert len(results) == len(targets) > 0
    results = [row for _, _, rows in path_calls for row in rows]
    assert len(optimisations) == len(results)

    for (target, k0, x, y, yaw, p), result in zip(optimisations, results):
        assert x is not None, f"no trajectory to {target}"
        # starts at the origin, heading along x, steering at k0
        assert (x[0], y[0], yaw[0]) == (0.0, 0.0, 0.0)
        assert yaw[1] == pytest.approx(
            math.tan(k0) * motion_model.ds / motion_model.L, abs=1e-12)
        # constant speed: every step is ds long
        np.testing.assert_allclose(pc.step_lengths(list(zip(x, y))),
                                   motion_model.ds, rtol=1e-9)
        # ends at the target within the optimiser's acceptance cost_th
        miss = np.hypot(np.hypot(target[0] - x[-1], target[1] - y[-1]),
                        pc.angle_diff(target[2], yaw[-1]))
        assert miss <= m.planner.cost_th
        # generate_path() reports this end pose and the parameters
        # (arc length, middle and final steering) that produced it
        assert result == [x[-1], y[-1], yaw[-1], *p[:, 0]]


def test_sampled_terminal_states():
    # uniform polar sampling: nxy positions on a circle of radius d from
    # a_min to a_max, each with nh headings
    d, a_min, a_max = 20.0, -np.deg2rad(45.0), np.deg2rad(45.0)
    states = m.calc_uniform_polar_states(5, 3, d, a_min, a_max,
                                         -np.deg2rad(45.0), np.deg2rad(45.0))
    assert len(states) == 5 * 3
    angles = [math.atan2(sy, sx) for sx, sy, _ in states]
    assert all(math.hypot(sx, sy) == pytest.approx(d) for sx, sy, _ in states)
    np.testing.assert_allclose(sorted(set(np.round(angles, 12))),
                               np.linspace(a_min, a_max, 5), atol=1e-12)
    # lane sampling: nxy lateral offsets across the lane, less the vehicle
    # width, all at the lane heading
    states = m.calc_lane_states(10.0, 0.0, 3.0, 1.0, 10.0, 5)
    np.testing.assert_allclose(states, [[10.0, y, 0.0] for y in
                                        np.linspace(9.0, 11.0, 5)])


def test_biased_terminal_states_cover_the_angle_range():
    nxy, nh, ns = 30, 2, 100
    a_min, a_max = np.deg2rad(-45.0), np.deg2rad(45.0)
    states = m.calc_biased_polar_states(0.0, ns, nxy, nh, 20.0, a_min, a_max,
                                        -np.deg2rad(20.0), np.deg2rad(20.0))
    assert len(states) == nxy * nh
    angles = [math.atan2(sy, sx) for sx, sy, _ in states]
    assert max(angles) == pytest.approx(a_max, abs=(a_max - a_min) / ns)


if __name__ == '__main__':
    conftest.run_this_test(__file__)
