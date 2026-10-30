import ast
import math
import pathlib

import numpy as np
import pytest

import conftest
import path_checks as pc
from PathPlanning.FrenetOptimalTrajectory import frenet_optimal_trajectory as m
from PathPlanning.FrenetOptimalTrajectory.frenet_optimal_trajectory import (
    LateralMovement,
    LongitudinalMovement,
)

# exact identities hold up to round-off
ROUND_OFF = 1e-9
# the high speed strategy converts d'(s) and d''(s) with 1e-6 guards against
# division by zero; the start states differ by up to 2.6e-7 because of that
LATERAL_DERIVATIVE_TOL = 6e-7


def scenario_settings(lateral, longitudinal):
    """Module settings for a scenario, the same as editing LATERAL_MOVEMENT
    and LONGITUDINAL_MOVEMENT at the top of the module would give.

    The module picks its parameters and strategies once, at import time, so
    assigning m.LATERAL_MOVEMENT afterwards changes nothing. The parameters
    of the lateral movement are read from the module's own source."""
    tree = ast.parse(pathlib.Path(m.__file__).read_text(encoding="utf-8"))
    branch = next(node for node in tree.body if isinstance(node, ast.If)
                  and ast.unparse(node.test)
                  == "LATERAL_MOVEMENT == LateralMovement.LOW_SPEED")
    body = branch.body if lateral is LateralMovement.LOW_SPEED else branch.orelse
    settings = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), m.__file__, "exec"),
         {"np": np}, settings)
    settings["LATERAL_MOVEMENT_STRATEGY"] = (
        m.LowSpeedLateralMovementStrategy()
        if lateral is LateralMovement.LOW_SPEED
        else m.HighSpeedLateralMovementStrategy())
    settings["LONGITUDINAL_MOVEMENT_STRATEGY"] = (
        m.MergingAndStoppingLongitudinalMovementStrategy()
        if longitudinal is LongitudinalMovement.MERGING_AND_STOPPING
        else m.VelocityKeepingLongitudinalMovementStrategy())
    settings["LATERAL_MOVEMENT"] = lateral
    settings["LONGITUDINAL_MOVEMENT"] = longitudinal
    return settings


def run_scenario(lateral, longitudinal, sim_loop):
    """Run main() for a scenario; return, for every planning cycle, the
    state it planned from, its best path and all the candidate paths."""
    cycles = []
    planning = m.frenet_optimal_planning

    def recording_planning(csp, s0, c_s_d, c_s_dd, c_d, c_d_d, c_d_dd, ob):
        best, paths = planning(csp, s0, c_s_d, c_s_dd, c_d, c_d_d, c_d_dd, ob)
        cycles.append(dict(state=(s0, c_s_d, c_s_dd, c_d, c_d_d, c_d_dd),
                           csp=csp, best=best, paths=paths))
        return best, paths

    with pytest.MonkeyPatch.context() as mp:
        for name, value in scenario_settings(lateral, longitudinal).items():
            mp.setattr(m, name, value)
        mp.setattr(m, "show_animation", False)
        mp.setattr(m, "SIM_LOOP", sim_loop)
        mp.setattr(m, "frenet_optimal_planning", recording_planning)
        m.main()
        # read the parameters while the scenario is still applied
        limits = dict(speed=m.MAX_SPEED, accel=m.MAX_ACCEL,
                      curvature=m.MAX_CURVATURE, radius=m.ROBOT_RADIUS,
                      obstacles=[(x, y, 0.0) for x, y in m.OBSTACLES])
    return cycles, limits


def check_scenario(lateral, longitudinal, sim_loop):
    cycles, limits = run_scenario(lateral, longitudinal, sim_loop)
    assert len(cycles) == sim_loop

    for i, cycle in enumerate(cycles):
        best, paths = cycle["best"], cycle["paths"]
        s0, s_d, s_dd, d0, d_d, d_dd = cycle["state"]
        assert best is not None, f"no feasible path in cycle {i}"
        # the cheapest of the feasible candidates
        assert any(best is p for p in paths["ok"])
        assert best.cf == min(p.cf for p in paths["ok"])

        # starts at the current state
        assert best.s[0] == pytest.approx(s0, abs=ROUND_OFF)
        assert best.s_d[0] == pytest.approx(s_d, abs=ROUND_OFF)
        assert best.s_dd[0] == pytest.approx(s_dd, abs=ROUND_OFF)
        assert best.d[0] == pytest.approx(d0, abs=ROUND_OFF)
        assert best.d_d[0] == pytest.approx(d_d, abs=LATERAL_DERIVATIVE_TOL)
        assert best.d_dd[0] == pytest.approx(d_dd, abs=LATERAL_DERIVATIVE_TOL)
        # d is measured to the left of the course, normal to its heading
        rx, ry = cycle["csp"].calc_position(best.s[0])
        ryaw = cycle["csp"].calc_yaw(best.s[0])
        assert best.x[0] == pytest.approx(rx - best.d[0] * math.sin(ryaw),
                                          abs=ROUND_OFF)
        assert best.y[0] == pytest.approx(ry + best.d[0] * math.cos(ryaw),
                                          abs=ROUND_OFF)

        # within the limits, and clear of the obstacles by the robot radius
        assert max(best.v) <= limits["speed"]
        assert max(abs(a) for a in best.a) <= limits["accel"]
        assert max(abs(c) for c in best.c) <= limits["curvature"]
        trajectory = list(zip(best.x, best.y))
        pc.assert_collision_free(trajectory, limits["obstacles"],
                                 limits["radius"],
                                 check_spacing=pc.max_step(trajectory))

    # each cycle starts from the second state of the previous best path
    for previous, cycle in zip(cycles[:-1], cycles[1:]):
        p = previous["best"]
        assert cycle["state"] == (p.s[1], p.s_d[1], p.s_dd[1],
                                  p.d[1], p.d_d[1], p.d_dd[1])
        assert cycle["state"][0] > previous["state"][0]


def test_default_scenario():
    check_scenario(m.LATERAL_MOVEMENT, m.LONGITUDINAL_MOVEMENT, sim_loop=5)


def test_high_speed_and_merging_and_stopping_scenario():
    check_scenario(LateralMovement.HIGH_SPEED,
                   LongitudinalMovement.MERGING_AND_STOPPING, sim_loop=3)


def test_low_speed_and_velocity_keeping_scenario():
    check_scenario(LateralMovement.LOW_SPEED,
                   LongitudinalMovement.VELOCITY_KEEPING, sim_loop=3)


def test_low_speed_and_merging_and_stopping_scenario():
    check_scenario(LateralMovement.LOW_SPEED,
                   LongitudinalMovement.MERGING_AND_STOPPING, sim_loop=3)


if __name__ == "__main__":
    conftest.run_this_test(__file__)
