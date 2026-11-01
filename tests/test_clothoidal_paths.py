import math

import numpy as np
import pytest

import conftest
import path_checks as pc
from PathPlanning.ClothoidPath import clothoid_path_planner as m


def run_demo():
    """Run main() without animation; return the generate_clothoid_paths()
    arguments and the paths it returned."""
    calls = []
    generate_clothoid_paths = m.generate_clothoid_paths

    def recording(start_point, start_yaw_list, goal_point, goal_yaw_list,
                  n_path_points):
        paths = generate_clothoid_paths(start_point, start_yaw_list,
                                        goal_point, goal_yaw_list,
                                        n_path_points)
        calls.append(((start_point, list(start_yaw_list), goal_point,
                       list(goal_yaw_list), n_path_points), paths))
        return paths

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m, "show_animation", False)
        mp.setattr(m, "generate_clothoid_paths", recording)
        m.main()
    assert len(calls) == 1
    return calls[0]


@pytest.fixture(scope="module")
def demo():
    return run_demo()


def test_1(demo):
    (_, start_yaws, _, goal_yaws, _), paths = demo
    # one path per pair of start and goal orientations
    assert len(paths) == len(start_yaws) * len(goal_yaws) == 75


@pytest.mark.parametrize("b, c", [(0.0, 0.3), (1.2, -0.7), (-2.5, 2.0)])
def test_fresnel_integrals_without_curvature_rate(b, c):
    # with a = 0 the integrals of cos / sin(b t + c) over [0, 1] have closed forms
    if b == 0.0:
        x, y = math.cos(c), math.sin(c)
    else:
        x = (math.sin(b + c) - math.sin(c)) / b
        y = (math.cos(c) - math.cos(b + c)) / b
    assert m.X(0.0, b, c) == pytest.approx(x, abs=1e-12)
    assert m.Y(0.0, b, c) == pytest.approx(y, abs=1e-12)


def test_demo_paths(demo):
    (start, start_yaws, goal, goal_yaws, n_points), paths = demo
    assert all(path is not None for path in paths)
    yaw_pairs = [(s, g) for s in start_yaws for g in goal_yaws]
    for (start_yaw, goal_yaw), path in zip(yaw_pairs, paths):
        p = np.array(path)
        assert p.shape == (n_points, 2)
        assert tuple(p[0]) == (start.x, start.y)
        # fsolve's residual; observed end errors are below 2e-13
        assert math.dist(p[-1], goal) <= 1e-9

        # tangents at the ends from second order one sided differences;
        # observed heading errors are below 8.3e-4 rad
        t0 = -3 * p[0] + 4 * p[1] - p[2]
        t1 = 3 * p[-1] - 4 * p[-2] + p[-3]
        assert abs(pc.angle_diff(math.atan2(t0[1], t0[0]), start_yaw)) < 2e-3
        assert abs(pc.angle_diff(math.atan2(t1[1], t1[0]), goal_yaw)) < 2e-3

        # points are evenly spaced in arc length (observed spread 4.9e-4)
        steps = pc.step_lengths(p)
        assert (steps.max() - steps.min()) / steps.mean() < 1e-3
        # a clothoid's curvature changes linearly with arc length; the
        # discrete estimate deviates from a straight line fit by 1.7e-4
        curvature = np.diff(pc.headings(p)) / steps[1:]
        arc_length = np.cumsum(steps)[:-1]
        fit = np.polyval(np.polyfit(arc_length, curvature, 1), arc_length)
        assert np.max(np.abs(fit - curvature)) < 4e-4


if __name__ == '__main__':
    conftest.run_this_test(__file__)
