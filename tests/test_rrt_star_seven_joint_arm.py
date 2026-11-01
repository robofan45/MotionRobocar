import conftest  # Add root path to sys.path
import math

import numpy as np
import pytest

from ArmNavigation.rrt_star_seven_joint_arm_control \
    import rrt_star_seven_joint_arm_control as m


def run_main(monkeypatch):
    monkeypatch.setattr(m, "show_animation", False)
    path, planner = m.main()
    assert path is not None, "no path found"
    return np.array(path, dtype=float), planner


def frame_origins(robot, joint_angles):
    """
    Origins of the base and of every link frame, from the standard
    Denavit-Hartenberg parameters [theta, alpha, a, d] of each link
    (theta replaced by the joint angle); independent of the module.
    """
    transform = np.eye(4)
    origins = [transform[:3, 3]]
    for link, theta in zip(robot.link_list, joint_angles):
        _, alpha, a, d = link.dh_params_
        ct, st = math.cos(theta), math.sin(theta)
        ca, sa = math.cos(alpha), math.sin(alpha)
        transform = transform @ np.array([[ct, -st * ca, st * sa, a * ct],
                                          [st, ct * ca, -ct * sa, a * st],
                                          [0.0, sa, ca, d],
                                          [0.0, 0.0, 0.0, 1.0]])
        origins.append(transform[:3, 3])
    return np.array(origins)


def densify(path, step=0.01):
    """Configurations along the path, at most `step` rad apart."""
    configs = [path[0]]
    for start, end in zip(path[:-1], path[1:]):
        n = max(1, math.ceil(np.linalg.norm(end - start) / step))
        configs += [start + (end - start) * i / n for i in range(1, n + 1)]
    return configs


def clearance(configs, planner, links):
    """
    Smallest distance between the arm and the obstacle surfaces over the
    configurations; the arm is either its frame origins (links=False) or
    the line segments between consecutive origins (links=True).
    """
    smallest = np.inf
    for q in configs:
        origins = frame_origins(planner.robot, q)
        start, segment = origins[:-1], np.diff(origins, axis=0)
        length2 = np.sum(segment ** 2, axis=1)
        for ox, oy, oz, radius in planner.obstacle_list:
            centre = np.array([ox, oy, oz])
            if links:
                t = np.divide(np.sum((centre - start) * segment, axis=1),
                              length2, out=np.zeros_like(length2),
                              where=length2 > 0)
                points = start + np.clip(t, 0.0, 1.0)[:, None] * segment
            else:
                points = origins
            distance = np.linalg.norm(points - centre, axis=1).min()
            smallest = min(smallest, distance - radius)
    return smallest


@pytest.fixture(scope="module")
def planned():
    """main() runs once for the tests that only inspect its result."""
    with pytest.MonkeyPatch.context() as mp:
        return run_main(mp)


def test1(planned):
    path, planner = planned
    start, goal = np.array(planner.start.x), np.array(planner.end.x)

    # The path connects the start and goal configurations (the order is
    # checked by test_path_runs_from_start_to_goal).
    ends = [path[0], path[-1]]
    assert any(np.allclose(e, start) for e in ends)
    assert any(np.allclose(e, goal) for e in ends)
    # every edge is at most one RRT* extension long
    assert np.linalg.norm(np.diff(path, axis=0), axis=1).max() \
        <= planner.expand_dis + 1e-9  # observed 0.2 rad
    # the joints stay outside every obstacle along the path, which is what
    # the planner checks (every 0.1 rad); here checked every 0.01 rad
    assert clearance(densify(path), planner, links=False) > 0.0  # observed 1.5 mm


def test_planning_is_reproducible(planned, monkeypatch):
    path, _ = run_main(monkeypatch)
    np.testing.assert_array_equal(path, planned[0])


def test_path_runs_from_start_to_goal(planned):
    path, planner = planned
    np.testing.assert_allclose(path[0], planner.start.x)
    np.testing.assert_allclose(path[-1], planner.end.x)


@pytest.mark.xfail(raises=AssertionError, strict=True, reason=(
    "check_collision() only tests the frame origins (joints) against the "
    "obstacles, so at path configurations it accepted a link between two "
    "joints passes 1.0 cm into an obstacle"))
def test_links_stay_clear_of_obstacles(planned):
    path, planner = planned
    # only the path nodes: configurations the planner itself checked
    assert clearance(path, planner, links=True) > 0.0


if __name__ == '__main__':
    conftest.run_this_test(__file__)
