import conftest
import math

import numpy as np

from Bipedal.bipedal_planner import bipedal_planner as m


def planned_footsteps(footsteps):
    """
    Chain the relative footsteps [dx, dy, theta]: footstep n is placed
    (dx, +-dy) from footstep n - 1 in the frame rotated by theta, with the
    lateral offset alternating between left and right foot.
    """
    x, y = 0.0, 0.0
    placed = [[x, y, 0.0]]
    for n, (dx, dy, theta) in enumerate(footsteps, start=1):
        lateral = dy if n % 2 else -dy
        x += math.cos(theta) * dx - math.sin(theta) * lateral
        y += math.sin(theta) * dx + math.cos(theta) * lateral
        placed.append([x, y, theta])
    return np.array(placed)


def test_1():
    bipedal_planner = m.BipedalPlanner()

    footsteps = [[0.0, 0.2, 0.0],
                 [0.3, 0.2, 0.0],
                 [0.3, 0.2, 0.2],
                 [0.3, 0.2, 0.2],
                 [0.0, 0.2, 0.2]]
    bipedal_planner.set_ref_footsteps(footsteps)
    bipedal_planner.walk(plot=False)

    reference = np.array(bipedal_planner.ref_p)
    actual = np.array(bipedal_planner.act_p)
    com = np.array(bipedal_planner.com_trajectory)

    # The reference footprints are the designated footsteps chained together.
    np.testing.assert_allclose(reference, planned_footsteps(footsteps),
                               atol=1e-12)

    # The modified footsteps (placed to stabilise the inverted pendulum)
    # stay close to the designated ones and keep their orientation.
    shift = np.linalg.norm(actual[:, :2] - reference[:, :2], axis=1)
    assert shift.max() < 0.1  # observed 0.050 m
    np.testing.assert_array_equal(actual[:, 2], reference[:, 2])

    # The COM is sampled 10 times per step and the walk ends with the feet
    # side by side (last step [0, 0.2]) and the COM between them.
    assert len(com) == 10 * len(footsteps)
    between_feet = (reference[-1, :2] + reference[-2, :2]) / 2
    assert np.linalg.norm(com[-1] - between_feet) < 0.025  # observed 0.012 m


if __name__ == '__main__':
    conftest.run_this_test(__file__)
