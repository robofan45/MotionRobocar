import math

import numpy as np

import conftest  # Add root path to sys.path
from Localization.histogram_filter import histogram_filter as m


def cell_positions(grid_map):
    x = np.arange(grid_map.x_w) * grid_map.xy_resolution + grid_map.min_x
    y = np.arange(grid_map.y_w) * grid_map.xy_resolution + grid_map.min_y
    return np.meshgrid(x, y, indexing="ij")


def test_demo_belief_matches_range_measurements(monkeypatch):
    monkeypatch.setattr(m, "show_animation", False)
    monkeypatch.setattr(m, "SIM_TIME", 1.0)
    grid_map, x_true, rf_id = m.main()
    belief = np.asarray(grid_map.data)
    assert belief.min() >= 0.0
    assert abs(belief.sum() - 1.0) < 1e-9

    # In the first second only one beacon is in range, so the belief is a
    # ring around it whose radius is the robot's true range.
    x, y = cell_positions(grid_map)
    ranges = np.hypot(*(rf_id - x_true[:2, 0]).T)
    assert np.count_nonzero(ranges <= m.MAX_RANGE) == 1
    beacon = rf_id[np.argmin(ranges)]
    ring = np.abs(np.hypot(x - beacon[0], y - beacon[1]) - ranges.min()) <= 2.0
    assert belief[ring].sum() > 0.8  # seeded run: 0.90 on 11 % of the cells

    ix = int(round((x_true[0, 0] - grid_map.min_x) / grid_map.xy_resolution))
    iy = int(round((x_true[1, 0] - grid_map.min_y) / grid_map.xy_resolution))
    assert belief[ix, iy] > 5.0 / belief.size  # seeded run: 11.8x uniform


def test_observation_update_localizes_with_three_beacons():
    # Uniform prior and exact ranges to three beacons: the posterior is
    # Bayes' rule with Gaussian range likelihoods and peaks at the truth.
    grid_map = m.init_grid_map(0.5, -5.0, -5.0, 5.0, 5.0)
    true_xy = np.array([1.5, 2.0])
    beacons = np.array([[-6.0, -4.0], [7.0, -3.0], [1.0, 8.0]])
    z = np.array([[math.hypot(*(true_xy - b)), b[0], b[1]] for b in beacons])
    grid_map = m.observation_update(grid_map, z, m.RANGE_STD)
    belief = np.asarray(grid_map.data)

    x, y = cell_positions(grid_map)
    likelihood = np.ones_like(belief)
    for r, bx, by in z:
        error = np.hypot(x - bx, y - by) - r
        likelihood *= np.exp(-0.5 * (error / m.RANGE_STD) ** 2)
    np.testing.assert_allclose(belief, likelihood / likelihood.sum(), rtol=1e-9)
    peak = np.unravel_index(np.argmax(belief), belief.shape)
    assert (x[peak], y[peak]) == (1.5, 2.0)


def test_motion_update_shifts_belief_by_travelled_distance():
    grid_map = m.init_grid_map(m.XY_RESOLUTION, m.MIN_X, m.MIN_Y,
                               m.MAX_X, m.MAX_Y)
    belief = np.zeros((grid_map.x_w, grid_map.y_w))
    belief[30, 30] = 1.0
    grid_map.data = belief.tolist()
    # 5 m/s along +x for one 0.1 s step moves exactly one 0.5 m cell.
    grid_map = m.motion_update(grid_map, np.array([[5.0], [0.0]]), 0.0)
    moved = np.asarray(grid_map.data)
    assert np.unravel_index(np.argmax(moved), moved.shape) == (31, 30)
    assert abs(moved.sum() - 1.0) < 1e-9
    x, y = cell_positions(grid_map)
    np.testing.assert_allclose([(moved * x).sum(), (moved * y).sum()],
                               [x[31, 30], y[31, 30]], atol=1e-9)


if __name__ == '__main__':
    conftest.run_this_test(__file__)
