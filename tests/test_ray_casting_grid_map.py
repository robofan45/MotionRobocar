import numpy as np

import conftest  # Add root path to sys.path
from Mapping.ray_casting_grid_map import ray_casting_grid_map as m

YAW_RESO = np.deg2rad(10.0)  # yaw resolution used by main()


def run_demo(monkeypatch):
    monkeypatch.setattr(m, "show_animation", False)
    return m.main()


def brute_force_map(ox, oy, minx, miny, shape, xyreso):
    """1.0 at obstacles, 0.5 behind an obstacle as seen from the origin
    (same yaw bin, farther away), 0.0 elsewhere."""
    x = np.arange(shape[0]) * xyreso + minx
    y = np.arange(shape[1]) * xyreso + miny
    x, y = np.meshgrid(x, y, indexing="ij")
    cell_bin = np.floor(np.mod(np.arctan2(y, x), 2 * np.pi) / YAW_RESO)
    cell_dist = np.hypot(x, y)
    expected = np.zeros(shape)
    for obx, oby in zip(ox, oy):
        obstacle_bin = np.floor(np.mod(np.arctan2(oby, obx), 2 * np.pi) / YAW_RESO)
        hidden = (cell_bin == obstacle_bin) & (cell_dist > np.hypot(obx, oby))
        expected[hidden] = 0.5
    ix = np.round((np.asarray(ox) - minx) / xyreso).astype(int)
    iy = np.round((np.asarray(oy) - miny) / xyreso).astype(int)
    expected[ix, iy] = 1.0
    return expected


def test_demo_map_matches_brute_force_ray_casting(monkeypatch):
    pmap, minx, maxx, miny, maxy, xyreso, ox, oy = run_demo(monkeypatch)
    pmap = np.array(pmap)
    expected = brute_force_map(ox, oy, minx, miny, pmap.shape, xyreso)
    np.testing.assert_array_equal(pmap, expected)
    # Cells behind the obstacles are marked as unknown (0.5).
    assert np.count_nonzero(pmap == 0.5) > 0
    assert np.count_nonzero(pmap == 1.0) == len(ox)


def test_two_runs_give_identical_results(monkeypatch):
    first = run_demo(monkeypatch)
    second = run_demo(monkeypatch)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    conftest.run_this_test(__file__)
