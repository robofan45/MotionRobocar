import math

import numpy as np
from scipy.special import erfc

import conftest  # Add root path to sys.path
from Mapping.gaussian_grid_map import gaussian_grid_map as m

XY_RESO = 0.5  # grid resolution used by main()
STD = 5.0  # standard deviation used by main()


def run_demo(monkeypatch):
    monkeypatch.setattr(m, "show_animation", False)
    return m.main()


def test_demo_map_is_gaussian_tail_of_obstacle_distance(monkeypatch):
    gmap, minx, maxx, miny, maxy, ox, oy = run_demo(monkeypatch)
    gmap = np.array(gmap)
    x = np.arange(gmap.shape[0]) * XY_RESO + minx
    y = np.arange(gmap.shape[1]) * XY_RESO + miny
    x, y = np.meshgrid(x, y, indexing="ij")
    # Brute force distance from every cell to its nearest obstacle.
    dist = np.min(np.hypot(x[..., None] - ox, y[..., None] - oy), axis=-1)
    expected = 0.5 * erfc(dist / (STD * math.sqrt(2.0)))  # 1 - Phi(d / std)
    np.testing.assert_allclose(gmap, expected, rtol=0.0, atol=1e-12)

    # The map covers every obstacle with a margin of about EXTEND_AREA / 2.
    margin = m.EXTEND_AREA / 2.0 - 0.5  # bounds are rounded to whole metres
    assert minx <= min(ox) - margin and max(ox) + margin <= maxx
    assert miny <= min(oy) - margin and max(oy) + margin <= maxy
    # Each obstacle's own cell is (almost) as likely as the 0.5 maximum.
    near = 0.5 * erfc(XY_RESO / math.sqrt(2.0) / (STD * math.sqrt(2.0)))
    for ix, iy in zip(np.round((ox - minx) / XY_RESO).astype(int),
                      np.round((oy - miny) / XY_RESO).astype(int)):
        assert near <= gmap[ix, iy] <= 0.5


def test_two_runs_give_identical_results(monkeypatch):
    first = run_demo(monkeypatch)
    second = run_demo(monkeypatch)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    conftest.run_this_test(__file__)
