import numpy as np
import pytest

import conftest  # Add root path to sys.path
from Mapping.ndt_map import ndt_map as m


@pytest.mark.xfail(strict=True, raises=AttributeError, reason=(
    "NDTMap.__init__ calls _create_grid_index_map(), which uses "
    "self.grid_map, before _construct_grid_map() creates self.grid_map, so "
    "building a map (and main()) fails"))
def test_cells_hold_mean_and_covariance_of_their_points():
    np.random.seed(42)
    ox, oy = m.create_dummy_observation_data()
    resolution = 10.0
    ndt = m.NDTMap(ox, oy, resolution)

    # Every point is in exactly one cell, and inside that cell's bounds.
    assert sorted(i for ids in ndt.grid_index_map.values() for i in ids) == \
        list(range(len(ox)))
    n_cells = 0
    for grid_index, ids in ndt.grid_index_map.items():
        cx, cy = ndt.grid_map.calc_grid_central_xy_position_from_grid_index(
            grid_index)
        assert np.all(np.abs(ox[ids] - cx) <= resolution / 2 + 1e-9)
        assert np.all(np.abs(oy[ids] - cy) <= resolution / 2 + 1e-9)
        if len(ids) >= ndt.min_n_points:
            n_cells += 1
            cell = ndt.grid_map.data[grid_index]
            assert cell.n_points == len(ids)
            np.testing.assert_allclose([cell.mean_x, cell.mean_y],
                                       [ox[ids].mean(), oy[ids].mean()])
            np.testing.assert_allclose(cell.covariance, np.cov(ox[ids], oy[ids]))
    assert n_cells > 0


if __name__ == '__main__':
    conftest.run_this_test(__file__)
