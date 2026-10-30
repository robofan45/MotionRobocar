"""
Answer checks for the elastic bands planner.

The demo (ElasticBandsVisualizer, plotting off) loads the initial path and
the obstacles and tightens the band for MAX_ITER updates. The band is a chain
of bubbles: a bubble's radius is its clearance (distance from its cell to
the nearest obstacle cell, clipped to [MIN_BUBBLE_RADIUS, MAX_BUBBLE_RADIUS]),
checked here by brute force instead of the distance transform. When every
bubble is free and neighbouring bubbles overlap, the band between their
centres is collision free.
"""
import pathlib

import numpy as np
import pytest

import conftest
from PathPlanning.ElasticBands import elastic_bands as m
from PathPlanning.ElasticBands.elastic_bands import ElasticBands

DATA = pathlib.Path(m.__file__).parent  # path.npy and obstacles.npy


@pytest.fixture(scope="module")
def demo():
    """The visualizer after the demo's MAX_ITER band updates."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m, "ENABLE_PLOT", False)
        return m.ElasticBandsVisualizer()


def clearance(cell_xy, obstacles):
    """Brute force distance from a cell to the nearest obstacle cell."""
    occupied = np.argwhere(obstacles == 1)
    return np.sqrt(((occupied - cell_xy) ** 2).sum(axis=1)).min()


def band_of(elastic_band):
    centres = np.array([b.pos for b in elastic_band.bubbles], dtype=float)
    radii = np.array([b.radius for b in elastic_band.bubbles], dtype=float)
    return centres, radii


def test_1():
    path = np.load(DATA / "path.npy")
    obstacles_points = np.load(DATA / "obstacles.npy")
    obstacles = np.zeros((500, 500))
    for x, y in obstacles_points:
        size = 30  # Side length of the square
        half_size = size // 2
        x_start = max(0, x - half_size)
        x_end = min(obstacles.shape[0], x + half_size)
        y_start = max(0, y - half_size)
        y_end = min(obstacles.shape[1], y + half_size)
        obstacles[x_start:x_end, y_start:y_end] = 1
    elastic_bands = ElasticBands(path, obstacles)
    elastic_bands.update_bubbles()
    centres, _ = band_of(elastic_bands)
    assert np.array_equal(centres[0], path[0])  # the ends do not move
    assert np.array_equal(centres[-1], path[-1])
    # gaps are closed one inserted bubble per update, so the chain is only
    # checked after the full demo below
    assert len(centres) > len(path)


def test_band_keeps_its_end_points(demo):
    path = np.load(DATA / "path.npy")
    centres, _ = band_of(demo.elastic_band)
    assert np.array_equal(centres[0], path[0])
    assert np.array_equal(centres[-1], path[-1])
    assert np.all((0 <= centres) & (centres <= 499))  # on the map


def test_bubble_radius_is_the_clipped_clearance(demo):
    centres, radii = band_of(demo.elastic_band)
    for centre, radius in zip(centres, radii):
        free = clearance(centre.astype(int), demo.obstacles)
        assert radius == pytest.approx(
            np.clip(free, m.MIN_BUBBLE_RADIUS, m.MAX_BUBBLE_RADIUS))
        assert radius <= free  # the bubble holds no obstacle (min 19)


def test_band_is_collision_free(demo):
    centres, radii = band_of(demo.elastic_band)
    gaps = np.linalg.norm(np.diff(centres, axis=0), axis=1)
    # overlapping free bubbles cover the straight piece between them; the
    # module inserts bubbles to keep gaps below 0.7 of this (measured 0.68)
    assert np.all(gaps <= radii[:-1] + radii[1:])
    for a, b in zip(centres, centres[1:]):  # and sample the band densely
        for s in np.linspace(0.0, 1.0, 100):
            x, y = (a + (b - a) * s).astype(int)
            assert demo.obstacles[x, y] == 0


def test_band_got_shorter(demo):
    path = np.load(DATA / "path.npy")
    centres, _ = band_of(demo.elastic_band)
    before = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
    after = np.linalg.norm(np.diff(centres, axis=0), axis=1).sum()
    assert after < 0.9 * before  # measured 1445 -> 1231


if __name__ == "__main__":
    conftest.run_this_test(__file__)
