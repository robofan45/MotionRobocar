"""
Monte Carlo localization: starting from particles spread over the whole map,
the filter finds the robot and then tracks it closely. The lidar simulation
and the distance map are checked against geometry worked out by hand.
"""
import math

import numpy as np
import pytest

import conftest  # noqa: F401  (adds the project root to sys.path)
from Localization.monte_carlo_localization import \
    monte_carlo_localization as m


@pytest.fixture(scope="module")
def run():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m, "show_animation", False)
        return m.main()


def errors(truths, guesses):
    position = np.hypot(*(guesses[:, :2] - truths[:, :2]).T)
    heading = np.abs((guesses[:, 2] - truths[:, 2] + np.pi) % (2 * np.pi)
                     - np.pi)
    return position, np.degrees(heading)


def test_filter_finds_the_robot_and_keeps_tracking_it(run):
    truths, guesses, _ = run
    position, heading = errors(truths, guesses)
    assert position[0] > 3.0  # it really starts lost (observed 9.2 m)
    settled = slice(150, None)  # observed: settled for good at step 102
    assert position[settled].max() < 0.2  # observed 0.095 m
    assert heading[settled].max() < 5.0  # observed 2.2 deg


def test_particle_cloud_shrinks_onto_the_robot(run):
    _, _, spreads = run
    rng = np.random.default_rng(0)
    distances = m.distance_map(m.make_grid())
    start = m.random_particles(m.N_PARTICLES, distances, rng)
    assert np.sqrt(np.sum(np.var(start[:, :2], axis=0))) > 4.0  # whole map
    assert spreads[-1] < 0.1  # observed 0.037 m


def test_true_path_stays_clear_of_walls(run):
    truths, _, _ = run
    grid = m.make_grid()
    row, col, inside = m.cell_index(truths[:, :2], grid)
    assert inside.all()
    assert m.distance_map(grid)[row, col].min() > 0.5  # observed 0.8 m


def test_rays_stop_at_the_walls():
    grid = m.make_grid()
    pose = np.array([2.0, 2.5, 0.0])
    ranges = m.cast_rays(pose, grid)
    angles = m.beam_angles()
    expected = {0.0: 5.0,  # +x: inner wall face at x = 7.0
                -math.pi: 1.8,  # -x: outer wall face at x = 0.2
                -math.pi / 2: 2.3,  # -y: outer wall face at y = 0.2
                math.pi / 2: m.MAX_RANGE}  # +y: nothing within range
    for angle, distance in expected.items():
        beam = int(np.argmin(np.abs(angles - angle)))
        assert ranges[beam] == pytest.approx(distance, abs=m.RESOLUTION)


def test_distance_map_matches_brute_force():
    grid = m.make_grid()
    distances = m.distance_map(grid)
    walls = np.argwhere(grid)
    rng = np.random.default_rng(1)
    for row, col in rng.integers([0, 0], grid.shape, size=(20, 2)):
        brute = np.min(np.hypot(walls[:, 0] - row, walls[:, 1] - col))
        assert distances[row, col] == pytest.approx(brute * m.RESOLUTION)


def test_true_pose_explains_the_scan_best():
    grid = m.make_grid()
    distances = m.distance_map(grid)
    truth = np.array([10.0, 6.0, 0.3])
    ranges = m.cast_rays(truth, grid)
    nearby = truth + np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0],
                               [0.0, -0.3, 0.0], [0.0, 0.0, 0.2]])
    log_likelihood = m.log_likelihoods(nearby, ranges, distances)
    assert np.argmax(log_likelihood) == 0
    inside_wall = np.array([[7.1, 3.0, 0.0]])
    assert m.log_likelihoods(inside_wall, ranges, distances)[0] == \
        m.IMPOSSIBLE


def test_same_result_twice(run, monkeypatch):
    monkeypatch.setattr(m, "show_animation", False)
    for first, second in zip(run, m.main()):
        np.testing.assert_array_equal(first, second)


if __name__ == '__main__':
    conftest.run_this_test(__file__)
