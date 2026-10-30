import math

import numpy as np
import pytest

import conftest  # Add root path to sys.path
from SLAM.ICPMatching import icp_matching as m

# main() moves the points by x = 0.5 m, y = 2 m, yaw = -10 deg;
# main_3d_points() by x = 0.5 m, y = 2 m, z = -5 m and -10 deg about y.
C, S = math.cos(np.deg2rad(-10.0)), math.sin(np.deg2rad(-10.0))
MOTION_2D = (np.array([[C, -S], [S, C]]), np.array([0.5, 2.0]))
MOTION_3D = (np.array([[C, 0.0, -S], [0.0, 1.0, 0.0], [S, 0.0, C]]),
             np.array([0.5, 2.0, -5.0]))


def inverse(motion):
    """ICP returns the transform taking current points back onto previous."""
    rot, trans = motion
    return rot.T, -rot.T @ trans


def run(demo):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m, "show_animation", False)
        return demo()


@pytest.fixture(scope="module")
def result_2d():
    return run(m.main)


@pytest.fixture(scope="module")
def result_3d():
    return run(m.main_3d_points)


def test_2d_rotation_is_recovered(result_2d):
    rot, _ = result_2d
    np.testing.assert_allclose(rot, inverse(MOTION_2D)[0], atol=1e-9)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "update_homogeneous_matrix() returns Hin @ H, composing the per-iteration "
    "transforms in the wrong order (should be H @ Hin), so the returned "
    "translation is wrong although the point sets align"))
def test_2d_transform_is_recovered(result_2d):
    rot, trans = result_2d
    expected_rot, expected_trans = inverse(MOTION_2D)
    np.testing.assert_allclose(rot, expected_rot, atol=1e-6)
    np.testing.assert_allclose(trans, expected_trans, atol=1e-6)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "update_homogeneous_matrix() returns Hin @ H, composing the per-iteration "
    "transforms in the wrong order (should be H @ Hin), so the returned "
    "rotation and translation are wrong although the point sets align"))
def test_3d_transform_is_recovered(result_3d):
    rot, trans = result_3d
    expected_rot, expected_trans = inverse(MOTION_3D)
    np.testing.assert_allclose(rot, expected_rot, atol=1e-6)
    np.testing.assert_allclose(trans, expected_trans, atol=1e-6)


def test_3d_two_runs_give_identical_results(result_3d):
    for a, b in zip(result_3d, run(m.main_3d_points)):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("dim", [2, 3])
def test_svd_motion_estimation_recovers_known_transform(dim):
    rng = np.random.default_rng(0)
    rot, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    if np.linalg.det(rot) < 0:  # make it a proper rotation (det = +1)
        rot[:, 0] *= -1.0
    trans = rng.normal(size=dim)
    previous = rng.uniform(-10.0, 10.0, size=(dim, 50))
    current = rot @ previous + trans[:, None]
    est_rot, est_trans = m.svd_motion_estimation(previous, current)
    np.testing.assert_allclose(est_rot, rot.T, atol=1e-10)
    np.testing.assert_allclose(est_trans, -rot.T @ trans, atol=1e-10)


def test_nearest_neighbor_association_matches_brute_force():
    rng = np.random.default_rng(1)
    previous = rng.uniform(-5.0, 5.0, size=(2, 30))
    current = rng.uniform(-5.0, 5.0, size=(2, 30))
    indexes, error = m.nearest_neighbor_association(previous, current)
    expected = [int(np.argmin(np.hypot(*(previous - c[:, None]))))
                for c in current.T]
    np.testing.assert_array_equal(indexes, expected)
    assert error == pytest.approx(np.sum(np.hypot(*(previous - current))))


if __name__ == '__main__':
    conftest.run_this_test(__file__)
