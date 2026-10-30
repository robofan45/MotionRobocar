import numpy as np
import pytest

import conftest  # Add root path to sys.path
from SLAM.GraphBasedSLAM import graph_based_slam as m

N_POSES = 10  # main() optimizes the first 10 poses at t = 20 s


def run_demo():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m, "show_animation", False)
        mp.setattr(m, "SIM_TIME", 20.0)
        return m.main()


@pytest.fixture(scope="module")
def demo():
    hx_true, hx_dr, hz = run_demo()
    # Repeat the (deterministic) optimization that main() ran at t = 20 s.
    x_opt = m.graph_based_slam(hx_dr[:, :N_POSES], hz[:N_POSES])
    return (hx_true, hx_dr, hz), x_opt


def aligned_rmse(estimate, truth):
    """Position RMSE after the best rigid 2D alignment: the shape error."""
    e = estimate - estimate.mean(axis=1, keepdims=True)
    t = truth - truth.mean(axis=1, keepdims=True)
    u, _, vt = np.linalg.svd(t @ e.T)
    rot = u @ np.diag([1.0, np.linalg.det(u @ vt)]) @ vt
    return np.sqrt(np.mean(np.sum((rot @ e - t) ** 2, axis=0)))


def test_first_pose_position_stays_at_origin(demo):
    _, x_opt = demo
    np.testing.assert_allclose(x_opt[:2, 0], 0.0, atol=1e-9)


def test_optimized_trajectory_shape_beats_dead_reckoning(demo):
    (hx_true, hx_dr, _), x_opt = demo
    truth = hx_true[:2, :N_POSES]
    opt_error = aligned_rmse(x_opt[:2], truth)
    assert opt_error < 0.04  # seeded run: 0.018 m
    assert opt_error < aligned_rmse(hx_dr[:2, :N_POSES], truth) / 4  # DR: 0.15 m


def test_two_runs_give_identical_results(demo):
    (hx_true, hx_dr, hz), _ = demo
    hx_true2, hx_dr2, hz2 = run_demo()
    np.testing.assert_array_equal(hx_true, hx_true2)
    np.testing.assert_array_equal(hx_dr, hx_dr2)
    assert len(hz) == len(hz2)
    for a, b in zip(hz, hz2):
        np.testing.assert_array_equal(a, b)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "'H[0:3, 0:3] += I' only damps each Gauss-Newton step of the first "
    "pose instead of fixing it: its heading drifts (to -0.07 rad here), so "
    "the whole optimized trajectory is rotated about the origin"))
def test_first_pose_heading_stays_fixed(demo):
    _, x_opt = demo
    assert abs(x_opt[2, 0]) < 1e-6


if __name__ == '__main__':
    conftest.run_this_test(__file__)
