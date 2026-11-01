import numpy as np
import pytest

import conftest  # Add root path to sys.path
from SLAM.FastSLAM2 import fast_slam2 as m


@pytest.fixture(scope="module")
def demo():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m, "show_animation", False)
        mp.setattr(m, "SIM_TIME", 3.0)
        return m.main()


def weighted_landmarks(particles):
    weights = np.array([p.w for p in particles])
    return sum(w * p.lm for w, p in zip(weights, particles)) / weights.sum()


def test_observed_landmarks_are_mapped_near_true_positions(demo):
    x_est, x_true, x_dr, particles, rfid = demo
    # Landmarks within sensor range of the true path are observed; the
    # others must never be initialized.
    dist = np.hypot(rfid[:, 0, None] - x_true[0], rfid[:, 1, None] - x_true[1])
    observed = np.min(dist, axis=1) <= m.MAX_RANGE
    assert observed.sum() == 5
    landmarks = weighted_landmarks(particles)
    errors = np.hypot(*(landmarks[observed] - rfid[observed]).T)
    assert errors.max() < 0.55  # seeded run: 0.32 m
    assert np.all(landmarks[~observed] == 0.0)


def test_pose_estimate_stays_near_truth(demo):
    x_est, x_true, x_dr, particles, rfid = demo
    rmse = np.sqrt(np.mean(np.sum((x_est[:2] - x_true[:2]) ** 2, axis=0)))
    assert rmse < 0.17  # seeded run: 0.063 m


def test_resampling_copies_the_selected_particles():
    n = m.N_PARTICLE
    particles = [m.Particle(1) for _ in range(n)]
    for i, p in enumerate(particles):
        p.x = float(i)
        p.lm[:] = i
        p.w = 2.0 / n if i < n // 2 else 0.0  # all weight on the first half
    np.random.seed(0)
    particles = m.resampling(particles)
    # Each particle of the first half has weight 2/n, so it is copied twice.
    assert sorted(p.x for p in particles) == sorted(2 * list(range(n // 2)))
    assert all(p.lm[0, 0] == p.x for p in particles)
    assert not any(np.shares_memory(a.lm, b.lm)
                   for i, a in enumerate(particles) for b in particles[i + 1:])


if __name__ == '__main__':
    conftest.run_this_test(__file__)
