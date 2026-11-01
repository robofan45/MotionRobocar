"""
MPPI control: the car finishes the course without touching an obstacle,
stays near the course away from the obstacles, and respects its limits.
"""
import numpy as np
import pytest

import conftest  # noqa: F401  (adds the project root to sys.path)
from PathTracking.mppi_control import mppi_control as m


@pytest.fixture(scope="module")
def run():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m, "show_animation", False)
        return m.main()


def test_car_finishes_the_course(run):
    states, controls, course, _ = run
    assert len(controls) * m.DT < 25.0  # observed 19.5 s
    assert np.linalg.norm(states[-1, :2] - course[-1]) < 1.0  # observed 0.13 m


def test_car_never_touches_an_obstacle(run):
    states, _, _, obstacles = run
    assert m.obstacle_gaps(states, obstacles).min() > 0.3  # observed 0.80 m


def test_car_stays_near_the_course_away_from_obstacles(run):
    states, _, course, obstacles = run
    distance = np.sqrt(m.squared_distance_to_course(states[:, :2], course))
    clear = m.obstacle_gaps(states, obstacles).min(axis=1) > 10.0
    assert clear.sum() > 50  # observed 77 of 196 steps
    assert distance[clear].max() < 2.0  # observed 1.40 m


def test_steering_acceleration_and_speed_stay_within_limits(run):
    states, controls, _, _ = run
    steer = states[:, 4]
    assert np.abs(steer).max() <= m.MAX_STEER
    assert np.abs(np.diff(steer)).max() <= m.MAX_STEER_RATE * m.DT + 1e-9
    assert np.abs(controls[:, 0]).max() <= m.MAX_ACCEL
    assert states[:, 3].max() < 1.1 * m.TARGET_SPEED  # observed 6.05 m/s


def test_squared_distance_to_course_matches_brute_force():
    rng = np.random.default_rng(0)
    points = rng.uniform(-5.0, 80.0, size=(7, 9, 2))
    course = m.make_course()
    brute = np.min(np.sum((points[..., None, :] - course) ** 2, axis=-1),
                   axis=-1)
    np.testing.assert_allclose(m.squared_distance_to_course(points, course),
                               brute, atol=1e-8)


def test_weights_favour_cheap_samples():
    costs = np.array([3.0, 1.0, 2.0, 10.0])
    weights = m.mppi_weights(costs, temperature=1.0)
    assert weights.sum() == pytest.approx(1.0)
    assert list(np.argsort(-weights)) == [1, 2, 0, 3]
    # only cost differences matter
    np.testing.assert_allclose(m.mppi_weights(costs + 1e6, 1.0), weights)
    # a cold temperature, or costs too large for exp(), keep the best only
    np.testing.assert_allclose(m.mppi_weights(costs, 1e-3), [0, 1, 0, 0],
                               atol=1e-12)
    np.testing.assert_allclose(m.mppi_weights(costs * 1e4, 1.0),
                               [0, 1, 0, 0], atol=1e-12)


def test_same_result_twice(run, monkeypatch):
    monkeypatch.setattr(m, "show_animation", False)
    for first, second in zip(run, m.main()):
        np.testing.assert_array_equal(first, second)


if __name__ == '__main__':
    conftest.run_this_test(__file__)
