"""
Answer checks for the particle swarm optimisation demo.

main() seeds numpy, so every run gives the same swarm. The swarm minimises
"distance to the target + obstacle penalty". The target is more than r + 5
away from every obstacle, so the true minimum is the target itself, with
fitness 0. The best position must converge there, the best fitness must
never get worse from one iteration to the next, no best position may lie
inside an obstacle, and every particle must stay inside the search area.
"""
import numpy as np
import pytest

import conftest  # Add root path to sys.path
from PathPlanning.ParticleSwarmOptimization import particle_swarm_optimization as m

# set up of main()
TARGET = np.array([40.0, 35.0])
OBSTACLES = [(10, 15, 8), (-20, 0, 12), (20, -25, 10), (-5, -30, 7)]
SEARCH_BOUNDS = [(-50, 50), (-50, 50)]
SPAWN_AREA = [(-45, -35), (-45, -35)]
MAX_ITER = 150


@pytest.fixture(scope="module")
def swarm():
    m.show_animation = False
    return m.main()


def fitness(pos):
    """Distance to the target plus obstacle penalty (the module's rule)."""
    value = np.linalg.norm(pos - TARGET)
    for ox, oy, r in OBSTACLES:
        d = np.linalg.norm(pos - np.array([ox, oy]))
        if d < r:
            value += 1000  # inside
        elif d < r + 5:
            value += 50 / (d - r + 0.1)  # too close
    return value


def test_swarm_converges_on_the_target(swarm):
    assert fitness(TARGET) == 0.0  # so the target is the global minimum
    assert swarm.iteration == MAX_ITER
    assert np.linalg.norm(swarm.gbest_position - TARGET) < 1e-3  # measured 1e-10
    assert swarm.gbest_value == pytest.approx(fitness(swarm.gbest_position),
                                              abs=1e-12)


def test_best_fitness_never_gets_worse(swarm):
    values = [fitness(p) for p in swarm.gbest_path]
    assert len(values) == MAX_ITER
    assert all(b <= a for a, b in zip(values, values[1:]))
    assert values[-1] == pytest.approx(swarm.gbest_value, abs=1e-12)


def test_best_positions_are_never_inside_an_obstacle(swarm):
    for p in swarm.gbest_path:
        for ox, oy, r in OBSTACLES:
            assert np.hypot(p[0] - ox, p[1] - oy) >= r


def test_particles_stay_inside_the_search_area(swarm):
    for particle in swarm.particles:
        path = np.array(particle.path)
        assert len(path) == MAX_ITER + 1
        for axis, (low, high) in enumerate(SEARCH_BOUNDS):
            assert np.all((low <= path[:, axis]) & (path[:, axis] <= high))
        for axis, (low, high) in enumerate(SPAWN_AREA):  # where they start
            assert low <= path[0, axis] <= high


def test_demo_is_reproducible(swarm):
    again = m.main()
    assert np.array_equal(again.gbest_path, swarm.gbest_path)
    for p, q in zip(again.particles, swarm.particles):
        assert np.array_equal(p.path, q.path)


if __name__ == '__main__':
    conftest.run_this_test(__file__)
