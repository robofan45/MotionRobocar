import conftest  # Add root path to sys.path
import warnings

import numpy as np
from scipy.integrate import solve_ivp

from AerialNavigation.rocket_powered_landing import rocket_powered_landing as m


def simulate(model, X, U, sigma):
    """
    Integrate the nonlinear 6-DoF dynamics from the optimised initial state
    with the optimised inputs (linear between the K knots, as in the
    discretisation) and return the states at the knots.
    """
    n_intervals = U.shape[1] - 1

    def dynamics(tau, x):
        k = min(int(tau * n_intervals), n_intervals - 1)
        beta = tau * n_intervals - k
        u = U[:, k] + beta * (U[:, k + 1] - U[:, k])
        return sigma * model.f_func(x, u).flatten()

    solution = solve_ivp(dynamics, (0.0, 1.0), X[:, 0],
                         t_eval=np.linspace(0.0, 1.0, n_intervals + 1),
                         max_step=1.0 / n_intervals, rtol=1e-8, atol=1e-10)
    assert solution.success
    return solution.y


def test1(monkeypatch):
    monkeypatch.setattr(m, "show_animation", False)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="You are solving a parameterized problem that is not DPP",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Solution may be inaccurate",
            category=UserWarning,
        )
        X, U, sigma = m.main()

    # main() starts from its seeded random initial state; only the
    # initial orientation is left free to the optimiser.
    model = m.Rocket_Model_6DoF(np.random.default_rng(42))
    fixed = np.r_[0:7, 11:14]
    np.testing.assert_allclose(X[fixed, 0], model.x_init[fixed], atol=1e-8)

    # Fly the optimised inputs through the nonlinear dynamics: the rocket
    # lands on the pad at the commanded touchdown velocity (-0.1, 0, 0).
    X_sim = simulate(model, X, U, sigma)
    assert np.linalg.norm(X_sim[1:4, -1]) < 6e-5  # observed 2.8e-5
    assert np.linalg.norm(X_sim[4:7, -1] - model.v_I_final) < 6e-5  # observed 2.7e-5
    # ... and follows the optimised trajectory all the way down
    assert np.abs(X_sim - X).max() < 6e-5  # observed 2.7e-5

    # Physical limits along the trajectory (1e-6: solver tolerance).
    thrust = np.linalg.norm(U, axis=0)
    assert thrust.min() >= model.T_min - 1e-6
    assert thrust.max() <= model.T_max + 1e-6
    assert X[1].min() >= -1e-6  # never below the ground
    assert X[0, -1] >= model.m_dry  # fuel left
    assert 0.1 <= sigma < model.t_f_guess  # observed 3.79 (time units)


def test_seeded_initial_state_is_reproducible():
    first = m.Rocket_Model_6DoF(np.random.default_rng(42)).x_init
    second = m.Rocket_Model_6DoF(np.random.default_rng(42)).x_init
    np.testing.assert_array_equal(first, second)


def test_library_default_initial_state_is_random():
    first = m.Rocket_Model_6DoF(None).x_init
    second = m.Rocket_Model_6DoF(None).x_init
    assert not np.array_equal(first, second)


if __name__ == '__main__':
    conftest.run_this_test(__file__)
