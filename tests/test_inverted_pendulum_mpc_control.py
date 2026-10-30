import conftest
import numpy as np
import pytest

from InvertedPendulum import inverted_pendulum_mpc_control as m

X0 = np.array([[0.0], [0.0], [0.3], [0.0]])  # initial state of main()


def number_of_steps():
    """Steps taken by main()'s simulation loop."""
    n, time = 0, 0.0
    while m.sim_time > time:
        time += m.delta_t
        n += 1
    return n


def finite_horizon_gain(A, B, Q, R, horizon):
    """
    First-step gain of the finite-horizon LQR solved by mpc_control():
    minimise sum_t x[t+1]' Q x[t+1] + u[t]' R u[t], t = 0 .. horizon - 1,
    by the backward Riccati recursion.
    """
    P = np.zeros_like(Q)
    for _ in range(horizon):
        W = Q + P
        K = np.linalg.solve(R + B.T @ W @ B, B.T @ W @ A)
        P = A.T @ W @ A - A.T @ W @ B @ K
    return K


def test1(monkeypatch):
    monkeypatch.setattr(m, "show_animation", False)
    x = m.main()

    # The pendulum, released 0.3 rad from upright, is balanced upright and
    # at rest after 5 s (the cart position is not penalised, so it drifts).
    assert abs(x[2, 0]) < 2.5e-4  # angle, observed 1.1e-4 rad
    assert abs(x[3, 0]) < 4e-4  # angular velocity, observed 1.9e-4 rad/s
    assert abs(x[1, 0]) < 6e-4  # cart velocity, observed 2.8e-4 m/s

    # Without input or state constraints the MPC problem is a finite-horizon
    # LQR, so its first input is -K0 x and the closed loop is linear.
    # (1e-6: solver tolerance; observed 2.6e-10 and 2.5e-11)
    A, B = m.get_model_matrix()
    K0 = finite_horizon_gain(A, B, m.Q, m.R, m.T)
    _, _, _, _, u = m.mpc_control(X0)
    assert u[0] == pytest.approx((-K0 @ X0)[0, 0], abs=1e-6)
    x_expected = np.linalg.matrix_power(A - B @ K0, number_of_steps()) @ X0
    assert np.abs(x - x_expected).max() < 1e-6


if __name__ == '__main__':
    conftest.run_this_test(__file__)
