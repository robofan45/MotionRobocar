import conftest
import numpy as np
import scipy.linalg

from InvertedPendulum import inverted_pendulum_lqr_control as m

X0 = np.array([[0.0], [0.0], [0.3], [0.0]])  # initial state of main()


def number_of_steps():
    """Steps taken by main()'s simulation loop."""
    n, time = 0, 0.0
    while m.sim_time > time:
        time += m.delta_t
        n += 1
    return n


def test_1(monkeypatch):
    monkeypatch.setattr(m, "show_animation", False)
    x = m.main()

    # The pendulum, released 0.3 rad from upright, is balanced upright and
    # at rest after 5 s (the cart position is not penalised, so it drifts).
    assert abs(x[2, 0]) < 2.5e-4  # angle, observed 1.1e-4 rad
    assert abs(x[3, 0]) < 4e-4  # angular velocity, observed 1.8e-4 rad/s
    assert abs(x[1, 0]) < 5e-4  # cart velocity, observed 2.5e-4 m/s

    # The gain from the iterative DARE solver matches scipy's exact solution.
    A, B = m.get_model_matrix()
    K, _, _ = m.dlqr(A, B, m.Q, m.R)
    P = scipy.linalg.solve_discrete_are(A, B, m.Q, m.R)
    K_exact = np.linalg.solve(m.R + B.T @ P @ B, B.T @ P @ A)
    assert np.abs(K - K_exact).max() < 4e-3  # observed 1.9e-3 (K up to 71)

    # The final state is the linear closed loop x_n = (A - B K)^n x_0.
    x_expected = np.linalg.matrix_power(A - B @ K_exact,
                                        number_of_steps()) @ X0
    assert np.abs(x - x_expected).max() < 1.5e-4  # observed 6.5e-5


if __name__ == '__main__':
    conftest.run_this_test(__file__)
