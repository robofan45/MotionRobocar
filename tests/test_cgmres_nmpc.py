import conftest
import numpy as np

from PathTracking.cgmres_nmpc import cgmres_nmpc as m


def test1(monkeypatch):
    monkeypatch.setattr(m, "show_animation", False)
    plant, controller = m.main()

    # The NMPC parks the car, which starts at (-4.5 m, -2.5 m, 45 deg) while
    # rolling backwards at 1 m/s, at the origin (its terminal cost).
    assert np.hypot(plant.x, plant.y) < 0.1  # observed 0.050 m
    assert abs(plant.yaw) < 0.16  # observed 0.080 rad
    assert abs(plant.v) < 0.02  # observed 0.008 m/s

    # ... and stays there: the last third of the run is spent near the goal.
    distance = np.hypot(plant.history_x[-500:], plant.history_y[-500:])
    assert distance.max() < 0.22  # observed 0.110 m

    # After a 10-update warm-up the inputs respect their limits. The limits
    # are equality constraints u**2 + dummy**2 = u_max**2 solved with a
    # residual of up to 0.02, hence the 1 % allowance.
    u_accel = np.abs(controller.history_u_1[10:])
    u_steer = np.abs(controller.history_u_2[10:])
    assert u_accel.max() <= 1.01 * m.U_A_MAX  # observed 0.990
    assert u_steer.max() <= 1.01 * m.U_OMEGA_MAX  # observed 0.777 (max 0.785)

    # The optimality residual of the continuation method has settled.
    assert controller.history_f[-1] < 0.15  # observed 0.073


if __name__ == '__main__':
    conftest.run_this_test(__file__)
