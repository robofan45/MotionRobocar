"""

Iterative Linear Quadratic Regulator (iLQR) trajectory optimization

author: Mahim Trivedi

iLQR finds the acceleration and steering sequence that takes a car from rest
to a goal pose, stopped there, with as little control effort as possible.
Each iteration linearises the car model around the current trajectory and
solves that linear-quadratic problem backwards in time, which gives every
step a correction: a feedforward change and a feedback gain on the state.
The corrected controls are rolled forward through the real (nonlinear) model,
with the step size halved until the cost goes down.

Reference:
- Li and Todorov, "Iterative Linear Quadratic Regulator Design for Nonlinear
  Biological Movement Systems", ICINCO 2004.
- Tassa, Erez and Todorov, "Synthesis and Stabilization of Complex Behaviors
  through Online Trajectory Optimization", IROS 2012.

"""
import math
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np

# vehicle
WB = 2.5  # [m] wheelbase
MAX_STEER = np.radians(35.0)  # [rad] kept by a steep penalty beyond it
CAR_LENGTH, CAR_WIDTH = 3.5, 1.6  # [m] body, centred on the wheelbase

DT = 0.1  # [s]
N_STEPS = 60  # trajectory length (6 s)

# cost: 0.5 u'Ru per step plus a steep penalty beyond the steering limit,
# and 0.5 e'Qe on the final error e = state - goal
R = np.diag([0.1, 1.0])  # (acceleration, steering)
W_STEER_LIMIT = 1e3
Q_FINAL = np.diag([100.0, 100.0, 1000.0, 100.0])  # (x, y, yaw, v)

START = np.array([0.0, 0.0, 0.0, 0.0])  # x, y, yaw, v (at rest)
GOAL = np.array([10.0, 7.0, np.pi / 2, 0.0])  # facing left, stopped

MAX_ITER = 100
TOLERANCE = 1e-6  # stop when the relative cost decrease is below this
MU_MIN, MU_MAX = 1e-6, 1e10  # regularisation range
STEP_SIZES = 0.5 ** np.arange(10)  # tried in turn by the forward pass

show_animation = True


def motion_model(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """One DT step of a kinematic bicycle model. State x = (x, y, yaw, v)
    at the rear axle, control u = (acceleration, steering angle)."""
    yaw, v = x[2], x[3]
    return np.array([x[0] + v * math.cos(yaw) * DT,
                     x[1] + v * math.sin(yaw) * DT,
                     yaw + v / WB * math.tan(u[1]) * DT,
                     v + u[0] * DT])


def jacobians(x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Derivatives of motion_model: fx = df/dx (4, 4) and fu = df/du (4, 2).
    """
    yaw, v, steer = x[2], x[3], u[1]
    fx = np.eye(4)
    fx[0, 2] = -v * math.sin(yaw) * DT
    fx[0, 3] = math.cos(yaw) * DT
    fx[1, 2] = v * math.cos(yaw) * DT
    fx[1, 3] = math.sin(yaw) * DT
    fx[2, 3] = math.tan(steer) / WB * DT
    fu = np.zeros((4, 2))
    fu[2, 1] = v / (WB * math.cos(steer) ** 2) * DT
    fu[3, 0] = DT
    return fx, fu


def running_cost(u: np.ndarray) -> float:
    excess = max(abs(u[1]) - MAX_STEER, 0.0)
    return float(0.5 * u @ R @ u + 0.5 * W_STEER_LIMIT * excess ** 2)


def running_cost_derivatives(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gradient lu (2,) and Hessian luu (2, 2) of running_cost."""
    lu = R @ u
    luu = R.copy()
    excess = abs(u[1]) - MAX_STEER
    if excess > 0.0:
        lu[1] += W_STEER_LIMIT * excess * math.copysign(1.0, u[1])
        luu[1, 1] += W_STEER_LIMIT
    return lu, luu


def final_cost(x: np.ndarray) -> float:
    error = x - GOAL
    return float(0.5 * error @ Q_FINAL @ error)


def total_cost(xs: np.ndarray, us: np.ndarray) -> float:
    return sum(running_cost(u) for u in us) + final_cost(xs[-1])


def rollout(start: np.ndarray, us: np.ndarray) -> np.ndarray:
    """States (T + 1, 4) visited by applying controls us (T, 2)."""
    xs = np.empty((len(us) + 1, 4))
    xs[0] = start
    for t, u in enumerate(us):
        xs[t + 1] = motion_model(xs[t], u)
    return xs


def backward_pass(xs: np.ndarray, us: np.ndarray, mu: float
                  ) -> tuple[np.ndarray, np.ndarray] | None:
    """Feedforward steps k (T, 2) and feedback gains K (T, 2, 4) from the
    linear-quadratic approximation around (xs, us), or None when a control
    Hessian is not positive definite (then mu must grow). mu regularises the
    value function's Hessian, as in Tassa et al."""
    k = np.zeros((len(us), 2))
    gains = np.zeros((len(us), 2, 4))
    vx = Q_FINAL @ (xs[-1] - GOAL)  # value function gradient
    vxx = Q_FINAL.copy()  # ... and Hessian
    for t in reversed(range(len(us))):
        fx, fu = jacobians(xs[t], us[t])
        lu, luu = running_cost_derivatives(us[t])
        qx = fx.T @ vx
        qu = lu + fu.T @ vx
        qxx = fx.T @ vxx @ fx
        vxx_reg = vxx + mu * np.eye(4)
        quu = luu + fu.T @ vxx_reg @ fu
        qux = fu.T @ vxx_reg @ fx
        if np.any(np.linalg.eigvalsh(quu) <= 0.0):
            return None
        k[t] = -np.linalg.solve(quu, qu)
        gains[t] = -np.linalg.solve(quu, qux)
        vx = qx + gains[t].T @ quu @ k[t] + gains[t].T @ qu + qux.T @ k[t]
        vxx = (qxx + gains[t].T @ quu @ gains[t] + gains[t].T @ qux
               + qux.T @ gains[t])
        vxx = 0.5 * (vxx + vxx.T)
    return k, gains


def forward_pass(xs: np.ndarray, us: np.ndarray, k: np.ndarray,
                 gains: np.ndarray, step: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Roll the corrected controls through the nonlinear model; the gains
    react to how far the new trajectory drifts from the old one."""
    new_xs = np.empty_like(xs)
    new_us = np.empty_like(us)
    new_xs[0] = xs[0]
    for t in range(len(us)):
        new_us[t] = us[t] + step * k[t] + gains[t] @ (new_xs[t] - xs[t])
        new_xs[t + 1] = motion_model(new_xs[t], new_us[t])
    return new_xs, new_us


def ilqr(start: np.ndarray, us: np.ndarray,
         on_iteration: Callable[[np.ndarray, list[float]], None] | None = None
         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimise the controls us (T, 2) from start. Returns the states
    (T + 1, 4), the controls, and the cost after each accepted iteration."""
    xs = rollout(start, us)
    costs = [total_cost(xs, us)]
    mu = 1.0
    for _ in range(MAX_ITER):
        result = backward_pass(xs, us, mu)
        if result is None:
            mu = min(mu * 10.0, MU_MAX)
            continue
        k, gains = result
        for step in STEP_SIZES:
            new_xs, new_us = forward_pass(xs, us, k, gains, step)
            new_cost = total_cost(new_xs, new_us)
            if new_cost < costs[-1]:
                break
        else:  # no step lowered the cost: lean towards gradient descent
            mu *= 10.0
            if mu > MU_MAX:
                break
            continue
        decrease = (costs[-1] - new_cost) / costs[-1]
        xs, us = new_xs, new_us
        costs.append(new_cost)
        mu = max(mu / 10.0, MU_MIN)
        if on_iteration is not None:
            on_iteration(xs, costs)
        if decrease < TOLERANCE:
            break
    return xs, us, np.array(costs)


def plot_pose(pose: np.ndarray, color: str) -> None:  # pragma: no cover
    plt.arrow(pose[0], pose[1], 1.5 * math.cos(pose[2]),
              1.5 * math.sin(pose[2]), head_width=0.5, color=color)


def plot_car(state: np.ndarray) -> None:  # pragma: no cover
    """Outline of the car body at its pose."""
    x, y, yaw = state[:3]
    rear, front = 0.5 * (WB - CAR_LENGTH), 0.5 * (WB + CAR_LENGTH)
    half = 0.5 * CAR_WIDTH
    body = np.array([[rear, -half], [front, -half], [front, half],
                     [rear, half], [rear, -half]])
    rot = np.array([[math.cos(yaw), -math.sin(yaw)],
                    [math.sin(yaw), math.cos(yaw)]])
    outline = body @ rot.T + [x, y]
    plt.fill(outline[:, 0], outline[:, 1], color="k", alpha=0.8)


def plot_background(title: str) -> None:  # pragma: no cover
    plt.cla()
    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if getattr(event, "key", None) == 'escape'
                       else None])
    plot_pose(START, "tab:blue")
    plot_pose(GOAL, "tab:green")
    plt.text(START[0], START[1] - 1.2, "start", ha="center")
    plt.text(GOAL[0] + 1.0, GOAL[1], "goal", va="center")
    plt.axis((-3.0, 15.0, -3.0, 10.0))
    plt.gca().set_aspect("equal")
    plt.grid(True)
    plt.title(title)


def plot_iteration(history: list[np.ndarray],
                   costs: list[float]) -> None:  # pragma: no cover
    plot_background(f"iLQR iteration {len(costs) - 1}: "
                    f"cost {costs[-1]:.2f}")
    for old in history[:-1]:
        plt.plot(old[:, 0], old[:, 1], "-", color="0.8", lw=1)
    plt.plot(history[-1][:, 0], history[-1][:, 1], "-r", lw=2,
             label="current trajectory")
    plt.legend(loc="upper left")
    plt.pause(0.001)


def plot_drive(xs: np.ndarray, us: np.ndarray) -> None:  # pragma: no cover
    for t in range(0, len(xs), 2):
        plot_background(f"iLQR result  t = {t * DT:3.1f} s   speed = "
                        f"{xs[t, 3] * 3.6:4.1f} km/h")
        plt.plot(xs[:, 0], xs[:, 1], "-r", lw=2, label="optimal trajectory")
        plt.plot(xs[:t + 1, 0], xs[:t + 1, 1], "-b", lw=3, label="driven")
        plot_car(xs[t])
        plt.legend(loc="upper left")
        plt.pause(0.001)


def main() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    print(__file__ + " start!!")
    history: list[np.ndarray] = []

    def record(xs: np.ndarray, costs: list[float]) -> None:
        history.append(xs.copy())
        if show_animation:  # pragma: no cover
            plot_iteration(history, costs)

    # start from a car that does nothing; iLQR finds speed and steering
    xs, us, costs = ilqr(START, np.zeros((N_STEPS, 2)), on_iteration=record)

    if show_animation:  # pragma: no cover
        plot_drive(xs, us)
        plt.show()

    return xs, us, costs


if __name__ == '__main__':
    main()
