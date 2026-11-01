"""

Interacting Multiple Model (IMM) filter

author: Mahim Trivedi

A vehicle drives straight, turns left, drives straight, turns right and
drives straight again, and only its position is measured, with GPS-like
noise. No single motion model fits all of that, so the IMM filter runs one
Kalman filter per model (straight, left turn, right turn) side by side and
keeps a probability for each model. Every step it
1. mixes the filters' estimates, weighted by how likely a switch between
   models is,
2. lets each filter predict with its own model and update with the
   measurement,
3. raises the probability of the models that predicted the measurement
   well, and
4. reports the probability-weighted combination of the filters.

Reference:
- Blom and Bar-Shalom, "The Interacting Multiple Model Algorithm for Systems
  with Markovian Switching Coefficients", IEEE Transactions on Automatic
  Control, 1988.
- Bar-Shalom, Li and Kirubarajan, "Estimation with Applications to Tracking
  and Navigation", Wiley, 2001, section 11.6.

"""
import math

import matplotlib.pyplot as plt
import numpy as np

DT = 0.1  # [s]
SPEED = 10.0  # [m/s]
TURN_RATE = math.radians(20.0)  # [rad/s]
# true motion: (duration [s], turn rate [rad/s]) of each segment
SEGMENTS = [(8.0, 0.0), (4.5, TURN_RATE), (6.0, 0.0), (4.5, -TURN_RATE),
            (6.0, 0.0)]
MEASUREMENT_STD = 1.0  # [m] position noise

# IMM models, all with the state (x, y, vx, vy)
MODEL_NAMES = ["straight", "left turn", "right turn"]
MODEL_TURN_RATES = [0.0, TURN_RATE, -TURN_RATE]
ACCEL_STD = 0.2  # [m/s^2] process noise of each IMM model
SWITCH_PROBABILITY = 0.005  # per step, from one model to each other model
# a single constant-velocity Kalman filter, for comparison. It needs much
# more process noise to follow the turns; 5 m/s^2 gives its lowest error.
CV_ACCEL_STD = 5.0  # [m/s^2]

H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])  # measure x, y
R = MEASUREMENT_STD ** 2 * np.eye(2)
WARM_UP = 2.0  # [s] left out of error statistics while filters settle

show_animation = True


def transition_matrix(turn_rate: float) -> np.ndarray:
    """State transition over DT for (x, y, vx, vy): straight line for a
    zero turn rate, otherwise a coordinated turn, in which the velocity
    vector rotates by turn_rate * DT."""
    if turn_rate == 0.0:
        return np.array([[1.0, 0.0, DT, 0.0], [0.0, 1.0, 0.0, DT],
                         [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    s, c = math.sin(turn_rate * DT), math.cos(turn_rate * DT)
    return np.array([[1.0, 0.0, s / turn_rate, -(1.0 - c) / turn_rate],
                     [0.0, 1.0, (1.0 - c) / turn_rate, s / turn_rate],
                     [0.0, 0.0, c, -s],
                     [0.0, 0.0, s, c]])


def process_noise(accel_std: float) -> np.ndarray:
    """Covariance of a random acceleration held constant during DT."""
    g = np.array([[0.5 * DT ** 2, 0.0], [0.0, 0.5 * DT ** 2],
                  [DT, 0.0], [0.0, DT]])
    return accel_std ** 2 * g @ g.T


def switch_matrix(n_models: int, probability: float) -> np.ndarray:
    """switch[i, j]: probability that model j holds now if model i held
    one step ago."""
    switch = np.full((n_models, n_models), probability)
    np.fill_diagonal(switch, 1.0 - probability * (n_models - 1))
    return switch


def kalman_step(mean: np.ndarray, cov: np.ndarray, z: np.ndarray,
                f: np.ndarray, q: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray, float]:
    """Predict with model (f, q), update with the position measurement z.
    Also returns the likelihood of z under this model's prediction."""
    mean = f @ mean
    cov = f @ cov @ f.T + q
    innovation = z - H @ mean
    s = H @ cov @ H.T + R
    gain = cov @ H.T @ np.linalg.inv(s)
    mean = mean + gain @ innovation
    i_kh = np.eye(4) - gain @ H
    cov = i_kh @ cov @ i_kh.T + gain @ R @ gain.T  # Joseph form
    likelihood = math.exp(-0.5 * innovation @ np.linalg.solve(s, innovation)) \
        / (2.0 * math.pi * math.sqrt(np.linalg.det(s)))
    return mean, cov, likelihood


def mixing_probabilities(probs: np.ndarray,
                         switch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predicted model probabilities c (M,), and mix (M, M) with
    mix[i, j] = probability that model i held a step ago, given that
    model j holds now."""
    predicted = switch.T @ probs
    return predicted, switch * probs[:, None] / predicted


def imm_step(means: np.ndarray, covs: np.ndarray, probs: np.ndarray,
             z: np.ndarray, transitions: list[np.ndarray],
             noises: list[np.ndarray], switch: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                        np.ndarray]:
    """One IMM cycle. means (M, 4), covs (M, 4, 4) and probs (M,) are the
    model-conditioned estimates; returns them updated, followed by the
    combined mean and covariance."""
    # 1. mixing: each filter starts from a blend of all filters' estimates
    predicted, mix = mixing_probabilities(probs, switch)
    mixed_means = mix.T @ means
    mixed_covs = np.empty_like(covs)
    for j in range(len(probs)):
        spread = means - mixed_means[j]
        mixed_covs[j] = np.einsum("i,ijk->jk", mix[:, j], covs
                                  + spread[:, :, None] * spread[:, None, :])
    # 2. model-matched filtering
    new_means, new_covs = np.empty_like(means), np.empty_like(covs)
    likelihoods = np.empty(len(probs))
    for j, (f, q) in enumerate(zip(transitions, noises)):
        new_means[j], new_covs[j], likelihoods[j] = kalman_step(
            mixed_means[j], mixed_covs[j], z, f, q)
    # 3. model probabilities: the models that predicted z well gain
    probs = likelihoods * predicted
    probs /= probs.sum()
    # 4. combination
    mean = probs @ new_means
    spread = new_means - mean
    cov = np.einsum("i,ijk->jk", probs,
                    new_covs + spread[:, :, None] * spread[:, None, :])
    return new_means, new_covs, probs, mean, cov


def simulate(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray,
                                                 np.ndarray]:
    """True states (T, 4), the index of the true model at each step (T,),
    and noisy position measurements (T, 2)."""
    state = np.array([0.0, 0.0, SPEED, 0.0])
    states, modes = [], []
    for duration, turn_rate in SEGMENTS:
        f = transition_matrix(turn_rate)
        for _ in range(round(duration / DT)):
            state = f @ state
            states.append(state)
            modes.append(MODEL_TURN_RATES.index(turn_rate))
    truth = np.array(states)
    measurements = truth[:, :2] + rng.normal(0.0, MEASUREMENT_STD,
                                             size=(len(truth), 2))
    return truth, np.array(modes), measurements


def initial_estimate(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """From the first measurement alone: the velocity is unknown."""
    return (np.array([z[0], z[1], 0.0, 0.0]),
            np.diag([MEASUREMENT_STD ** 2, MEASUREMENT_STD ** 2,
                     15.0 ** 2, 15.0 ** 2]))


def plot_frame(ax_map: plt.Axes, ax_prob: plt.Axes, k: int,
               truth: np.ndarray, modes: np.ndarray, measurements: np.ndarray,
               imm: np.ndarray, cv: np.ndarray,
               probs: np.ndarray) -> None:  # pragma: no cover
    ax_map.cla()
    ax_map.plot(measurements[:k + 1, 0], measurements[:k + 1, 1], ".",
                color="0.7", ms=3, label="measurements")
    ax_map.plot(truth[:k + 1, 0], truth[:k + 1, 1], "-k", label="truth")
    ax_map.plot(cv[:k + 1, 0], cv[:k + 1, 1], "--", color="tab:blue",
                label="single CV filter")
    ax_map.plot(imm[:k + 1, 0], imm[:k + 1, 1], "-", color="tab:red", lw=2,
                label="IMM")
    ax_map.set_xlim(truth[:, 0].min() - 10.0, truth[:, 0].max() + 10.0)
    ax_map.set_ylim(truth[:, 1].min() - 10.0, truth[:, 1].max() + 10.0)
    ax_map.set_aspect("equal")
    ax_map.grid(True)
    ax_map.legend(loc="lower right", fontsize=8)
    ax_map.set_title("IMM filter: position only measurements")

    ax_prob.cla()
    time = np.arange(len(truth)) * DT
    for j, name in enumerate(MODEL_NAMES):
        ax_prob.plot(time[:k + 1], probs[:k + 1, j], label=name)
        # shade where this model is the true one
        ax_prob.fill_between(time, 0.0, 1.0, where=modes == j, alpha=0.08,
                             color=f"C{j}", step="mid")
    ax_prob.set_xlim(0.0, time[-1])
    ax_prob.set_ylim(0.0, 1.02)
    ax_prob.set_xlabel("time [s]")
    ax_prob.set_ylabel("model probability")
    ax_prob.grid(True)
    ax_prob.legend(loc="center right", fontsize=8)
    ax_prob.set_title("shaded: true motion")
    plt.pause(0.001)


def main() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                    np.ndarray, np.ndarray]:
    print(__file__ + " start!!")
    rng = np.random.default_rng(2)
    truth, modes, measurements = simulate(rng)

    transitions = [transition_matrix(w) for w in MODEL_TURN_RATES]
    noises = [process_noise(ACCEL_STD)] * len(MODEL_TURN_RATES)
    switch = switch_matrix(len(MODEL_TURN_RATES), SWITCH_PROBABILITY)
    mean, cov = initial_estimate(measurements[0])
    means = np.array([mean] * len(MODEL_TURN_RATES))
    covs = np.array([cov] * len(MODEL_TURN_RATES))
    probs = np.full(len(MODEL_TURN_RATES), 1.0 / len(MODEL_TURN_RATES))
    cv_mean, cv_cov = mean, cov
    cv_f, cv_q = transition_matrix(0.0), process_noise(CV_ACCEL_STD)

    imm_estimates, cv_estimates, model_probs = [mean], [cv_mean], [probs]
    if show_animation:  # pragma: no cover
        fig, (ax_map, ax_prob) = plt.subplots(
            1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [3, 2]})
        # for stopping simulation with the esc key.
        fig.canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if getattr(event, "key", None) == 'escape'
                           else None])

    for k, z in enumerate(measurements[1:], start=1):
        means, covs, probs, mean, cov = imm_step(means, covs, probs, z,
                                                 transitions, noises, switch)
        cv_mean, cv_cov, _ = kalman_step(cv_mean, cv_cov, z, cv_f, cv_q)
        imm_estimates.append(mean)
        cv_estimates.append(cv_mean)
        model_probs.append(probs)

        if show_animation and (k % 3 == 0 or k == len(truth) - 1):  # pragma: no cover
            plot_frame(ax_map, ax_prob, k, truth, modes, measurements,
                       np.array(imm_estimates), np.array(cv_estimates),
                       np.array(model_probs))

    if show_animation:  # pragma: no cover
        plt.show()

    return (truth, modes, measurements, np.array(imm_estimates),
            np.array(cv_estimates), np.array(model_probs))


if __name__ == '__main__':
    main()
