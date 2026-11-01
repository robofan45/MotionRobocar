"""

Monte Carlo localization (MCL) with a lidar on a grid map

author: Mahim Trivedi

A robot drives through a known floor plan without knowing where it started.
Particles (guesses of its pose) begin spread over every free part of the map.
Each step every particle moves with the noisy odometry, then is weighted by
how well the lidar scan, placed at that particle's pose, lines up with the
walls of the map. Resampling multiplies the particles that explain the scans,
and the cloud collapses onto the true pose.

Scans are scored with the likelihood field model: a beam endpoint is likely
when it lands near a wall, and the distance to the nearest wall is read from
a distance map computed once, so no ray has to be cast per particle.

This is Augmented MCL: when the scans suddenly fit the particles worse than
they used to, a share of the particles is replaced by random ones, so a
cloud that settled on the wrong place can still find the right one.

Reference:
- Thrun, Burgard and Fox, "Probabilistic Robotics", MIT Press, 2005
  (Augmented MCL: section 8.3.5; likelihood field: section 6.4).
- Dellaert, Fox, Burgard and Thrun, "Monte Carlo Localization for Mobile
  Robots", ICRA 1999.

"""
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

# map: 20 m x 14 m, walls as rectangles (x_min, y_min, x_max, y_max) [m]
RESOLUTION = 0.1  # [m] grid cell size
MAP_WIDTH, MAP_HEIGHT = 20.0, 14.0  # [m]
WALLS = [
    (0.0, 0.0, 20.0, 0.2), (0.0, 13.8, 20.0, 14.0),  # outer walls
    (0.0, 0.0, 0.2, 14.0), (19.8, 0.0, 20.0, 14.0),
    (7.0, 0.0, 7.2, 5.0), (7.0, 7.0, 7.2, 14.0),  # inner wall with a door
    (12.0, 9.0, 14.5, 11.0),  # box
    (14.0, 3.0, 19.8, 3.2),  # short wall
    (3.0, 9.5, 4.0, 10.5),  # pillar
    (16.5, 6.5, 17.0, 7.0),  # small pillar
]

# lidar
N_BEAMS = 36
MAX_RANGE = 8.0  # [m]
RANGE_STD = 0.05  # [m]

# robot
DT = 0.1  # [s]
SPEED = 1.0  # [m/s]
ODOMETRY_STD = np.array([0.1, np.radians(5.0)])  # (speed, yaw rate) noise
START_POSE = np.array([2.0, 2.5, 0.0])  # x [m], y [m], yaw [rad]
WAYPOINTS = [(5.5, 2.5), (5.5, 6.0), (10.0, 6.0), (15.5, 8.0), (18.2, 8.0),
             (18.2, 5.0), (13.0, 5.0), (10.0, 1.8)]

# particle filter
N_PARTICLES = 2000
SIGMA_HIT = 0.2  # [m] how far from a wall an endpoint is still likely
Z_HIT, Z_RANDOM = 0.9, 0.1  # endpoint near a wall / anywhere (outliers)
# Beams of one scan are not independent, so the plain product of their
# likelihoods is far too sure of itself and wipes out good guesses early.
# Raising it to a power below one keeps several hypotheses alive.
LIKELIHOOD_POWER = 0.3
FAR_DISTANCE = 2.0  # [m] distance used for endpoints outside the map
IMPOSSIBLE = -1e3  # log likelihood of a pose inside a wall or off the map
# Augmented MCL: short- and long-term averages of the scan likelihood
ALPHA_FAST, ALPHA_SLOW = 0.2, 0.02

show_animation = True


def make_grid() -> np.ndarray:
    """Occupancy grid (rows: y, columns: x); True where there is a wall."""
    grid = np.zeros((round(MAP_HEIGHT / RESOLUTION),
                     round(MAP_WIDTH / RESOLUTION)), dtype=bool)
    for x0, y0, x1, y1 in WALLS:
        grid[round(y0 / RESOLUTION):round(y1 / RESOLUTION),
             round(x0 / RESOLUTION):round(x1 / RESOLUTION)] = True
    return grid


def distance_map(grid: np.ndarray) -> np.ndarray:
    """Distance [m] from every cell to the nearest wall cell."""
    return distance_transform_edt(~grid) * RESOLUTION


def cell_index(points: np.ndarray, grid: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row and column of points (..., 2), and whether they lie on the map
    (row and column are clipped to the map, so they are always usable)."""
    col = np.floor(points[..., 0] / RESOLUTION).astype(int)
    row = np.floor(points[..., 1] / RESOLUTION).astype(int)
    inside = ((row >= 0) & (row < grid.shape[0])
              & (col >= 0) & (col < grid.shape[1]))
    return (np.clip(row, 0, grid.shape[0] - 1),
            np.clip(col, 0, grid.shape[1] - 1), inside)


def beam_angles() -> np.ndarray:
    """Beam directions relative to the robot's heading."""
    return np.linspace(-math.pi, math.pi, N_BEAMS, endpoint=False)


def cast_rays(pose: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Noise-free ranges (N_BEAMS,) from pose to the first wall cell along
    each beam, marching in steps of half a cell; MAX_RANGE if none."""
    step = 0.5 * RESOLUTION
    distances = np.arange(1, round(MAX_RANGE / step) + 1) * step
    angles = pose[2] + beam_angles()
    points = (pose[:2] + distances[:, None, None]
              * np.stack([np.cos(angles), np.sin(angles)], axis=-1))
    row, col, inside = cell_index(points, grid)
    hit = grid[row, col] | ~inside  # (distances, beams)
    return np.where(hit.any(axis=0), distances[np.argmax(hit, axis=0)],
                    MAX_RANGE)


def motion_model(poses: np.ndarray, speed: np.ndarray | float,
                 yaw_rate: np.ndarray | float) -> np.ndarray:
    """Unicycle step for poses (..., 3)."""
    x, y, yaw = poses[..., 0], poses[..., 1], poses[..., 2]
    return np.stack([x + speed * np.cos(yaw) * DT,
                     y + speed * np.sin(yaw) * DT,
                     yaw + yaw_rate * DT], axis=-1)


def log_likelihoods(particles: np.ndarray, ranges: np.ndarray,
                    distances: np.ndarray) -> np.ndarray:
    """Log likelihood (N,) of the scan for each particle (N, 3), from the
    distance between each beam endpoint and the nearest wall."""
    valid = ranges < MAX_RANGE  # a max-range reading hit nothing: skip it
    angles = particles[:, 2:3] + beam_angles()[valid]
    endpoints = (particles[:, None, :2] + ranges[valid, None]
                 * np.stack([np.cos(angles), np.sin(angles)], axis=-1))
    row, col, inside = cell_index(endpoints, distances)
    wall_distance = np.where(inside, distances[row, col], FAR_DISTANCE)
    likelihood = (Z_HIT * np.exp(-0.5 * (wall_distance / SIGMA_HIT) ** 2)
                  + Z_RANDOM / MAX_RANGE)
    log_likelihood = LIKELIHOOD_POWER * np.sum(np.log(likelihood), axis=1)
    # the robot cannot stand inside a wall or off the map
    row, col, inside = cell_index(particles[:, :2], distances)
    in_free_space = inside & (distances[row, col] > 0.0)
    return np.where(in_free_space, log_likelihood, IMPOSSIBLE)


def systematic_resample(weights: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray:
    """Indices of the particles to keep: one random offset, then evenly
    spaced picks, so a particle with weight w gets about N * w copies."""
    positions = (rng.random() + np.arange(len(weights))) / len(weights)
    indices = np.searchsorted(np.cumsum(weights), positions)
    return np.minimum(indices, len(weights) - 1)


def estimate(particles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted mean pose; the heading is averaged as a unit vector."""
    x, y = weights @ particles[:, :2]
    yaw = math.atan2(weights @ np.sin(particles[:, 2]),
                     weights @ np.cos(particles[:, 2]))
    return np.array([x, y, yaw])


def random_particles(n: int, distances: np.ndarray,
                     rng: np.random.Generator) -> np.ndarray:
    """n poses spread uniformly over free space (at least 0.2 m from any
    wall), each with a random heading."""
    free_rows, free_cols = np.nonzero(distances > 0.2)
    pick = rng.integers(len(free_rows), size=n)
    xy = (np.column_stack([free_cols[pick], free_rows[pick]])
          + rng.random((n, 2))) * RESOLUTION
    return np.column_stack([xy, rng.uniform(-math.pi, math.pi, n)])


def drive(pose: np.ndarray, waypoint: tuple[float, float]
          ) -> tuple[float, float]:
    """Speed and yaw rate that steer the true robot to the waypoint."""
    heading = math.atan2(waypoint[1] - pose[1], waypoint[0] - pose[0])
    error = (heading - pose[2] + math.pi) % (2.0 * math.pi) - math.pi
    return (SPEED * max(math.cos(error), 0.2),
            float(np.clip(2.0 * error, -1.0, 1.0)))


def plot_frame(grid: np.ndarray, particles: np.ndarray, truth: np.ndarray,
               guess: np.ndarray, ranges: np.ndarray, trail: np.ndarray,
               step: int) -> None:  # pragma: no cover
    plt.cla()
    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if getattr(event, "key", None) == 'escape'
                       else None])
    plt.imshow(grid, origin="lower", cmap="Greys",
               extent=(0.0, MAP_WIDTH, 0.0, MAP_HEIGHT))
    plt.plot(particles[:, 0], particles[:, 1], ".", color="tab:red", ms=1.5,
             label="particles")
    angles = truth[2] + beam_angles()
    hits = ranges < MAX_RANGE
    plt.plot(truth[0] + ranges[hits] * np.cos(angles[hits]),
             truth[1] + ranges[hits] * np.sin(angles[hits]), ".",
             color="tab:cyan", ms=4, label="lidar hits")
    plt.plot(trail[:, 0], trail[:, 1], "-b", lw=1)
    for pose, color, label in [(truth, "tab:blue", "true pose"),
                               (guess, "tab:green", "estimate")]:
        plt.plot(pose[0], pose[1], "o", color=color, ms=7, label=label)
        plt.plot([pose[0], pose[0] + 0.8 * math.cos(pose[2])],
                 [pose[1], pose[1] + 0.8 * math.sin(pose[2])], "-",
                 color=color, lw=2)
    error = math.hypot(*(guess[:2] - truth[:2]))
    plt.title(f"Monte Carlo localization  step {step}:  "
              f"error {error:.2f} m")
    plt.legend(loc="upper right", fontsize=8, markerscale=2)
    plt.axis((0.0, MAP_WIDTH, 0.0, MAP_HEIGHT))
    plt.pause(0.001)


def main() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    print(__file__ + " start!!")
    rng = np.random.default_rng(5)
    grid = make_grid()
    distances = distance_map(grid)
    particles = random_particles(N_PARTICLES, distances, rng)
    w_fast = w_slow = 0.0

    truth = START_POSE.copy()
    truths: list[np.ndarray] = []
    guesses: list[np.ndarray] = []
    spreads: list[float] = []
    waypoints = list(WAYPOINTS)
    while waypoints:
        if math.hypot(waypoints[0][0] - truth[0],
                      waypoints[0][1] - truth[1]) < 0.4:
            waypoints.pop(0)
            continue
        speed, yaw_rate = drive(truth, waypoints[0])
        truth = motion_model(truth, speed, yaw_rate)
        odometry = np.array([speed, yaw_rate]) + rng.normal(0.0, ODOMETRY_STD)
        ranges = cast_rays(truth, grid)
        ranges = np.where(ranges < MAX_RANGE,
                          ranges + rng.normal(0.0, RANGE_STD, N_BEAMS),
                          MAX_RANGE)

        # predict: every particle moves with its own noisy copy of the odometry
        noise = rng.normal(0.0, ODOMETRY_STD, size=(N_PARTICLES, 2))
        particles = motion_model(particles, odometry[0] + noise[:, 0],
                                 odometry[1] + noise[:, 1])
        # update: weight by how well the scan fits the map at each particle
        log_weights = log_likelihoods(particles, ranges, distances)
        weights = np.exp(log_weights - log_weights.max())
        weights /= weights.sum()
        guess = estimate(particles, weights)

        # Augmented MCL: compare how well the scans fit now with how well
        # they fitted lately, and inject random particles when it got worse
        mean_likelihood = float(np.mean(np.exp(log_weights)))
        if not truths:
            w_fast = w_slow = mean_likelihood
        w_fast += ALPHA_FAST * (mean_likelihood - w_fast)
        w_slow += ALPHA_SLOW * (mean_likelihood - w_slow)
        particles = particles[systematic_resample(weights, rng)]
        n_random = rng.binomial(N_PARTICLES, max(0.0, 1.0 - w_fast / w_slow))
        if n_random:
            replace = rng.choice(N_PARTICLES, n_random, replace=False)
            particles[replace] = random_particles(n_random, distances, rng)

        truths.append(truth)
        guesses.append(guess)
        spreads.append(float(np.sqrt(np.sum(np.var(particles[:, :2],
                                                    axis=0)))))
        if show_animation:  # pragma: no cover
            plot_frame(grid, particles, truth, guess, ranges,
                       np.array(truths), len(truths))

    if show_animation:  # pragma: no cover
        plt.show()

    return np.array(truths), np.array(guesses), np.array(spreads)


if __name__ == '__main__':
    main()
