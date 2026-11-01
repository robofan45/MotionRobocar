"""

Multi-object tracking with Kalman filters and the Hungarian algorithm

author: Mahim Trivedi

Several cars drive across a sensor's field of view. Each scan gives noisy
detections: most cars are seen, some are missed, and a few false detections
(clutter) appear at random places. The tracker keeps one constant-velocity
Kalman filter per track, and every scan it
1. predicts every track to the time of the scan,
2. scores each track-detection pair by the Mahalanobis distance between the
   detection and the track's predicted position, ruling out pairs that are
   too far apart (gating),
3. pairs tracks with detections so that as many as possible are paired at
   the lowest total score (the Hungarian algorithm), and updates them,
4. starts a tentative track from every unpaired detection, confirms a track
   after a few hits, and deletes one that keeps being missed.

Reference:
- Bar-Shalom, Willett and Tian, "Tracking and Data Fusion: A Handbook of
  Algorithms", YBS Publishing, 2011.
- Kuhn, "The Hungarian Method for the Assignment Problem", Naval Research
  Logistics Quarterly, 1955.

"""
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

DT = 0.1  # [s] time between scans
N_STEPS = 120  # 12 s
AREA = (-30.0, 30.0, -20.0, 20.0)  # field of view x_min, x_max, y_min, y_max

# true cars: start position (x, y) [m] and constant velocity (vx, vy) [m/s].
# Their paths cross, but at different times: no two cars come within 8 m.
CARS = [((-27.0, -4.0), (4.0, 0.0)),  # east along a road
        ((27.0, 4.0), (-4.0, 0.0)),  # west along it
        ((10.0, -18.0), (0.0, 3.0)),  # north across it
        ((-12.0, 18.0), (0.0, -3.0)),  # south across it
        ((-14.0, -17.0), (3.5, 1.5))]  # diagonally
DETECTION_PROBABILITY = 0.9
MEASUREMENT_STD = 0.3  # [m]
CLUTTER_RATE = 2.0  # false detections per scan, on average

# tracker
ACCEL_STD = 1.0  # [m/s^2] process noise of the constant-velocity model
GATE = 9.21  # squared Mahalanobis distance holding 99 % of true detections
FORBIDDEN_COST = 1e6  # pairing cost outside the gate: far above any real total
CONFIRM_HITS = 3  # a track is confirmed after this many detections
MAX_MISSES = 5  # a confirmed track is deleted after this many misses in a row
NEW_TRACK_SPEED_STD = 8.0  # [m/s] velocity uncertainty of a new track

F = np.array([[1.0, 0.0, DT, 0.0], [0.0, 1.0, 0.0, DT],
              [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
G = np.array([[0.5 * DT ** 2, 0.0], [0.0, 0.5 * DT ** 2], [DT, 0.0],
              [0.0, DT]])
Q = ACCEL_STD ** 2 * G @ G.T
H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
R = MEASUREMENT_STD ** 2 * np.eye(2)

show_animation = True


@dataclass
class Track:
    track_id: int
    mean: np.ndarray  # (x, y, vx, vy)
    cov: np.ndarray
    hits: int = 1
    misses: int = 0

    @property
    def confirmed(self) -> bool:
        return self.hits >= CONFIRM_HITS


def new_track(track_id: int, z: np.ndarray) -> Track:
    """A track at detection z whose velocity is still unknown."""
    cov = np.diag([MEASUREMENT_STD ** 2, MEASUREMENT_STD ** 2,
                   NEW_TRACK_SPEED_STD ** 2, NEW_TRACK_SPEED_STD ** 2])
    return Track(track_id, np.array([z[0], z[1], 0.0, 0.0]), cov)


def predict(track: Track) -> None:
    track.mean = F @ track.mean
    track.cov = F @ track.cov @ F.T + Q


def update(track: Track, z: np.ndarray) -> None:
    s = H @ track.cov @ H.T + R
    gain = track.cov @ H.T @ np.linalg.inv(s)
    track.mean = track.mean + gain @ (z - H @ track.mean)
    track.cov = (np.eye(4) - gain @ H) @ track.cov


def association_costs(tracks: list[Track], detections: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray]:
    """For each (track, detection): the squared Mahalanobis distance d2 of
    the detection from the track's predicted position, used for gating, and
    the pairing cost d2 + ln det(2 pi S), twice the negative log likelihood.
    Without the determinant term a new, uncertain track would look close to
    everything and steal detections from established tracks."""
    d2 = np.empty((len(tracks), len(detections)))
    cost = np.empty_like(d2)
    for i, track in enumerate(tracks):
        s = H @ track.cov @ H.T + R
        residual = detections - H @ track.mean
        d2[i] = np.einsum("nj,jk,nk->n", residual, np.linalg.inv(s),
                          residual)
        cost[i] = d2[i] + np.log(np.linalg.det(2.0 * np.pi * s))
    return d2, cost


def associate(d2: np.ndarray, cost: np.ndarray,
              gate: float) -> list[tuple[int, int]]:
    """(track, detection) pairs: as many pairs inside the gate as possible,
    then the lowest total cost (Hungarian algorithm)."""
    if d2.size == 0:
        return []
    rows, cols = linear_sum_assignment(np.where(d2 <= gate, cost,
                                                FORBIDDEN_COST))
    return [(int(r), int(c)) for r, c in zip(rows, cols) if d2[r, c] <= gate]


def tracker_step(tracks: list[Track], detections: np.ndarray,
                 next_id: int) -> tuple[list[Track], int]:
    """One scan: predict, associate, update, then start, confirm and delete
    tracks. Returns the tracks and the next free track id."""
    for track in tracks:
        predict(track)
    pairs = associate(*association_costs(tracks, detections), GATE)
    paired_tracks = {i for i, _ in pairs}
    paired_detections = {j for _, j in pairs}
    for i, j in pairs:
        update(tracks[i], detections[j])
        tracks[i].hits += 1
        tracks[i].misses = 0
    for i, track in enumerate(tracks):
        if i not in paired_tracks:
            track.misses += 1
    # a tentative track must not miss at all; a confirmed one gets slack
    tracks = [t for t in tracks
              if t.misses == 0 or (t.confirmed and t.misses <= MAX_MISSES)]
    for j, z in enumerate(detections):
        if j not in paired_detections:
            tracks.append(new_track(next_id, z))
            next_id += 1
    return tracks, next_id


def simulate(rng: np.random.Generator) -> tuple[np.ndarray, list[np.ndarray]]:
    """True car positions (T, cars, 2) and the detections of each scan:
    missed cars, measurement noise and clutter, in random order."""
    starts = np.array([start for start, _ in CARS])
    velocities = np.array([velocity for _, velocity in CARS])
    truth = np.array([starts + velocities * k * DT for k in range(N_STEPS)])
    scans = []
    for positions in truth:
        seen = positions[rng.random(len(positions)) < DETECTION_PROBABILITY]
        detected = seen + rng.normal(0.0, MEASUREMENT_STD, size=seen.shape)
        clutter = rng.uniform(AREA[::2], AREA[1::2],
                              size=(rng.poisson(CLUTTER_RATE), 2))
        scan = np.vstack([detected, clutter])
        scans.append(scan[rng.permutation(len(scan))])
    return truth, scans


def plot_frame(k: int, truth: np.ndarray, detections: np.ndarray,
               tracks: list[Track], trails: dict[int, list[np.ndarray]],
               colors: dict[int, str]) -> None:  # pragma: no cover
    plt.cla()
    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if getattr(event, "key", None) == 'escape'
                       else None])
    for car in range(truth.shape[1]):
        plt.plot(truth[:, car, 0], truth[:, car, 1], "-", color="0.88", lw=6)
    plt.plot(detections[:, 0], detections[:, 1], "x", color="0.4",
             label="detections (with clutter)")
    for track in tracks:
        x, y = track.mean[:2]
        if track.confirmed:
            color = colors[track.track_id]
            trail = np.array(trails[track.track_id][-25:])
            plt.plot(trail[:, 0], trail[:, 1], "-", color=color, lw=2)
            plt.plot(x, y, "o", color=color, ms=8)
            plt.text(x + 0.8, y + 0.8, f"ID {track.track_id}", color=color,
                     fontsize=9, weight="bold")
        else:
            plt.plot(x, y, "o", mfc="none", color="0.5", ms=6)
    plt.plot([], [], "o", mfc="none", color="0.5", label="tentative track")
    plt.axis(AREA)
    plt.gca().set_aspect("equal")
    plt.grid(True)
    # below the plot, where it cannot cover a track
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2,
               fontsize=8)
    n_confirmed = sum(t.confirmed for t in tracks)
    plt.title(f"Multi-object tracking  t = {k * DT:4.1f} s:  "
              f"{n_confirmed} confirmed tracks, {truth.shape[1]} cars")
    plt.pause(0.001)


def main() -> tuple[np.ndarray, list[np.ndarray],
                    list[list[tuple[int, float, float]]]]:
    print(__file__ + " start!!")
    rng = np.random.default_rng(4)
    truth, scans = simulate(rng)
    tracks: list[Track] = []
    next_id = 1
    history: list[list[tuple[int, float, float]]] = []
    trails: dict[int, list[np.ndarray]] = {}
    colors: dict[int, str] = {}  # the next colour for each newly confirmed track

    for k, detections in enumerate(scans):
        tracks, next_id = tracker_step(tracks, detections, next_id)
        history.append([(t.track_id, float(t.mean[0]), float(t.mean[1]))
                        for t in tracks if t.confirmed])
        for track in tracks:
            trails.setdefault(track.track_id, []).append(track.mean[:2])
            if track.confirmed and track.track_id not in colors:
                colors[track.track_id] = f"C{len(colors) % 10}"
        if show_animation:  # pragma: no cover
            plot_frame(k, truth[:k + 1], detections, tracks, trails, colors)

    if show_animation:  # pragma: no cover
        plt.show()

    return truth, scans, history


if __name__ == '__main__':
    main()
