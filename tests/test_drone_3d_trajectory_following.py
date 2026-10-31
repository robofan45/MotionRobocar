import conftest
import numpy as np

from AerialNavigation.drone_3d_trajectory_following \
    import drone_3d_trajectory_following as m

# main() flies two laps of this 10 m square at 5 m altitude.
WAYPOINTS = np.array([[-5, -5, 5], [5, -5, 5], [5, 5, 5], [-5, 5, 5]], float)


def distance_to_closed_polyline(points, corners):
    """Shortest distance from every point to the closed polyline."""
    start = corners
    segment = np.roll(corners, -1, axis=0) - corners
    p = points[:, None, :]
    t = np.sum((p - start) * segment, axis=2) / np.sum(segment ** 2, axis=1)
    closest = start + np.clip(t, 0.0, 1.0)[..., None] * segment
    return np.linalg.norm(p - closest, axis=2).min(axis=1)


def test1(monkeypatch):
    monkeypatch.setattr(m, "show_animation", False)
    x, y, z = m.main()
    position = np.column_stack([x, y, z])

    # The drone stays on the square: horizontally close to its edges and
    # near the commanded altitude.
    horizontal = distance_to_closed_polyline(position[:, :2], WAYPOINTS[:, :2])
    assert horizontal.max() < 0.42  # observed 0.21 m
    assert np.abs(position[:, 2] - 5.0).max() < 0.52  # observed 0.26 m

    # It passes every corner ...
    for corner in WAYPOINTS:
        closest = np.linalg.norm(position - corner, axis=1).min()
        assert closest < 0.5  # observed at most 0.25 m

    # ... and ends the second lap back at the start corner.
    assert np.linalg.norm(position[-1] - WAYPOINTS[0]) < 0.33  # observed 0.165 m


def test_runs_with_the_animation_on(monkeypatch):
    """Running the script shows the animation; that path must work too
    (it crashed under NumPy 2.5, which no longer turns 1-element arrays
    into Python floats)."""
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "pause", lambda *args, **kwargs: None)
    monkeypatch.setattr(m, "show_animation", True)
    try:
        x, y, z = m.main()
    finally:
        plt.ioff()  # the quadrotor plot switches interactive mode on
        plt.close("all")
    assert len(x) == len(y) == len(z) > 0


if __name__ == '__main__':
    conftest.run_this_test(__file__)
