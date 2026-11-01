"""
Multi-object tracking: after a short warm-up there is exactly one confirmed
track per car, each car keeps one track id despite missed detections and
clutter, and the pairing step agrees with a brute-force search.
"""
import itertools

import numpy as np
import pytest

import conftest  # noqa: F401  (adds the project root to sys.path)
from Mapping.multi_object_tracking import multi_object_tracking as m

WARM_UP = 5  # scans: a track needs CONFIRM_HITS detections to be confirmed


@pytest.fixture(scope="module")
def run():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m, "show_animation", False)
        return m.main()


def nearest_tracks(cars, confirmed):
    """(track id, distance) of the confirmed track nearest to each car."""
    ids = [track_id for track_id, _, _ in confirmed]
    xy = np.array([(x, y) for _, x, y in confirmed]).reshape(-1, 2)
    result = []
    for car in cars:
        distance = np.linalg.norm(xy - car, axis=1)
        j = int(np.argmin(distance))
        result.append((ids[j], float(distance[j])))
    return result


def test_exactly_one_confirmed_track_per_car(run):
    truth, _, history = run
    exact = 0
    for k in range(WARM_UP, len(history)):
        if len(history[k]) == len(m.CARS):
            nearest = nearest_tracks(truth[k], history[k])
            exact += (all(d < 1.5 for _, d in nearest)
                      and len({i for i, _ in nearest}) == len(m.CARS))
    assert exact >= len(history) - WARM_UP - 5  # observed: all 115 scans


def test_each_car_keeps_one_track_id(run):
    truth, _, history = run
    ids_per_car = [set() for _ in m.CARS]
    for k in range(WARM_UP, len(history)):
        for car, (track_id, _) in enumerate(nearest_tracks(truth[k],
                                                           history[k])):
            ids_per_car[car].add(track_id)
    assert all(len(ids) == 1 for ids in ids_per_car)


def test_tracks_are_more_accurate_than_the_detections(run):
    truth, _, history = run
    errors = [d for k in range(WARM_UP, len(history))
              for _, d in nearest_tracks(truth[k], history[k])]
    rmse = np.sqrt(np.mean(np.square(errors)))
    assert rmse < 0.8 * m.MEASUREMENT_STD  # observed 0.199 m vs 0.3 m


def test_established_track_keeps_its_detection_from_a_new_track():
    """A brand-new track is uncertain, so by Mahalanobis distance alone it
    looks closer to everything. The log-determinant in the pairing cost
    must stop it from stealing an established track's detection."""
    established = m.Track(1, np.array([0.0, 0.0, 4.0, 0.0]),
                          np.diag([0.05, 0.05, 0.1, 0.1]), hits=10)
    newcomer = m.new_track(2, np.array([0.3, 0.0]))
    for track in (established, newcomer):
        m.predict(track)
    detections = np.array([[0.45, 0.05]])
    d2, cost = m.association_costs([established, newcomer], detections)
    assert d2[1, 0] < d2[0, 0]  # distance alone would pick the newcomer
    assert m.associate(d2, cost, m.GATE) == [(0, 0)]


def best_pairing(d2, cost, gate):
    """Brute force: (number of pairs, total cost) of the best pairing,
    which pairs as many gated (row, column) pairs as possible, then the
    cheapest. Every row picks a distinct column or none."""
    rows, cols = d2.shape
    best = (0, 0.0)
    for choice in itertools.product([None, *range(cols)], repeat=rows):
        picked = [c for c in choice if c is not None]
        if len(picked) != len(set(picked)):
            continue
        pairs = [(r, c) for r, c in enumerate(choice) if c is not None]
        if any(d2[r, c] > gate for r, c in pairs):
            continue
        key = (len(pairs), sum(cost[r, c] for r, c in pairs))
        if key[0] > best[0] or (key[0] == best[0] and key[1] < best[1]):
            best = key
    return best


def test_pairing_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(200):
        shape = tuple(rng.integers(1, 5, size=2))
        d2 = rng.uniform(0.0, 2.0 * m.GATE, size=shape)
        cost = d2 + rng.normal(size=shape)
        pairs = m.associate(d2, cost, m.GATE)
        assert len({r for r, _ in pairs}) == len({c for _, c in pairs}) \
            == len(pairs)
        count, total = best_pairing(d2, cost, m.GATE)
        assert len(pairs) == count
        assert sum(cost[r, c] for r, c in pairs) == pytest.approx(total)


def test_same_result_twice(run, monkeypatch):
    monkeypatch.setattr(m, "show_animation", False)
    truth, scans, history = m.main()
    np.testing.assert_array_equal(truth, run[0])
    for first, second in zip(scans, run[1]):
        np.testing.assert_array_equal(first, second)
    assert history == run[2]


if __name__ == '__main__':
    conftest.run_this_test(__file__)
