import math

import conftest  # Add root path to sys.path
from Mapping.kmeans_clustering import kmeans_clustering as m


def run_demo(monkeypatch):
    monkeypatch.setattr(m, "show_animation", False)
    return m.main()


def test_clusters_match_generated_objects(monkeypatch):
    clusters, cx, cy = run_demo(monkeypatch)
    # calc_raw_data() generates the same number of points for each object,
    # object by object, so the true labels are [0, ..., 0, 1, ..., 1].
    n_points = clusters.n_data // 2
    true_labels = [0] * n_points + [1] * n_points
    assert clusters.labels in (true_labels, [1 - label for label in true_labels])

    for obj in range(2):
        label = clusters.labels[obj * n_points]
        error = math.hypot(clusters.center_x[label] - cx[obj],
                           clusters.center_y[label] - cy[obj])
        assert error < 0.5  # seeded run: 0.23 m (points spread +-1.5 m)


def test_two_runs_give_identical_results(monkeypatch):
    first, cx1, cy1 = run_demo(monkeypatch)
    second, cx2, cy2 = run_demo(monkeypatch)
    assert (first.x, first.y, first.labels) == (second.x, second.y, second.labels)
    assert (first.center_x, first.center_y) == (second.center_x, second.center_y)
    assert (cx1, cy1) == (cx2, cy2)


if __name__ == '__main__':
    conftest.run_this_test(__file__)
