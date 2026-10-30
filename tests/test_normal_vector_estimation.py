import math

import numpy as np

import conftest  # Add root path to sys.path
from Mapping.normal_vector_estimation import normal_vector_estimation as m


def test_triangle_normal(monkeypatch):
    monkeypatch.setattr(m, "show_animation", False)
    center, normal_vector = m.main1()
    # Triangle (0, 0, 1), (1, 1, 0), (0, 1, 0): cross product is (0, 1, 1).
    np.testing.assert_allclose(center, [1 / 3, 2 / 3, 1 / 3], atol=1e-12)
    np.testing.assert_allclose(normal_vector, [0.0, 1.0, 1.0] / np.sqrt(2.0),
                               atol=1e-12)


def test_ransac_finds_plane_normal(monkeypatch):
    monkeypatch.setattr(m, "show_animation", False)
    result = m.main2()
    assert result is not None, "RANSAC did not find the plane"
    true_normal, center, estimated_normal = result
    assert abs(np.linalg.norm(estimated_normal) - 1.0) < 1e-12
    # A plane normal has no sign, so compare the line directions.
    cos_angle = min(1.0, abs(float(estimated_normal @ true_normal)))
    assert math.degrees(math.acos(cos_angle)) < 5.0  # seeded run: 2.6 deg

    # A second run must give exactly the same result.
    for a, b in zip(result, m.main2()):
        np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    conftest.run_this_test(__file__)
