import math
import random

import numpy as np
import pytest

import conftest  # Add root path to sys.path
from Mapping.rectangle_fitting import rectangle_fitting as m


def outline_corners(vehicle):
    gx, gy = vehicle.calc_global_contour()
    # The outline has 20 interpolated points per side, starting at a corner.
    return np.array([[gx[i], gy[i]] for i in (0, 20, 40, 60)])


def test_demo_clusters_partition_points_and_rectangles_enclose_them(monkeypatch):
    monkeypatch.setattr(m, "show_animation", False)
    rects, id_sets, ox, oy = m.main()
    assert sorted(i for ids in id_sets for i in ids) == list(range(len(ox)))
    for rect, ids in zip(rects, id_sets):
        for i in ids:
            for k in (0, 1):  # the two side directions of the rectangle
                d = rect.a[k] * ox[i] + rect.b[k] * oy[i]
                assert rect.c[k] - 1e-9 <= d <= rect.c[k + 2] + 1e-9


@pytest.mark.parametrize("x, y, yaw_deg, width, length", [
    (-10.0, 0.0, 90.0, 3.0, 5.0),  # first vehicle of main() at its start
    (8.0, 6.0, 30.0, 3.0, 5.0),
    (12.0, -4.0, -20.0, 3.0, 5.0),
    (5.0, 12.0, 75.0, 4.0, 10.0),
])
def test_fit_matches_vehicle_outline(x, y, yaw_deg, width, length):
    random.seed(42)  # lidar range noise
    vehicle = m.VehicleSimulator(x, y, np.deg2rad(yaw_deg), 0.0, 50.0 / 3.6,
                                 width, length)
    ox, oy = m.LidarSimulator().get_observation_points(
        [vehicle], np.deg2rad(3.0))
    rects, id_sets = m.LShapeFitting().fitting(ox, oy)
    assert len(rects) == 1

    rect = rects[0]
    rect.calc_rect_contour()
    corners = outline_corners(vehicle)
    side = corners[1] - corners[2]  # a long side of the true outline
    angle = (math.atan2(rect.b[0], rect.a[0])
             - math.atan2(side[1], side[0])) % (math.pi / 2)
    assert math.degrees(min(angle, math.pi / 2 - angle)) < 6.0  # worst: 3 deg
    for cx, cy in zip(rect.rect_c_x[:4], rect.rect_c_y[:4]):
        nearest = np.min(np.hypot(corners[:, 0] - cx, corners[:, 1] - cy))
        assert nearest < 1.0  # worst seeded case: 0.50 m


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "VehicleSimulator.calc_global_contour() multiplies row vectors by "
    "rot_mat_2d(yaw), which rotates the outline by -yaw, so the simulated "
    "body does not point in the direction the vehicle drives"))
def test_vehicle_outline_points_along_heading():
    vehicle = m.VehicleSimulator(0.0, 0.0, np.deg2rad(30.0), 1.0, 10.0, 3.0, 5.0)
    corners = outline_corners(vehicle)
    forward = corners[1] - corners[2]  # rear corner -> front corner
    x0, y0 = vehicle.x, vehicle.y
    vehicle.update(1.0, 0.0, 0.0)
    assert math.atan2(forward[1], forward[0]) == pytest.approx(
        math.atan2(vehicle.y - y0, vehicle.x - x0), abs=1e-9)


if __name__ == '__main__':
    conftest.run_this_test(__file__)
