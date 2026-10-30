import math

import numpy as np
import pytest

import conftest  # Add root path to sys.path
from Mapping.circle_fitting import circle_fitting as m


def run_demo(monkeypatch):
    monkeypatch.setattr(m, "show_animation", False)
    return m.main()


def test_demo_fit_is_close_to_true_circle(monkeypatch):
    (cx, cy, cr), (ex, ey, er) = run_demo(monkeypatch)
    # Last step: 4 points of a 1 m circle seen 13 m away with 5 % range noise.
    assert math.hypot(ex - cx, ey - cy) < 0.9  # seeded run: 0.44 m
    assert abs(er - cr) < 0.4  # seeded run: 0.19 m


def test_two_runs_give_identical_results(monkeypatch):
    first_truth, first_fit = run_demo(monkeypatch)
    second_truth, second_fit = run_demo(monkeypatch)
    assert first_fit == second_fit
    assert first_truth == second_truth


@pytest.mark.parametrize("cx, cy, r, start, span", [
    (0.0, 0.0, 1.0, 0.0, 2.0 * math.pi),  # full circle
    (-2.0, -8.0, 1.0, 0.5, 1.5),  # arcs, as seen by the ray casting sensor
    (11.86, 5.86, 1.0, 3.5, 1.2),
    (3.0, 4.0, 10.0, 1.0, 0.6),
])
def test_fit_recovers_exact_circle(cx, cy, r, start, span):
    theta = np.linspace(start, start + span, 12)
    x = list(cx + r * np.cos(theta))
    y = list(cy + r * np.sin(theta))
    ex, ey, er, error = m.circle_fitting(x, y)
    # observed errors are below 1e-10 (roundoff)
    assert ex == pytest.approx(cx, abs=1e-8)
    assert ey == pytest.approx(cy, abs=1e-8)
    assert er == pytest.approx(r, abs=1e-8)
    assert error == pytest.approx(0.0, abs=1e-7)


if __name__ == '__main__':
    conftest.run_this_test(__file__)
