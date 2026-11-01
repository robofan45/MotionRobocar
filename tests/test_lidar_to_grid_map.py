"""
Lidar to grid map: the demo must find its scan file wherever it is started
from, not only from its own folder.
"""
import matplotlib.pyplot as plt

import conftest  # noqa: F401  (adds the project root to sys.path)
from Mapping.lidar_to_grid_map import lidar_to_grid_map as m


def test_demo_runs_from_any_working_directory(tmp_path, monkeypatch):
    """The scan file used to be opened relative to the current directory,
    so starting the demo from the project root failed."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    try:
        m.main()
    finally:
        plt.close("all")


if __name__ == '__main__':
    conftest.run_this_test(__file__)
