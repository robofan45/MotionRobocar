"""Path hack to make tests work."""
import sys
import os
import pytest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(TEST_DIR)  # to import this file from test code.
ROOT_DIR = os.path.dirname(TEST_DIR)
sys.path.append(ROOT_DIR)

# Draw off screen, as CI does, so tests that check a figure's layout do not
# depend on the GUI backend or screen of the machine running them.
os.environ.setdefault("MPLBACKEND", "Agg")


def run_this_test(file):
    pytest.main(args=["-W", "error", "-Werror", "--pythonwarnings=error", os.path.abspath(file)])
