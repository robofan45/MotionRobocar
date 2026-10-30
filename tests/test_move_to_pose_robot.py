import conftest  # Add root path to sys.path
import math
import os
import sys

from PathTracking.move_to_pose import move_to_pose as m
from utils.angle import angle_mod

# move_to_pose_robot.py imports its sibling module by the bare name
# "move_to_pose", so its folder has to be on sys.path.
sys.path.append(os.path.join(conftest.ROOT_DIR, "PathTracking", "move_to_pose"))
import move_to_pose_robot as robot_demo  # noqa: E402

MOVE_TO_POSE = m.move_to_pose  # the unwrapped function


def test_1(monkeypatch):
    """
    This unit test tests the move_to_pose_robot.py program
    """
    monkeypatch.setattr(robot_demo, "SHOW_ANIMATION", False)
    # main() leaves simulation_running False, which would stop a second run
    monkeypatch.setattr(robot_demo, "simulation_running", True)
    robots = robot_demo.main()

    # All three robots (different gains and speed limits) reach the target
    # pose from the same start pose.
    assert len(robots) == 3
    for robot in robots:
        target = robot.pose_target
        assert robot.is_at_target
        # stops within 0.01 m; the last step is taken after the check
        assert math.hypot(robot.pose.x - target.x,
                          robot.pose.y - target.y) < 0.02  # observed 0.0093 m
        # the controller does not wait for the heading to converge
        assert abs(angle_mod(robot.pose.theta - target.theta)) < 0.5  # observed 0.24 rad


def run_random_poses(monkeypatch):
    """Run move_to_pose.main(); return the arguments and result of each move."""
    moves = []

    def recording_move_to_pose(*pose_args):
        trajectory = MOVE_TO_POSE(*pose_args)
        moves.append((pose_args, trajectory))
        return trajectory

    monkeypatch.setattr(m, "show_animation", False)
    monkeypatch.setattr(m, "move_to_pose", recording_move_to_pose)
    # the demo's own speed limits (tests may have changed them)
    monkeypatch.setattr(m, "MAX_LINEAR_SPEED", 15)
    monkeypatch.setattr(m, "MAX_ANGULAR_SPEED", 7)
    m.main()
    return moves


def test_random_poses(monkeypatch):
    """move_to_pose.main() drives the robot from 5 random poses to 5 goals."""
    moves = run_random_poses(monkeypatch)

    assert len(moves) == 5
    for pose_args, trajectory in moves:
        x_start, y_start, theta_start, x_goal, y_goal, theta_goal = pose_args
        x_traj, y_traj, v_traj, w_traj = trajectory
        # stopped because it arrived (rho < 0.001), not at MAX_SIM_TIME
        assert math.hypot(x_traj[-1] - x_goal, y_traj[-1] - y_goal) < 0.001
        # final heading = start heading + integrated turn rate commands
        theta_end = theta_start + m.dt * sum(w_traj)
        assert abs(angle_mod(theta_end - theta_goal)) < 0.25  # observed 0.12 rad
        # commands respect the speed limits
        assert max(abs(v) for v in v_traj) <= m.MAX_LINEAR_SPEED
        assert max(abs(w) for w in w_traj) <= m.MAX_ANGULAR_SPEED


def test_random_poses_are_reproducible(monkeypatch):
    assert run_random_poses(monkeypatch) == run_random_poses(monkeypatch)


if __name__ == '__main__':
    conftest.run_this_test(__file__)
