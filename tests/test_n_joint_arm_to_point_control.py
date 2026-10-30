import conftest  # Add root path to sys.path
import numpy as np
import pytest

from ArmNavigation.n_joint_arm_to_point_control\
    import n_joint_arm_to_point_control as m


def run_animation(monkeypatch):
    """
    Run animation() and return, for every goal in order, the goal and the
    joint angles the arm had when it moved on to the next goal.
    """
    arms = []

    class RecordingArm(m.NLinkArm):
        def __init__(self, *args, **kwargs):
            self.history = []  # (goal, joint angles) after every update
            super().__init__(*args, **kwargs)
            arms.append(self)

        def update_joints(self, joint_angles):
            super().update_joints(joint_angles)
            self.history.append((tuple(self.goal), tuple(joint_angles)))

    monkeypatch.setattr(m, "show_animation", False)
    monkeypatch.setattr(m, "NLinkArm", RecordingArm)
    m.animation()

    last_angles_per_goal = {}
    for goal, joint_angles in arms[0].history:
        last_angles_per_goal[goal] = joint_angles  # dicts keep goal order
    return list(last_angles_per_goal.items())


def end_effector(joint_angles, link_length=1.0):
    """Forward kinematics of the planar arm, independent of the module."""
    absolute_angles = np.cumsum(joint_angles)
    return link_length * np.array([np.sum(np.cos(absolute_angles)),
                                   np.sum(np.sin(absolute_angles))])


@pytest.fixture(scope="module")
def reached():
    """animation() runs once for the tests that only inspect its result."""
    with pytest.MonkeyPatch.context() as mp:
        return run_animation(mp)


def test1(reached):
    # animation() stops after 5 goals; with the fixed seed every inverse
    # kinematics solve succeeds, so no goal is skipped.
    assert len(reached) == 5
    for goal, joint_angles in reached:
        # the arm moves on once it is within 0.1 m of the goal
        distance = np.linalg.norm(end_effector(joint_angles) - goal)
        assert distance < 0.1  # observed at most 0.093 m
        # goals are drawn from the 15 m x 15 m square around the base
        assert max(abs(goal[0]), abs(goal[1])) <= 7.5


def test_animation_is_reproducible(reached, monkeypatch):
    assert run_animation(monkeypatch) == reached


if __name__ == '__main__':
    conftest.run_this_test(__file__)
