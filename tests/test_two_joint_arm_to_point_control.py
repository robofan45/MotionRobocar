import conftest  # Add root path to sys.path
import math

from ArmNavigation.two_joint_arm_to_point_control \
    import two_joint_arm_to_point_control as m

TWO_JOINT_ARM = m.two_joint_arm  # the unwrapped function


def run_animation(monkeypatch):
    """Run animation(); return the goal and final joint angles of each move."""
    moves = []

    def recording_two_joint_arm(**kwargs):
        theta1, theta2 = TWO_JOINT_ARM(**kwargs)
        moves.append(((m.x, m.y), (theta1, theta2)))
        return theta1, theta2

    monkeypatch.setattr(m, "show_animation", False)
    monkeypatch.setattr(m, "two_joint_arm", recording_two_joint_arm)
    # animation() overwrites the goal globals; restore them afterwards
    monkeypatch.setattr(m, "x", m.x)
    monkeypatch.setattr(m, "y", m.y)
    m.animation()
    return moves


def test1(monkeypatch):
    moves = run_animation(monkeypatch)

    assert len(moves) == 5
    for (goal_x, goal_y), (theta1, theta2) in moves:
        # goals are drawn from the square [-1, 1] x [-1, 1]
        assert abs(goal_x) <= 1.0 and abs(goal_y) <= 1.0
        # forward kinematics, independent of the module's plot_arm()
        wrist_x = m.l1 * math.cos(theta1) + m.l2 * math.cos(theta1 + theta2)
        wrist_y = m.l1 * math.sin(theta1) + m.l2 * math.sin(theta1 + theta2)
        # animation() moves on once the wrist is within GOAL_TH = 0.01 m
        assert math.hypot(wrist_x - goal_x, wrist_y - goal_y) < 0.01  # observed at most 0.0096 m


def test_animation_is_reproducible(monkeypatch):
    assert run_animation(monkeypatch) == run_animation(monkeypatch)


if __name__ == '__main__':
    conftest.run_this_test(__file__)
