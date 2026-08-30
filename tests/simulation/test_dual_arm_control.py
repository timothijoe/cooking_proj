import numpy as np
import pytest

from twin_sim.robot import RightArmRobot


def test_left_command_moves_left_and_preserves_right_target():
    robot = RightArmRobot()
    right_target = robot._right_target.copy()
    left = robot.left_joint_positions
    goal = left.copy()
    goal[1] += 0.02
    robot.command_left(goal)
    for _ in range(20):
        robot.step(0.01)
    assert robot.left_joint_positions[1] > left[1]
    np.testing.assert_array_equal(robot._right_target, right_target)
    robot.close()


def test_invalid_left_target_is_atomic():
    robot = RightArmRobot()
    before = robot._left_target.copy()
    with pytest.raises(ValueError, match="left joints_rad"):
        robot.command_left(np.zeros(6))
    np.testing.assert_array_equal(robot._left_target, before)
    robot.close()


def test_out_of_range_left_target_is_atomic():
    robot = RightArmRobot()
    before = robot._left_target.copy()
    invalid = before.copy()
    invalid[3] = robot.sim.model.jnt_range[robot.sim.left.joint_ids[3], 1] + 0.1
    with pytest.raises(ValueError, match="left joint 4 target"):
        robot.command_left(invalid)
    np.testing.assert_array_equal(robot._left_target, before)
    robot.close()


def test_left_palm_pose_matches_left_kinematics_fk():
    robot = RightArmRobot()
    np.testing.assert_allclose(
        robot.left_palm_pose(),
        robot.left_kinematics.fk(robot.left_joint_positions),
    )
    robot.close()
