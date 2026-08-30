import numpy as np
import pytest

from twin_sim.kinematics import Kinematics
from twin_sim.robot import RIGHT_HOME_RAD, RightArmRobot
from twin_sim.trajectory import cartesian_trajectory, joint_trajectory


def test_joint_trajectory_has_exact_endpoints_and_monotonic_time():
    start, goal = np.zeros(7), np.ones(7) * 0.1

    points = joint_trajectory(start, goal, duration_s=1.0, control_dt_s=0.01)

    np.testing.assert_array_equal(points[0].joints_rad, start)
    np.testing.assert_array_equal(points[-1].joints_rad, goal)
    assert all(a.time_s < b.time_s for a, b in zip(points, points[1:]))


def test_joint_trajectory_starts_and_ends_at_rest():
    points = joint_trajectory(np.zeros(7), np.ones(7) * 0.1, 1.0, 0.01)

    np.testing.assert_array_equal(points[0].velocity_rad_s, np.zeros(7))
    np.testing.assert_allclose(points[-1].velocity_rad_s, np.zeros(7), atol=1e-12)


@pytest.mark.parametrize(
    ("duration_s", "control_dt_s", "message"),
    ((0.0, 0.01, "positive"), (1.0, 0.0, "positive"), (1.0, 0.03, "integer")),
)
def test_joint_trajectory_rejects_invalid_timing(duration_s, control_dt_s, message):
    with pytest.raises(ValueError, match=message):
        joint_trajectory(np.zeros(7), np.ones(7), duration_s, control_dt_s)


def test_joint_trajectory_rejects_motion_above_default_velocity_limit():
    with pytest.raises(ValueError, match="velocity limit"):
        joint_trajectory(np.zeros(7), np.ones(7), 0.01, 0.01)


def test_cartesian_trajectory_preserves_pose_endpoints_and_preflights_ik():
    robot = RightArmRobot()
    kinematics = Kinematics(robot.sim)
    start = kinematics.fk(RIGHT_HOME_RAD)
    goal_joints = RIGHT_HOME_RAD + np.array((0.02, 0, 0, 0, 0, 0, 0))
    goal = kinematics.fk(goal_joints)

    points = cartesian_trajectory(
        kinematics,
        start,
        goal,
        RIGHT_HOME_RAD,
        duration_s=0.2,
        control_dt_s=0.01,
    )

    np.testing.assert_array_equal(points[0].target_pose, start)
    np.testing.assert_array_equal(points[-1].target_pose, goal)
    for point in points:
        np.testing.assert_allclose(
            kinematics.fk(point.joints_rad), point.target_pose, atol=1e-4
        )


def test_cartesian_trajectory_rejects_undersampled_nonzero_motion():
    robot = RightArmRobot()
    kinematics = Kinematics(robot.sim)
    start = kinematics.fk(RIGHT_HOME_RAD)
    goal = kinematics.fk(
        RIGHT_HOME_RAD + np.array((0.01, 0, 0, 0, 0, 0, 0))
    )

    with pytest.raises(ValueError, match="acceleration limit"):
        cartesian_trajectory(
            kinematics,
            start,
            goal,
            RIGHT_HOME_RAD,
            duration_s=0.01,
            control_dt_s=0.01,
        )


def test_joint_trajectory_acceleration_includes_zero_endpoint_velocities():
    with pytest.raises(ValueError, match="acceleration limit"):
        joint_trajectory(np.zeros(7), np.ones(7) * 0.01, 0.01, 0.01)
