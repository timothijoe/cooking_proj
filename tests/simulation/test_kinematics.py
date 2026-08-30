import mujoco
import numpy as np
import pytest

from twin_sim.kinematics import Kinematics, PathIkError
from twin_sim.robot import RIGHT_HOME_RAD, RightArmRobot


def _state_snapshot(robot):
    data = robot.sim.data
    return {
        "qpos": data.qpos.copy(),
        "qvel": data.qvel.copy(),
        "ctrl": data.ctrl.copy(),
        "time": data.time,
        "site_xpos": data.site_xpos.copy(),
        "site_xmat": data.site_xmat.copy(),
    }


def _assert_state_matches(robot, expected):
    data = robot.sim.data
    np.testing.assert_array_equal(data.qpos, expected["qpos"])
    np.testing.assert_array_equal(data.qvel, expected["qvel"])
    np.testing.assert_array_equal(data.ctrl, expected["ctrl"])
    assert data.time == expected["time"]
    np.testing.assert_allclose(data.site_xpos, expected["site_xpos"], atol=1e-12)
    np.testing.assert_allclose(data.site_xmat, expected["site_xmat"], atol=1e-12)


def test_fk_ik_roundtrip():
    robot = RightArmRobot()
    kin = Kinematics(robot.sim)
    target = kin.fk(RIGHT_HOME_RAD)
    result = kin.ik(target, RIGHT_HOME_RAD)
    assert result.success
    assert np.allclose(kin.fk(result.joints_rad), target, atol=1e-4)


def test_unreachable_path_fails_before_execution():
    robot = RightArmRobot()
    kin = Kinematics(robot.sim)
    pose = np.eye(4)
    pose[:3, 3] = (10, 10, 10)
    with pytest.raises(PathIkError, match="sample 0"):
        kin.solve_path([pose], RIGHT_HOME_RAD)


def test_malformed_path_pose_reports_its_sample_index_and_cause():
    robot = RightArmRobot()
    kin = Kinematics(robot.sim)
    valid_pose = kin.fk(RIGHT_HOME_RAD)

    with pytest.raises(PathIkError, match="sample 1") as error:
        kin.solve_path([valid_pose, np.eye(3)], RIGHT_HOME_RAD)

    assert isinstance(error.value.__cause__, ValueError)


@pytest.mark.parametrize(
    "pose",
    (
        np.array(
            (
                (1, 0, 0, 0),
                (0, 1, 0, 0),
                (0, 0, 1, 0),
                (1, 0, 0, 1),
            ),
            dtype=float,
        ),
        np.diag((-1.0, 1.0, 1.0, 1.0)),
    ),
)
def test_ik_rejects_a_non_rigid_target_pose(pose):
    robot = RightArmRobot()
    kin = Kinematics(robot.sim)

    with pytest.raises(ValueError, match="rigid 4x4 pose"):
        kin.ik(pose, RIGHT_HOME_RAD)


def test_ik_rejects_a_fractional_iteration_limit():
    robot = RightArmRobot()
    kin = Kinematics(robot.sim)

    with pytest.raises(ValueError, match="non-negative integer"):
        kin.ik(np.eye(4), RIGHT_HOME_RAD, max_iterations=1.5)


@pytest.mark.parametrize("query", ["fk", "jacobian", "ik"])
def test_queries_restore_shared_mujoco_state(query):
    robot = RightArmRobot()
    kin = Kinematics(robot.sim)
    data = robot.sim.data
    data.qvel[:] = np.linspace(-0.7, 0.7, data.qvel.size)
    data.ctrl[:] = np.linspace(-1.0, 1.0, data.ctrl.size)
    data.time = 1.25
    mujoco.mj_forward(robot.sim.model, data)
    expected = _state_snapshot(robot)

    if query == "fk":
        kin.fk(RIGHT_HOME_RAD + 0.01)
    elif query == "jacobian":
        kin.jacobian(RIGHT_HOME_RAD + 0.01)
    else:
        kin.ik(kin.fk(RIGHT_HOME_RAD + 0.01), RIGHT_HOME_RAD)

    _assert_state_matches(robot, expected)


def test_fk_restores_shared_state_when_mujoco_query_fails(monkeypatch):
    robot = RightArmRobot()
    kin = Kinematics(robot.sim)
    expected = _state_snapshot(robot)
    real_forward = mujoco.mj_forward
    calls = 0

    def fail_once(model, data):
        nonlocal calls
        calls += 1
        real_forward(model, data)
        if calls == 1:
            raise RuntimeError("injected forward failure")

    monkeypatch.setattr(mujoco, "mj_forward", fail_once)
    with pytest.raises(RuntimeError, match="injected"):
        kin.fk(RIGHT_HOME_RAD + 0.01)

    _assert_state_matches(robot, expected)


@pytest.mark.parametrize("query", ["fk", "jacobian", "ik"])
def test_queries_restore_post_step_derived_state(query):
    robot = RightArmRobot()
    kin = Kinematics(robot.sim)
    robot.command(RIGHT_HOME_RAD + 0.01)
    robot.step(0.002)
    data = robot.sim.data
    expected = {
        name: getattr(data, name).copy()
        for name in (
            "qacc",
            "qfrc_bias",
            "qfrc_constraint",
            "sensordata",
            "site_xpos",
            "site_xmat",
        )
    }

    if query == "fk":
        kin.fk(RIGHT_HOME_RAD + 0.02)
    elif query == "jacobian":
        kin.jacobian(RIGHT_HOME_RAD + 0.02)
    else:
        kin.ik(kin.fk(RIGHT_HOME_RAD + 0.02), RIGHT_HOME_RAD)

    for name, values in expected.items():
        np.testing.assert_array_equal(getattr(data, name), values)


def test_solve_path_seeds_each_sample_from_previous_solution():
    robot = RightArmRobot()
    kin = Kinematics(robot.sim)
    joints = np.vstack(
        (
            RIGHT_HOME_RAD,
            RIGHT_HOME_RAD + np.array((0.02, 0, 0, 0, 0, 0, 0)),
            RIGHT_HOME_RAD + np.array((0.04, 0, 0, 0, 0, 0, 0)),
        )
    )
    poses = [kin.fk(sample) for sample in joints]

    solved = kin.solve_path(poses, RIGHT_HOME_RAD)

    assert solved.shape == (3, 7)
    for actual, target in zip(solved, poses, strict=True):
        np.testing.assert_allclose(kin.fk(actual), target, atol=1e-4)


def test_solve_path_rejects_a_joint_discontinuity():
    robot = RightArmRobot()
    kin = Kinematics(robot.sim)
    pose = kin.fk(RIGHT_HOME_RAD + np.array((0.2, 0, 0, 0, 0, 0, 0)))

    with pytest.raises(PathIkError, match="joint-step violation.*sample 0"):
        kin.solve_path([pose], RIGHT_HOME_RAD, max_joint_step_rad=0.01)
