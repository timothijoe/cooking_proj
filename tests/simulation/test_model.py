import os
from xml.etree import ElementTree

import mujoco
import numpy as np
import pytest

from twin_sim.model import ModelValidationError, SimulationModel
from twin_sim.hand_names import HAND_ACTUATORS, HAND_JOINTS
from robot_core.paths import asset_root
from twin_sim.robot import RightArmRobot


def _custom_scene_without_guarded_features(tmp_path):
    root = asset_root()
    scene_dir = tmp_path / "robot_assets" / "mujoco"
    hand_dir = scene_dir / "wuji_hand"
    hand_dir.mkdir(parents=True)
    os.symlink(root / "MarvinCCS", tmp_path / "MarvinCCS")
    os.symlink(root / "robot_assets" / "mujoco" / "meshes", scene_dir / "meshes")
    os.symlink(
        root / "robot_assets" / "mujoco" / "wuji_hand" / "meshes",
        hand_dir / "meshes",
    )

    scene = ElementTree.parse(
        root / "robot_assets" / "mujoco" / "right_chopping_scene.xml"
    )
    worldbody = scene.getroot().find("worldbody")
    guarded_body = next(
        body
        for body in worldbody
        if body.get("name") == "guarded_chop_cube_body"
    )
    worldbody.remove(guarded_body)
    custom_path = scene_dir / "custom_right_task.xml"
    scene.write(custom_path, encoding="unicode")

    hand = ElementTree.parse(
        root / "robot_assets" / "mujoco" / "wuji_hand" / "left_hand.xml"
    )
    excluded_sites = {"left_palm_tcp_site", "left_guard_knuckle_site"}
    for parent in hand.iter():
        for child in list(parent):
            if child.get("name") in excluded_sites:
                parent.remove(child)
    hand.write(hand_dir / "left_hand.xml", encoding="unicode")
    return custom_path


def test_active_scene_has_arm_and_hand_position_actuators():
    sim = SimulationModel.load()
    actuator_ids = (*sim.left.actuator_ids, *sim.right.actuator_ids)
    assert sim.model.njnt == 35
    assert sim.model.nu == 34
    assert len(actuator_ids) == 14
    assert sim.left.actuator_ids.shape == (7,)
    assert sim.right.actuator_ids.shape == (7,)
    assert all(
        sim.model.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_AFFINE
        for i in actuator_ids
    )


def test_custom_right_task_model_does_not_require_guarded_chop_objects(
    tmp_path,
):
    sim = SimulationModel.load(
        _custom_scene_without_guarded_features(tmp_path)
    )

    assert sim.require_site("right_tool_tip_site") >= 0
    with pytest.raises(ModelValidationError, match="guarded_chop_cube"):
        sim.require_geom("guarded_chop_cube")


def test_custom_right_task_robot_does_not_require_left_palm_tcp(tmp_path):
    robot = RightArmRobot(
        _custom_scene_without_guarded_features(tmp_path), viewer=False
    )
    try:
        assert np.isfinite(robot.tcp_pose()).all()
        robot.step(0.002)
    finally:
        robot.close()


def test_left_hand_is_mounted_below_left_wrist():
    sim = SimulationModel.load()
    mount_id = mujoco.mj_name2id(
        sim.model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_mount"
    )
    palm_id = mujoco.mj_name2id(
        sim.model, mujoco.mjtObj.mjOBJ_BODY, "left_palm_link"
    )
    wrist_id = mujoco.mj_name2id(
        sim.model, mujoco.mjtObj.mjOBJ_BODY, "left_link7"
    )
    assert min(mount_id, palm_id, wrist_id) >= 0
    ancestors = []
    body_id = palm_id
    while body_id:
        ancestors.append(body_id)
        body_id = int(sim.model.body_parentid[body_id])
    assert mount_id in ancestors
    assert wrist_id in ancestors


def test_left_hand_indices_preserve_upstream_order():
    sim = SimulationModel.load()
    expected_joints = tuple(
        f"left_finger{finger}_joint{joint}"
        for finger in range(1, 6)
        for joint in range(1, 5)
    )
    assert HAND_JOINTS == expected_joints
    assert HAND_ACTUATORS == tuple(f"{name}_actuator" for name in expected_joints)
    assert sim.hand.joint_ids.shape == (20,)
    assert sim.hand.qpos_ids.shape == (20,)
    assert sim.hand.dof_ids.shape == (20,)
    assert sim.hand.actuator_ids.shape == (20,)
    np.testing.assert_array_equal(
        sim.model.actuator_trnid[sim.hand.actuator_ids, 0],
        sim.hand.joint_ids,
    )


def test_active_scene_has_task_objects():
    sim = SimulationModel.load()
    assert sim.require_site("right_tool_tip_site") >= 0
    assert sim.require_geom("chopping_board") >= 0
    assert sim.require_sensor("right_tool_force") >= 0


def test_missing_model_object_has_clear_error():
    sim = SimulationModel.load()
    with pytest.raises(ModelValidationError, match="missing site: absent"):
        sim.require_site("absent")


def test_model_rejects_an_actuator_transmitted_to_the_wrong_joint():
    sim = SimulationModel.load()
    actuator_id = sim.right.actuator_ids[0]
    sim.model.actuator_trnid[actuator_id, 0] = sim.right.joint_ids[1]

    with pytest.raises(ModelValidationError, match="transmission.*act_right_joint1"):
        sim.validate_actuator_contract()


def test_model_rejects_non_unit_actuator_gear():
    sim = SimulationModel.load()
    actuator_id = sim.right.actuator_ids[0]
    sim.model.actuator_gear[actuator_id, 0] = 2.0

    with pytest.raises(ModelValidationError, match="unit gear.*act_right_joint1"):
        sim.validate_actuator_contract()


def test_active_wrist_force_limits_match_canonical_model():
    sim = SimulationModel.load()
    wrist_ids = sim.right.actuator_ids[4:]
    np.testing.assert_array_equal(
        sim.model.actuator_forcerange[wrist_ids],
        np.tile((-18.0, 18.0), (3, 1)),
    )
