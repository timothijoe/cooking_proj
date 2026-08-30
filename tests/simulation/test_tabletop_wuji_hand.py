import mujoco
import numpy as np
import pytest

import tianji_robotics.simulation.tabletop_wuji_hand as tabletop_module
from tianji_robotics.simulation.tabletop_wuji_hand import TabletopWujiHand


def test_tabletop_scene_contains_only_hand_table_and_tip_sites():
    hand = TabletopWujiHand(viewer=False, table_height_m=0.0)
    try:
        assert (hand.model.nq, hand.model.nv, hand.model.nu, hand.model.nmocap) == (20, 20, 20, 1)
        assert mujoco.mj_name2id(hand.model, mujoco.mjtObj.mjOBJ_GEOM, "table") >= 0
        assert all(mujoco.mj_name2id(hand.model, mujoco.mjtObj.mjOBJ_SITE, f"finger{i}_contact") >= 0 for i in range(2, 6))
        assert mujoco.mj_name2id(hand.model, mujoco.mjtObj.mjOBJ_SITE, "thumb_clearance") >= 0
        assert mujoco.mj_name2id(hand.model, mujoco.mjtObj.mjOBJ_SITE, "palmar_reference") >= 0
        assert mujoco.mj_name2id(hand.model, mujoco.mjtObj.mjOBJ_SITE, "dorsal_reference") >= 0
        names = bytes(hand.model.names).decode(errors="ignore").lower()
        assert "left_link" not in names and "right_link" not in names
    finally:
        hand.close()


def test_command_pose_updates_palm_joints_and_observations():
    hand = TabletopWujiHand(viewer=False)
    try:
        joints = np.array([(lo + hi) / 2 for lo, hi in hand.joint_ranges_rad.values()])
        position = np.array([0.1, -0.2, 0.3])
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        hand.command_pose(joints, position, quaternion)
        np.testing.assert_allclose(hand.read_target_position_rad(), joints)
        np.testing.assert_allclose(hand.read_palm_position_m(), position)
        assert hand.fingertip_positions_m().shape == (4, 3)
        assert hand.thumb_position_m().shape == (3,)
    finally:
        hand.close()


def test_anatomical_landmarks_return_four_world_positions():
    hand = TabletopWujiHand(viewer=False)
    try:
        hand.set_kinematic_pose(np.zeros(20), [0, 0, 0.2], [1, 0, 0, 0])
        assert hand.long_finger_root_positions_m().shape == (4, 3)
    finally:
        hand.close()


def test_palm_reference_positions_are_distinct_world_points():
    hand = TabletopWujiHand(viewer=False)
    try:
        hand.set_kinematic_pose(np.zeros(20), [0, 0, .2], [1, 0, 0, 0])
        palmar = hand.palmar_reference_position_m()
        dorsal = hand.dorsal_reference_position_m()
        assert palmar.shape == (3,) and dorsal.shape == (3,)
        assert np.linalg.norm(palmar - dorsal) > .01
    finally:
        hand.close()


def test_real_collision_clearance_detects_table_crossing():
    hand = TabletopWujiHand(viewer=False)
    try:
        hand.set_kinematic_pose(np.zeros(20), [0, 0, -0.05], [1, 0, 0, 0])
        assert hand.maximum_table_penetration_m() > 0.01
        assert hand.minimum_hand_table_clearance_m() < -0.01
    finally:
        hand.close()


@pytest.mark.parametrize("position,quaternion", [([0, 0], [1, 0, 0, 0]), ([0, 0, 0], [0, 0, 0, 0])])
def test_command_pose_rejects_invalid_palm_pose(position, quaternion):
    hand = TabletopWujiHand(viewer=False)
    try:
        with pytest.raises(ValueError, match="palm pose"):
            hand.command_pose(np.zeros(20), position, quaternion)
    finally:
        hand.close()


class FakeViewer:
    def __init__(self, running_sequence):
        self._running = iter(running_sequence)
        self.is_running_calls = 0
        self.close_calls = 0

    def is_running(self):
        self.is_running_calls += 1
        return next(self._running, False)

    def close(self):
        self.close_calls += 1


def test_close_does_not_close_viewer_twice_after_user_closed_window():
    hand = TabletopWujiHand(viewer=False)
    fake = FakeViewer([False])
    hand._viewer = fake

    hand.close()
    hand.close()

    assert fake.close_calls == 0


def test_wait_returns_when_viewer_window_closes(monkeypatch):
    hand = TabletopWujiHand(viewer=False)
    fake = FakeViewer([True, True, False])
    hand._viewer = fake
    monkeypatch.setattr(tabletop_module.time, "sleep", lambda _: None)
    try:
        hand.wait_until_viewer_closes()
        assert fake.is_running_calls == 3
    finally:
        hand.close()


def test_headless_wait_is_a_noop():
    hand = TabletopWujiHand(viewer=False)
    try:
        assert hand.has_viewer is False
        hand.wait_until_viewer_closes()
    finally:
        hand.close()
