import mujoco
import numpy as np
import pytest

from twin_sim.guarded_chop_safety import (
    CAT_PAW_OPEN_RAD,
    CAT_PAW_RAD,
    GUARD_RELAXED_RAD,
    GUARD_RETRACTED_RAD,
)


def _finger_link_state(robot, hand_rad, left_rad, finger, link):
    saved = mujoco.MjData(robot.sim.model)
    mujoco.mj_copyData(saved, robot.sim.model, robot.sim.data)
    try:
        robot.sim.data.qpos[robot.sim.left.qpos_ids] = left_rad
        robot.sim.data.qpos[robot.sim.hand.qpos_ids] = hand_rad
        mujoco.mj_forward(robot.sim.model, robot.sim.data)
        palm = robot.sim.require_body("left_palm_link")
        body = robot.sim.require_body(f"left_finger{finger}_link{link}")
        palm_rotation = robot.sim.data.xmat[palm].reshape(3, 3)
        position = palm_rotation.T @ (
            robot.sim.data.xpos[body] - robot.sim.data.xpos[palm]
        )
        axis = (
            palm_rotation.T
            @ robot.sim.data.xmat[body].reshape(3, 3)[:, 2]
        )
        return position.copy(), axis.copy()
    finally:
        mujoco.mj_copyData(robot.sim.data, robot.sim.model, saved)
from twin_sim.model import SimulationModel
from twin_sim.robot import RightArmRobot
from twin_sim.tasks.guarded_chop import (
    GuardedChopConfig,
    _finger_pad_position,
    _preflight_guarded_chop,
)


def test_cat_paw_target_is_valid_and_tucks_thumb():
    sim = SimulationModel.load()
    ranges = sim.model.actuator_ctrlrange[sim.hand.actuator_ids]

    assert CAT_PAW_RAD.shape == (20,)
    assert np.all(CAT_PAW_RAD >= ranges[:, 0])
    assert np.all(CAT_PAW_RAD <= ranges[:, 1])
    assert CAT_PAW_RAD[0] > 0.6
    assert np.all(CAT_PAW_RAD[[2, 6, 10, 14, 18]] > 0.45)


def test_open_cat_paw_is_visible_and_within_hand_ranges():
    sim = SimulationModel.load()
    ranges = sim.model.actuator_ctrlrange[sim.hand.actuator_ids]

    assert CAT_PAW_OPEN_RAD.shape == (20,)
    assert np.all(CAT_PAW_OPEN_RAD >= ranges[:, 0])
    assert np.all(CAT_PAW_OPEN_RAD <= ranges[:, 1])
    assert np.linalg.norm(CAT_PAW_OPEN_RAD - CAT_PAW_RAD) >= 1.0
    assert np.all(CAT_PAW_OPEN_RAD[[6, 10, 14, 18]] < CAT_PAW_RAD[[6, 10, 14, 18]])


def test_guard_synergy_is_layered_and_within_ranges():
    sim = SimulationModel.load()
    ranges = sim.model.actuator_ctrlrange[sim.hand.actuator_ids]

    for pose in (GUARD_RELAXED_RAD, GUARD_RETRACTED_RAD):
        assert pose.shape == (20,)
        assert np.all(pose >= ranges[:, 0])
        assert np.all(pose <= ranges[:, 1])

    changes = np.asarray(
        [
            np.linalg.norm(
                GUARD_RETRACTED_RAD[start : start + 4]
                - GUARD_RELAXED_RAD[start : start + 4]
            )
            for start in range(0, 20, 4)
        ]
    )
    assert changes[2] > changes[1] > changes[3] >= changes[4]
    assert changes[0] <= 0.25
    principal_degrees = np.asarray(
        [
            np.max(
                np.abs(
                    GUARD_RETRACTED_RAD[start : start + 4]
                    - GUARD_RELAXED_RAD[start : start + 4]
                )
            )
            * 180.0
            / np.pi
            for start in range(0, 20, 4)
        ]
    )
    assert principal_degrees[0] <= 5.0
    assert principal_degrees[1] >= 15.0
    assert principal_degrees[2] >= 20.0
    assert principal_degrees[3] >= 7.5
    assert principal_degrees[4] >= 7.5


def test_guard_synergy_retracts_finger3_pad_two_centimeters():
    robot = RightArmRobot(viewer=False)
    try:
        plan = _preflight_guarded_chop(robot, GuardedChopConfig())
        direction = plan.cut_points_xy[1] - plan.cut_points_xy[0]
        direction /= np.linalg.norm(direction)
        relaxed = _finger_pad_position(
            robot, GUARD_RELAXED_RAD, plan.left_ready_rad
        )
        retracted = _finger_pad_position(
            robot, GUARD_RETRACTED_RAD, plan.left_ready_rad
        )
    finally:
        robot.close()

    delta = retracted - relaxed
    retreat = float(delta[:2] @ direction)
    orthogonal = float(
        np.linalg.norm(delta - retreat * np.r_[direction, 0.0])
    )
    assert retreat == pytest.approx(0.020, abs=0.002)
    assert orthogonal <= 0.008


def test_guard_synergy_moves_visible_middle_links():
    robot = RightArmRobot(viewer=False)
    try:
        plan = _preflight_guarded_chop(robot, GuardedChopConfig())
        link_displacements = {}
        finger3_link3_axis_change = None
        for finger in range(2, 6):
            for link in (3, 4):
                relaxed_position, relaxed_axis = _finger_link_state(
                    robot,
                    GUARD_RELAXED_RAD,
                    plan.left_ready_rad,
                    finger,
                    link,
                )
                retracted_position, retracted_axis = _finger_link_state(
                    robot,
                    GUARD_RETRACTED_RAD,
                    plan.left_ready_rad,
                    finger,
                    link,
                )
                link_displacements[(finger, link)] = np.linalg.norm(
                    retracted_position - relaxed_position
                )
                if finger == 3 and link == 3:
                    finger3_link3_axis_change = np.degrees(
                        np.arccos(
                            np.clip(
                                relaxed_axis @ retracted_axis,
                                -1.0,
                                1.0,
                            )
                        )
                    )
    finally:
        robot.close()

    assert link_displacements[(3, 3)] >= 0.012
    assert link_displacements[(3, 4)] >= 0.018
    assert finger3_link3_axis_change >= 20.0
    assert link_displacements[(2, 4)] > link_displacements[(4, 4)]
    assert link_displacements[(4, 4)] >= link_displacements[(5, 4)]
