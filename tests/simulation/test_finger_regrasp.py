"""Geometric contracts for the user's fingertip-anchored regrasp preview."""
import importlib
import importlib.util

import numpy as np
import pytest
import mujoco
from robot_core.paths import data_root


@pytest.fixture(scope="module")
def plan():
    if not (data_root()/"recordings/recorded_hand_guarded_chop_200hz_latest.npz").is_file():
        pytest.skip("local recorded arm pose is unavailable")
    assert importlib.util.find_spec("twin_sim.finger_regrasp") is not None, "finger-led regrasp is missing"
    return importlib.import_module("twin_sim.finger_regrasp").build_finger_regrasp_preview()


def test_knuckles_retract_while_three_tips_and_wrist_stay_fixed(plan):
    mask = np.array(plan.phases) == "KNUCKLE_BACK"
    tip_drift = np.linalg.norm(plan.tips_m[mask] - plan.tips_m[0], axis=2)
    assert tip_drift.max() < .0003
    assert np.all(plan.leads_m[0] > .004)
    assert np.all(np.abs(plan.leads_m[mask][-1]) < .002)
    np.testing.assert_allclose(plan.qpos[:, plan.left_qpos_ids],
                               np.broadcast_to(plan.qpos[0, plan.left_qpos_ids], (len(plan.qpos), 7)), atol=1e-12)


def test_three_fingers_lift_and_move_back_on_the_same_clock(plan):
    lift = np.array(plan.phases) == "TRIO_LIFT"
    retreat = np.array(plan.phases) == "TRIO_RETREAT"
    assert lift.any() and retreat.any()
    assert np.all(plan.tips_m[lift][-1, :, 2] > plan.tips_m[0, :, 2]+.005)
    assert np.all(plan.tips_m[retreat][-1, :, 1] > plan.tips_m[0, :, 1]+.005)
    for mask in (lift, retreat):
        displacement = plan.tips_m[mask]-plan.tips_m[mask][0]
        assert np.max(np.ptp(displacement, axis=1)) < .0003


def test_thumb_and_little_finger_support_before_and_during_trio_movement(plan):
    phases = np.array(plan.phases)
    establish = np.where(phases == "SUPPORT_ESTABLISH")[0]
    lift = np.where(phases == "TRIO_LIFT")[0]
    assert len(establish) and len(lift) and establish[-1] < lift[0]
    active = np.isin(phases, ["TRIO_LIFT", "TRIO_RETREAT", "TRIO_PLACE"])
    assert np.max(np.abs(plan.support_distances_m[active])) < .0005
    drift = np.linalg.norm(plan.support_tips_m[active]-plan.support_tips_m[active][0], axis=2)
    assert drift.max() < .0001


def test_three_fingers_have_clearance_throughout_and_palm_is_lower(plan):
    assert np.min(plan.finger_clearances_m) > .0005
    data = mujoco.MjData(plan.model)
    data.qpos[:] = plan.qpos[0]
    mujoco.mj_forward(plan.model, data)
    palm = plan.model.body("left_palm_link").id
    height = data.xpos[palm, 2]
    with np.load(data_root()/"recordings/recorded_hand_guarded_chop_200hz_latest.npz") as source:
        data.qpos[plan.left_qpos_ids] = source["left_arm_target_rad"][0]
    mujoco.mj_forward(plan.model, data)
    previous_height = data.xpos[palm, 2]+.024
    assert previous_height-height == pytest.approx(.008, abs=.00005)


def test_three_fingertips_make_contact_at_start_and_finish(plan):
    assert np.max(np.abs(plan.tip_distances_m[[0, -1]])) < .0008


def test_plan_stays_within_joint_limits_and_has_finite_states(plan):
    assert np.isfinite(plan.qpos).all()
    assert np.all(np.diff(plan.times_s) > 0)
    assert np.max(np.abs(np.diff(plan.qpos[:, plan.hand_qpos_ids], axis=0))) < .10
    model = plan.model
    qids = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    assert np.all(plan.qpos[:, qids] >= model.actuator_ctrlrange[:, 0]-1e-8)
    assert np.all(plan.qpos[:, qids] <= model.actuator_ctrlrange[:, 1]+1e-8)


def _static_hold_loads(plan, index):
    model, data = plan.model, mujoco.MjData(plan.model)
    data.qpos[:] = plan.qpos[index]
    data.ctrl[:] = data.qpos[model.jnt_qposadr[model.actuator_trnid[:, 0]]]
    potato = model.geom("potato_collision").id
    loads = {}
    for step in range(500):
        mujoco.mj_step(model, data)
        if step < 250:
            continue
        for index, contact in enumerate(data.contact):
            pair = (int(contact.geom1), int(contact.geom2))
            if potato not in pair:
                continue
            other = pair[1] if pair[0] == potato else pair[0]
            name = model.geom(other).name
            if name.startswith("left_"):
                wrench = np.zeros(6)
                mujoco.mj_contactForce(model, data, index, wrench)
                loads[name] = loads.get(name, 0)+max(0, wrench[0])/250
    return loads


def test_static_physical_hold_loads_only_the_three_requested_fingertips(plan):
    loads = _static_hold_loads(plan, 0)
    expected = {f"left_finger{f}_pad" for f in (2, 3, 4)}
    assert all(loads.get(name, 0) > .1 for name in expected)
    assert sum(value for name, value in loads.items() if name not in expected) < .01


def test_support_fingers_carry_load_with_three_main_fingers_raised(plan):
    index = np.where(np.array(plan.phases) == "TRIO_RETREAT")[0][-1]
    loads = _static_hold_loads(plan, index)
    expected = {"left_finger1_pad", "left_finger5_pad"}
    assert all(loads.get(name, 0) > .2 for name in expected)
    assert sum(value for name, value in loads.items() if name not in expected) < .02
