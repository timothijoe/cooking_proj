"""Small vertical wrist assistance must preserve palm orientation and trio motion."""
import mujoco
import numpy as np
from twin_sim.dynamic_regrasp import DynamicRegraspEnv


def test_vertical_assistance_keeps_orientation_and_finger_retreat():
    env = DynamicRegraspEnv(cycles=1, wrist_lift_m=.005)
    plan = env.plan
    data = mujoco.MjData(env.model)
    site = env.model.site('left_palm_tcp_site').id
    positions, rotations = [], []
    for qpos in plan.qpos:
        data.qpos[:] = qpos
        mujoco.mj_fwdPosition(env.model, data)
        positions.append(data.site_xpos[site].copy())
        rotations.append(data.site_xmat[site].copy())
    positions = np.asarray(positions)
    rotations = np.asarray(rotations)
    np.testing.assert_allclose(positions[:, :2], np.broadcast_to(positions[0, :2], positions[:, :2].shape), atol=1e-6)
    np.testing.assert_allclose(rotations, np.broadcast_to(rotations[0], rotations.shape), atol=1e-5)
    assert .00499 < np.ptp(positions[:, 2]) < .00501
    np.testing.assert_allclose(positions[-1], positions[0], atol=1e-6)
    lift = np.where(np.array(plan.phases) == 'TRIO_LIFT')[0]
    retreat = np.where(np.array(plan.phases) == 'TRIO_RETREAT')[0]
    np.testing.assert_allclose(plan.tips_m[lift[-1], :, 2] - plan.tips_m[lift[0]-1, :, 2], .012, atol=.0001)
    np.testing.assert_allclose(plan.tips_m[retreat[-1], :, 1] - plan.tips_m[retreat[0]-1, :, 1], .008, atol=.0001)
