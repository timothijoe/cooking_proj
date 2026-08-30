import numpy as np

from twin_sim.kinematics import Kinematics
from twin_sim.model import SimulationModel


def test_left_kinematics_changes_only_temporary_left_configuration():
    sim = SimulationModel.load()
    before_qpos = sim.data.qpos.copy()
    before_qvel = sim.data.qvel.copy()
    kine = Kinematics(sim, sim.left, "left_palm_tcp_site")
    pose = kine.fk(sim.data.qpos[sim.left.qpos_ids])
    assert pose.shape == (4, 4)
    np.testing.assert_array_equal(sim.data.qpos, before_qpos)
    np.testing.assert_array_equal(sim.data.qvel, before_qvel)


def test_left_ik_reaches_nearby_palm_pose_without_moving_right_state():
    sim = SimulationModel.load()
    kine = Kinematics(sim, sim.left, "left_palm_tcp_site")
    seed = sim.data.qpos[sim.left.qpos_ids].copy()
    target = kine.fk(seed)
    target[2, 3] += 0.02
    right_before = sim.data.qpos[sim.right.qpos_ids].copy()
    result = kine.ik(target, seed)
    assert result.success
    assert result.residual <= 1e-5
    np.testing.assert_array_equal(
        sim.data.qpos[sim.right.qpos_ids], right_before
    )


def test_default_kinematics_remains_right_arm():
    sim = SimulationModel.load()
    default = Kinematics(sim)
    explicit = Kinematics(sim, sim.right, "right_tool_tip_site")
    joints = sim.data.qpos[sim.right.qpos_ids]
    np.testing.assert_allclose(default.fk(joints), explicit.fk(joints))
