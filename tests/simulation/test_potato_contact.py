"""Physical contracts for the tactile potato task, using real MuJoCo contacts."""

import importlib
import importlib.util

import mujoco
import numpy as np
import pytest


def api():
    assert importlib.util.find_spec("twin_sim.potato_contact") is not None, "potato contact implementation is missing"
    return importlib.import_module("twin_sim.potato_contact")


def test_potato_falls_settles_and_moves_under_force():
    module = api()
    model = mujoco.MjModel.from_xml_string(module.build_contact_probe_xml())
    data = mujoco.MjData(model)
    potato = model.body("potato").id
    initial_z = data.qpos[2]
    for _ in range(1000):
        mujoco.mj_step(model, data)
    assert data.xpos[potato, 2] < initial_z - 0.01
    assert 0.015 < data.xpos[potato, 2] < 0.06
    assert np.linalg.norm(data.qvel) < 0.02
    start = data.xpos[potato].copy()
    for _ in range(100):
        data.xfrc_applied[potato, 0] = 4.0
        mujoco.mj_step(model, data)
    assert data.xpos[potato, 0] > start[0] + 0.005


def test_contact_readings_balance_weight_and_reverse_direction():
    module = api()
    model = mujoco.MjModel.from_xml_string(module.build_contact_probe_xml())
    data = mujoco.MjData(model)
    for _ in range(1000):
        mujoco.mj_step(model, data)
    # Refresh contact forces at the final state, not the previous integrator state.
    mujoco.mj_forward(model, data)
    potato = model.geom("potato_collision").id
    board = model.geom("probe_table").id
    on_potato = module.read_contact_patch(model, data, [potato])
    on_board = module.read_contact_patch(model, data, [board])
    assert on_potato.normal_force_n == pytest.approx(0.18 * 9.81, rel=0.03)
    assert on_potato.force_world_n[2] > 0
    np.testing.assert_allclose(on_potato.force_world_n, -on_board.force_world_n, atol=1e-8)
    assert on_potato.contact_count > 0


def test_no_contact_produces_zero_tactile_signal():
    module = api()
    model = mujoco.MjModel.from_xml_string(module.build_contact_probe_xml())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sample = module.read_contact_patch(model, data, [model.geom("potato_collision").id])
    assert sample.contact_count == 0
    assert sample.normal_force_n == 0
    np.testing.assert_array_equal(sample.force_world_n, np.zeros(3))
    np.testing.assert_array_equal(sample.center_world_m, np.zeros(3))


def test_combined_scene_keeps_robot_and_adds_free_potato():
    module = api()
    model = mujoco.MjModel.from_xml_string(module.build_robot_potato_xml())
    assert model.joint("potato_free").type == mujoco.mjtJoint.mjJNT_FREE
    assert model.geom("right_knife_blade").id >= 0
    assert model.geom("left_finger2_pad").id >= 0
    assert model.body("potato").mass == pytest.approx(0.18)


@pytest.mark.parametrize("size", [(0, .03, .03), (.05, -.02, .03), (np.nan, .03, .03)])
def test_invalid_potato_shape_is_rejected(size):
    with pytest.raises(ValueError, match="radii"):
        api().PotatoConfig(radii_m=size)


def test_flat_bottom_is_a_real_planar_collision_face():
    module=api()
    config=module.PotatoConfig(radii_m=(.04,.065,.03),bottom_cut_m=.003)
    original=module.potato_vertices(module.PotatoConfig(radii_m=config.radii_m))
    points=module.potato_vertices(config)
    floor=original[:,2].min()+.003
    assert points[:,2].min()==pytest.approx(floor,abs=1e-10)
    flat=points[np.abs(points[:,2]-floor)<1e-9]
    assert len(flat)>=16
    assert np.ptp(flat[:,0])>.02 and np.ptp(flat[:,1])>.03
    m=mujoco.MjModel.from_xml_string(module.build_contact_probe_xml(config))
    d=mujoco.MjData(m)
    for _ in range(1200):mujoco.mj_step(m,d)
    mujoco.mj_forward(m,d)
    # The free body settles upright on the new cap without a weld or state reset.
    body=m.body('potato').id
    assert d.xmat[body].reshape(3,3)[2,2]>.999
    assert np.linalg.norm(d.qvel)<.005
    mesh=int(m.geom_dataid[m.geom('potato_collision').id])
    verts=m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]]
    geom=m.geom('potato_collision').id
    world=verts@d.geom_xmat[geom].reshape(3,3).T+d.geom_xpos[geom]
    assert abs(world[:,2].min())<.001
    faces=m.mesh_face[m.mesh_faceadr[mesh]:m.mesh_faceadr[mesh]+m.mesh_facenum[mesh]]
    assert any(np.ptp(world[face,2])<1e-5 and np.max(world[face,2])<.001 for face in faces)


@pytest.mark.parametrize('cut',[-.001,np.nan,.03])
def test_invalid_flat_bottom_cut_is_rejected(cut):
    with pytest.raises(ValueError,match='bottom_cut_m'):
        api().PotatoConfig(radii_m=(.04,.065,.03),bottom_cut_m=cut)
