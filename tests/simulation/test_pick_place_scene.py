import mujoco
import numpy as np

from twin_sim.model import SimulationModel


def test_pick_place_scene_has_free_unattached_cube_and_named_sites():
    sim = SimulationModel.load()
    for site in ("left_palm_tcp_site", "pick_cube_site", "pick_target_site"):
        assert sim.require_site(site) >= 0
    joint = mujoco.mj_name2id(
        sim.model, mujoco.mjtObj.mjOBJ_JOINT, "pick_cube_free"
    )
    assert joint >= 0
    assert sim.model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE
    assert sim.model.neq == 0


def test_pick_cube_has_mass_collision_and_friction():
    sim = SimulationModel.load()
    body = sim.require_body("pick_cube")
    geom = sim.require_geom("pick_cube_geom")
    assert sim.model.body_mass[body] > 0.0
    assert sim.model.geom_contype[geom] != 0
    assert sim.model.geom_conaffinity[geom] != 0
    assert np.all(sim.model.geom_friction[geom] > 0.0)


def test_target_region_is_visual_only():
    sim = SimulationModel.load()
    geom = sim.require_geom("pick_target_region")
    assert sim.model.geom_contype[geom] == 0
    assert sim.model.geom_conaffinity[geom] == 0
