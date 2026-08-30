import mujoco
import numpy as np

from twin_sim.model import SimulationModel
from twin_sim.robot import RightArmRobot
from twin_sim.tasks.guarded_chop import (
    _activate_guarded_scene,
    _geom_belongs_to_body_tree,
)


def test_guarded_chop_proxy_is_fixed_and_hidden_by_default():
    sim = SimulationModel.load()
    geom = sim.require_geom("guarded_chop_cube")
    body = int(sim.model.geom_bodyid[geom])

    assert sim.model.body_jntnum[body] == 0
    assert sim.model.geom_rgba[geom, 3] == 0.0
    assert sim.model.geom_contype[geom] == 0
    assert sim.model.geom_conaffinity[geom] == 0
    np.testing.assert_allclose(
        sim.model.geom_pos[geom, :2], (0.0, 0.0), atol=1e-12
    )
    np.testing.assert_allclose(
        sim.model.body_pos[body, :2], (0.62, -0.04), atol=1e-12
    )
    np.testing.assert_allclose(
        sim.model.geom_size[geom, :2], (0.08, 0.08), atol=1e-12
    )


def test_guard_knuckle_site_belongs_to_left_hand():
    sim = SimulationModel.load()
    site = sim.require_site("left_guard_knuckle_site")
    body = int(sim.model.site_bodyid[site])
    name = mujoco.mj_id2name(
        sim.model, mujoco.mjtObj.mjOBJ_BODY, body
    )

    assert name == "left_finger3_link3"
    assert np.isfinite(sim.data.site_xpos[site]).all()


def test_cube_collision_is_paired_only_with_middle_finger_pad():
    robot = RightArmRobot(viewer=False)
    try:
        _activate_guarded_scene(robot)
        model = robot.sim.model
        cube = robot.sim.require_geom("guarded_chop_cube")
        palm = robot.sim.require_body("left_palm_link")
        partners = set()
        for geom in range(model.ngeom):
            if not _geom_belongs_to_body_tree(robot, geom, palm):
                continue
            compatible = (
                model.geom_contype[cube] & model.geom_conaffinity[geom]
            ) or (
                model.geom_contype[geom] & model.geom_conaffinity[cube]
            )
            if compatible:
                partners.add(
                    mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, geom
                    )
                )

        assert partners == {"left_finger3_pad"}
    finally:
        robot.close()


def test_blade_collision_is_paired_with_every_named_hand_proxy():
    robot = RightArmRobot(viewer=False)
    try:
        _activate_guarded_scene(robot)
        model = robot.sim.model
        blade = robot.sim.require_geom("right_knife_blade")
        palm = robot.sim.require_body("left_palm_link")
        expected = set()
        partners = set()
        for geom in range(model.ngeom):
            name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, geom
            )
            if (
                name is None
                or not _geom_belongs_to_body_tree(robot, geom, palm)
                or model.geom_contype[geom] == 0
            ):
                continue
            expected.add(name)
            compatible = (
                model.geom_contype[blade] & model.geom_conaffinity[geom]
            ) or (
                model.geom_contype[geom] & model.geom_conaffinity[blade]
            )
            if compatible:
                partners.add(name)

        assert partners == expected
    finally:
        robot.close()
