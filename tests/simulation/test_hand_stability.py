import mujoco
import numpy as np

from twin_sim.robot import RightArmRobot


def _descendants(model, root_name):
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)
    descendants = {root}
    changed = True
    while changed:
        changed = False
        for body_id, parent_id in enumerate(model.body_parentid):
            if int(parent_id) in descendants and body_id not in descendants:
                descendants.add(body_id)
                changed = True
    return descendants


def test_default_hand_pose_is_clear_and_stable():
    robot = RightArmRobot()
    try:
        model, data = robot.sim.model, robot.sim.data
        hand_bodies = _descendants(model, "left_hand_mount")
        board_geom = robot.sim.require_geom("chopping_board")

        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            bodies = {
                int(model.geom_bodyid[contact.geom1]),
                int(model.geom_bodyid[contact.geom2]),
            }
            assert not (
                board_geom in (contact.geom1, contact.geom2)
                and bodies.intersection(hand_bodies)
            )

        for _ in range(100):
            robot.step(0.01)

        assert np.isfinite(data.qpos[robot.sim.hand.qpos_ids]).all()
        assert np.isfinite(data.qvel[robot.sim.hand.dof_ids]).all()
        assert np.max(np.abs(data.qvel[robot.sim.hand.dof_ids])) < 20.0
    finally:
        robot.close()
