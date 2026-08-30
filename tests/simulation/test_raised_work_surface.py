import mujoco
import numpy as np
import pytest

from twin_sim.raised_work_surface import (
    apply_work_surface_offset,
    restore_work_surface,
    work_surface_height_m,
)
from twin_sim.robot import RightArmRobot


def _landmark_heights(robot):
    board = robot.sim.require_geom("chopping_board")
    cube_body = robot.sim.require_body("guarded_chop_cube_body")
    cube_site = robot.sim.require_site("guarded_chop_cube_center")
    return np.array(
        [
            robot.sim.data.geom_xpos[board, 2],
            robot.sim.data.xpos[cube_body, 2],
            robot.sim.data.site_xpos[cube_site, 2],
        ]
    )


def test_shared_offset_moves_board_object_and_site_together():
    robot = RightArmRobot(viewer=False)
    try:
        before = _landmark_heights(robot)
        left_base = robot.sim.require_body("left_link1")
        base_before = robot.sim.data.xpos[left_base].copy()

        snapshot = apply_work_surface_offset(robot.sim, .05)
        after = _landmark_heights(robot)

        np.testing.assert_allclose(after - before, .05)
        np.testing.assert_allclose(robot.sim.data.xpos[left_base], base_before)
        assert work_surface_height_m(robot.sim) == pytest.approx(before[0] + .05 + .025)
        restore_work_surface(robot.sim, snapshot)
        np.testing.assert_allclose(_landmark_heights(robot), before)
    finally:
        robot.close()


@pytest.mark.parametrize("offset", [-.01, np.inf, np.nan])
def test_shared_offset_rejects_invalid_values(offset):
    robot = RightArmRobot(viewer=False)
    try:
        with pytest.raises(ValueError, match="offset"):
            apply_work_surface_offset(robot.sim, offset)
    finally:
        robot.close()
