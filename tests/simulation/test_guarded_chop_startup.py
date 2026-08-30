import numpy as np
import pytest

import twin_sim.tasks.guarded_chop as guarded_chop
from twin_sim.guarded_chop_safety import CAT_PAW_RAD
from twin_sim.robot import RightArmRobot
from twin_sim.tasks.guarded_chop import GuardedChopConfig, run_guarded_chop


class _Trace:
    def __init__(self, viewer, events):
        events.append("create_trace")

    def set_plan(self, points):
        pass

    def append(self, **sample):
        pass

    def set_abort(self, reason):
        pass


def test_viewer_opens_only_after_guarded_scene_and_ready_pose(monkeypatch):
    events = []
    constructor_viewer_values = []
    original_configure = guarded_chop._configure_guarded_scene
    original_preflight = guarded_chop._preflight_guarded_chop
    original_robot = RightArmRobot
    original_forward = guarded_chop.mujoco.mj_forward
    robot_holder = []
    plan_holder = []
    configured = False

    def robot_factory(*args, **kwargs):
        constructor_viewer_values.append(kwargs.get("viewer", False))
        robot = original_robot(*args, **kwargs)
        robot_holder.append(robot)
        return robot

    def configure(robot, scene_mode):
        nonlocal configured
        events.append("configure")
        result = original_configure(robot, scene_mode)
        configured = True
        return result

    def preflight(robot, config):
        plan = original_preflight(robot, config)
        plan_holder.append(plan)
        return plan

    def forward(model, data):
        result = original_forward(model, data)
        if configured and robot_holder and plan_holder:
            robot = robot_holder[0]
            plan = plan_holder[0]
            ready = (
                np.allclose(
                    data.qpos[robot.sim.right.qpos_ids],
                    plan.right_ready_rad,
                )
                and np.allclose(
                    data.qpos[robot.sim.left.qpos_ids],
                    plan.left_ready_rad,
                )
                and np.allclose(
                    data.ctrl[robot.sim.right.actuator_ids],
                    plan.right_ready_rad,
                )
                and np.allclose(
                    data.ctrl[robot.sim.left.actuator_ids],
                    plan.left_ready_rad,
                )
                and np.allclose(
                    data.qpos[robot.sim.hand.qpos_ids], CAT_PAW_RAD
                )
                and np.allclose(
                    data.ctrl[robot.sim.hand.actuator_ids], CAT_PAW_RAD
                )
            )
            if ready and "final_forward" not in events:
                events.append("final_forward")
        return result

    def open_viewer(robot):
        plan = plan_holder[0]
        for name in (
            "pick_source_pedestal",
            "pick_target_pedestal",
            "pick_cube_geom",
            "pick_target_region",
        ):
            geom = robot.sim.require_geom(name)
            assert robot.sim.model.geom_rgba[geom, 3] == 0.0
        np.testing.assert_allclose(
            robot.sim.data.qpos[robot.sim.hand.qpos_ids],
            CAT_PAW_RAD,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            robot.sim.data.qpos[robot.sim.right.qpos_ids],
            plan.right_ready_rad,
        )
        np.testing.assert_allclose(
            robot.sim.data.qpos[robot.sim.left.qpos_ids],
            plan.left_ready_rad,
        )
        np.testing.assert_allclose(
            robot.sim.data.ctrl[robot.sim.right.actuator_ids],
            plan.right_ready_rad,
        )
        np.testing.assert_allclose(
            robot.sim.data.ctrl[robot.sim.left.actuator_ids],
            plan.left_ready_rad,
        )
        events.append("open_viewer")

    def prepare_camera(robot):
        events.append("prepare_camera")

    monkeypatch.setattr(guarded_chop, "RightArmRobot", robot_factory)
    monkeypatch.setattr(
        guarded_chop, "_configure_guarded_scene", configure
    )
    monkeypatch.setattr(
        guarded_chop, "_preflight_guarded_chop", preflight
    )
    monkeypatch.setattr(RightArmRobot, "open_viewer", open_viewer)
    monkeypatch.setattr(guarded_chop.mujoco, "mj_forward", forward)
    monkeypatch.setattr(
        guarded_chop, "_prepare_guarded_chop_viewer", prepare_camera
    )
    monkeypatch.setattr(
        "twin_sim.guarded_chop_visualization.GuardedChopTrace",
        lambda viewer: _Trace(viewer, events),
    )

    result = run_guarded_chop(
        GuardedChopConfig(final_hold_s=0.0), viewer=True
    )

    assert result.success, result.reason
    assert constructor_viewer_values == [False]
    assert events.index("configure") < events.index("open_viewer")
    assert events.index("final_forward") < events.index("open_viewer")
    assert events.index("open_viewer") < events.index("prepare_camera")
    assert events.index("prepare_camera") < events.index("create_trace")


def test_viewer_startup_error_propagates_after_robot_cleanup(monkeypatch):
    closed = []
    original_close = RightArmRobot.close

    def fail_open(robot):
        raise RuntimeError("viewer launch failed")

    def record_close(robot):
        closed.append(True)
        original_close(robot)

    monkeypatch.setattr(RightArmRobot, "open_viewer", fail_open)
    monkeypatch.setattr(RightArmRobot, "close", record_close)

    with pytest.raises(RuntimeError, match="viewer launch failed"):
        run_guarded_chop(
            GuardedChopConfig(final_hold_s=0.0), viewer=True
        )

    assert closed == [True]
