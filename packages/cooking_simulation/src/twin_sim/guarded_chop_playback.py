from pathlib import Path

import mujoco
import numpy as np

from twin_sim.guarded_chop_recording import (
    load_recording,
    replay_recording,
    validate_replay_rate,
)
from twin_sim.guarded_chop_visualization import GuardedChopTrace
from twin_sim.robot import RightArmRobot
from twin_sim.tasks.guarded_chop import (
    GuardedChopConfig,
    _blade_center_for_joints,
    _configure_guarded_scene,
    _preflight_guarded_chop,
    _prepare_guarded_chop_viewer,
)


def play_guarded_chop_recording(
    path: Path, *, rate: float = 2.0
) -> None:
    playback_rate = validate_replay_rate(rate)
    robot = RightArmRobot(viewer=False)
    try:
        config = GuardedChopConfig(final_hold_s=0.0)
        plan = _preflight_guarded_chop(robot, config)
        _configure_guarded_scene(robot, "plane")
        recording = load_recording(Path(path), robot.sim.model)
        first = recording.frames[0]
        robot.sim.data.time = first.time_s
        robot.sim.data.qpos[:] = first.qpos
        robot.sim.data.qvel[:] = first.qvel
        robot.sim.data.ctrl[:] = first.ctrl
        mujoco.mj_forward(robot.sim.model, robot.sim.data)
        robot.open_viewer()
        _prepare_guarded_chop_viewer(robot)
        trace = GuardedChopTrace(robot._viewer)
        heights = np.asarray(
            [
                _blade_center_for_joints(
                    robot, cut.descent[-1].joints_rad
                )[2]
                for cut in plan.cuts
            ]
        )
        trace.set_plan(np.column_stack((plan.cut_points_xy, heights)))
        replay_recording(
            robot,
            recording,
            rate=playback_rate,
            trace=trace,
        )
    finally:
        robot.close()
