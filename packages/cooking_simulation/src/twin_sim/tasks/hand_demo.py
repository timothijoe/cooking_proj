from dataclasses import dataclass
import time

import numpy as np

from twin_sim.hand import DEFAULT_OPEN_RAD
from twin_sim.robot import RightArmRobot


RELAXED_CLOSE_RAD = np.asarray(
    (0.8, 0.3, 0.8, 0.8) + (0.8, 0.0, 0.8, 0.8) * 4,
    dtype=float,
)


@dataclass(frozen=True)
class HandDemoConfig:
    control_dt_s: float = 0.01
    close_duration_s: float = 2.0
    open_duration_s: float = 2.0
    viewer_start_hold_s: float = 0.0
    viewer_end_hold_s: float = 0.0


@dataclass(frozen=True)
class HandDemoResult:
    samples: int
    open_target_rad: np.ndarray
    final_target_rad: np.ndarray


def run_hand_demo(
    config: HandDemoConfig = HandDemoConfig(), *, viewer: bool = True
) -> HandDemoResult:
    robot = RightArmRobot(viewer=viewer)
    samples = 1
    try:
        _prepare_viewer(robot, config)
        for start, goal, duration in (
            (DEFAULT_OPEN_RAD, RELAXED_CLOSE_RAD, config.close_duration_s),
            (RELAXED_CLOSE_RAD, DEFAULT_OPEN_RAD, config.open_duration_s),
        ):
            steps = _step_count(duration, config.control_dt_s)
            for fraction in np.linspace(0.0, 1.0, steps + 1)[1:]:
                robot.hand.command(start + fraction * (goal - start))
                robot.step(config.control_dt_s)
                samples += 1
        _finish_viewer(robot, config)
        return HandDemoResult(
            samples=samples,
            open_target_rad=DEFAULT_OPEN_RAD.copy(),
            final_target_rad=robot.hand.target,
        )
    finally:
        robot.close()


def _prepare_viewer(robot: RightArmRobot, config: HandDemoConfig) -> None:
    viewer = robot._viewer
    if viewer is None or not viewer.is_running():
        return
    viewer.cam.azimuth = 140.0
    viewer.cam.elevation = -15.0
    viewer.cam.distance = 0.7
    viewer.cam.lookat[:] = (0.0, 0.86, 0.53)
    viewer.sync()
    if config.viewer_start_hold_s:
        time.sleep(config.viewer_start_hold_s)


def _finish_viewer(robot: RightArmRobot, config: HandDemoConfig) -> None:
    viewer = robot._viewer
    if (
        viewer is not None
        and viewer.is_running()
        and config.viewer_end_hold_s
    ):
        time.sleep(config.viewer_end_hold_s)


def _step_count(duration_s: float, control_dt_s: float) -> int:
    duration = float(duration_s)
    control_dt = float(control_dt_s)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("phase duration must be positive and finite")
    if not np.isfinite(control_dt) or control_dt <= 0.0:
        raise ValueError("control_dt_s must be positive and finite")
    steps = round(duration / control_dt)
    if steps < 1 or not np.isclose(steps * control_dt, duration):
        raise ValueError("phase duration must be an integer multiple of control_dt_s")
    return steps
