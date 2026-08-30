from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterable

import numpy as np

from twin_sim.force_monitor import ForceMonitor
from twin_sim.kinematics import Kinematics
from twin_sim.logging import CsvLogger, SimulationSample
from twin_sim.robot import RIGHT_HOME_RAD, RightArmRobot
from twin_sim.trajectory import (
    TrajectoryPoint,
    cartesian_trajectory,
    joint_trajectory,
)


CHOP_READY_RAD = np.array(
    (-0.87765, -1.62365, 0.90279, -1.27548, -1.40617, 0.65275, 0.36011),
    dtype=float,
)


@dataclass(frozen=True)
class ChopConfig:
    control_dt_s: float = 0.01
    orient_duration_s: float = 1.5
    approach_duration_s: float = 1.0
    descent_duration_s: float = 1.0
    hold_duration_s: float = 0.2
    retract_duration_s: float = 1.0
    penetration_m: float = 0.003
    force_filter_alpha: float = 0.2
    force_warning_threshold_n: float = 30.0
    safe_clearance_m: float = 0.08
    viewer_start_hold_s: float = 0.0
    viewer_end_hold_s: float = 0.0
    full_motion: bool = False


@dataclass(frozen=True)
class ChopResult:
    completed: bool
    samples: tuple[SimulationSample, ...]
    warning_count: int
    final_tip_z_m: float
    safe_tip_z_m: float


def run_chop(
    config: ChopConfig, *, log_path: Path, viewer: bool = False
) -> ChopResult:
    destination = Path(log_path)
    with CsvLogger(destination) as csv_logger:
        _validate_config(config)
        robot = RightArmRobot(viewer=viewer)
        try:
            kinematics = Kinematics(robot.sim)
            force_monitor = ForceMonitor(
                robot.sim,
                alpha=config.force_filter_alpha,
                warning_threshold_n=config.force_warning_threshold_n,
            )
            trajectories, safe_tip_z = _preflight(robot, kinematics, config)
            robot.validate_targets(
                [
                    point.joints_rad
                    for phase in ("ORIENT", "APPROACH", "DESCEND", "RETRACT")
                    for point in trajectories[phase]
                ]
            )
            if not config.full_motion:
                robot.reset(trajectories["APPROACH"][-1].joints_rad)
            _prepare_viewer_presentation(robot, config)
            samples: list[SimulationSample] = []
            warning_count = 0
            was_over_threshold = False

            def execute(phase: str, points: Iterable[TrajectoryPoint]) -> None:
                nonlocal warning_count, was_over_threshold
                for point in points:
                    robot.command(point.joints_rad)
                    robot.step(config.control_dt_s)
                    force = force_monitor.sample()
                    if force.over_threshold and not was_over_threshold:
                        warning_count += 1
                    was_over_threshold = force.over_threshold
                    sample = SimulationSample(
                        time_s=float(robot.sim.data.time),
                        phase=phase,
                        target_joints_rad=point.joints_rad.copy(),
                        actual_joints_rad=robot.joint_positions,
                        target_pose=(
                            point.target_pose.copy()
                            if point.target_pose is not None
                            else kinematics.fk(point.joints_rad)
                        ),
                        actual_pose=kinematics.fk(robot.joint_positions),
                        raw_force_n=force.raw_force_n,
                        filtered_force_n=force.filtered_force_n,
                        force_over_threshold=force.over_threshold,
                    )
                    samples.append(sample)
                    csv_logger.write(sample)

            if config.full_motion:
                execute("ORIENT", trajectories["ORIENT"][1:])
                execute("APPROACH", trajectories["APPROACH"][1:])
            else:
                execute("READY", [trajectories["APPROACH"][-1]])
            execute("DESCEND", trajectories["DESCEND"][1:])
            hold_count = int(round(config.hold_duration_s / config.control_dt_s))
            execute("HOLD", [trajectories["DESCEND"][-1]] * hold_count)
            execute("RETRACT", trajectories["RETRACT"][1:])
            final_pose = kinematics.fk(robot.joint_positions)
            final_force = force_monitor.sample()
            final_target = trajectories["RETRACT"][-1]
            complete = SimulationSample(
                time_s=float(robot.sim.data.time),
                phase="COMPLETE",
                target_joints_rad=final_target.joints_rad.copy(),
                actual_joints_rad=robot.joint_positions,
                target_pose=final_target.target_pose.copy(),
                actual_pose=final_pose,
                raw_force_n=final_force.raw_force_n,
                filtered_force_n=final_force.filtered_force_n,
                force_over_threshold=final_force.over_threshold,
            )
            samples.append(complete)
            csv_logger.write(complete)
            _finish_viewer_presentation(robot, config)
            return ChopResult(
                True,
                tuple(samples),
                warning_count,
                float(final_pose[2, 3]),
                safe_tip_z,
            )
        finally:
            robot.close()


def _preflight(
    robot: RightArmRobot, kinematics: Kinematics, config: ChopConfig
) -> tuple[dict[str, list[TrajectoryPoint]], float]:
    model, data = robot.sim.model, robot.sim.data
    board_id = robot.sim.require_geom("chopping_board")
    blade_id = robot.sim.require_site("right_blade_edge_bot")
    board_top = float(data.geom_xpos[board_id, 2] + model.geom_size[board_id, 2])
    with kinematics._configuration(CHOP_READY_RAD):
        initial_blade_z = float(data.site_xpos[blade_id, 2])
    initial_pose = kinematics.fk(CHOP_READY_RAD)
    safe_pose = initial_pose.copy()
    safe_pose[2, 3] += board_top + config.safe_clearance_m - initial_blade_z
    contact_pose = safe_pose.copy()
    contact_pose[2, 3] -= config.safe_clearance_m + config.penetration_m

    orient = joint_trajectory(
        RIGHT_HOME_RAD,
        CHOP_READY_RAD,
        config.orient_duration_s,
        config.control_dt_s,
    )
    approach = cartesian_trajectory(
        kinematics, initial_pose, safe_pose, CHOP_READY_RAD,
        config.approach_duration_s, config.control_dt_s,
    )
    descent = cartesian_trajectory(
        kinematics, safe_pose, contact_pose, approach[-1].joints_rad,
        config.descent_duration_s, config.control_dt_s,
    )
    retract = cartesian_trajectory(
        kinematics, contact_pose, safe_pose, descent[-1].joints_rad,
        config.retract_duration_s, config.control_dt_s,
    )
    return {
        "ORIENT": orient,
        "APPROACH": approach,
        "DESCEND": descent,
        "RETRACT": retract,
    }, float(safe_pose[2, 3])


def _validate_config(config: ChopConfig) -> None:
    positive = (
        config.control_dt_s,
        config.orient_duration_s,
        config.approach_duration_s,
        config.descent_duration_s,
        config.hold_duration_s,
        config.retract_duration_s,
        config.safe_clearance_m,
    )
    if not all(np.isfinite(value) and value > 0.0 for value in positive):
        raise ValueError("timing and safe clearance values must be positive and finite")
    if not np.isfinite(config.penetration_m) or config.penetration_m < 0.0:
        raise ValueError("penetration_m must be non-negative and finite")
    if config.hold_duration_s / config.control_dt_s != round(
        config.hold_duration_s / config.control_dt_s
    ):
        raise ValueError("hold_duration_s must be an integer multiple of control_dt_s")
    viewer_holds = (config.viewer_start_hold_s, config.viewer_end_hold_s)
    if not all(np.isfinite(value) and value >= 0.0 for value in viewer_holds):
        raise ValueError("viewer hold values must be non-negative and finite")


def _prepare_viewer_presentation(
    robot: RightArmRobot, config: ChopConfig
) -> None:
    viewer = robot._viewer
    if viewer is None or not viewer.is_running():
        return
    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -20.0
    viewer.cam.distance = 1.6
    viewer.cam.lookat[:] = (0.48, 0.0, 0.48)
    viewer.sync()
    if config.viewer_start_hold_s:
        time.sleep(config.viewer_start_hold_s)


def _finish_viewer_presentation(
    robot: RightArmRobot, config: ChopConfig
) -> None:
    viewer = robot._viewer
    if (
        viewer is not None
        and viewer.is_running()
        and config.viewer_end_hold_s
    ):
        time.sleep(config.viewer_end_hold_s)
