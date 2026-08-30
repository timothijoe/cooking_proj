from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from twin_sim.force_monitor import ForceMonitor
from twin_sim.kinematics import Kinematics
from twin_sim.logging import CsvLogger, SimulationSample
from twin_sim.robot import RightArmRobot
from twin_sim.tasks.chop import (
    CHOP_READY_RAD,
    ChopConfig,
    _finish_viewer_presentation,
    _preflight,
    _prepare_viewer_presentation,
    _validate_config,
)
from twin_sim.trajectory import TrajectoryPoint, cartesian_trajectory
from twin_sim.trajectory_plot import write_trajectory_svg
from twin_sim.viewer_trail import ViewerTrajectoryTrail


@dataclass(frozen=True)
class LineChopConfig:
    chop: ChopConfig = field(default_factory=ChopConfig)
    cuts: int = 5
    spacing_m: float = 0.03
    shift_duration_s: float = 1.0


@dataclass(frozen=True)
class LineChopResult:
    completed: bool
    samples: tuple[SimulationSample, ...]
    warning_count: int
    cut_points_xy: np.ndarray


@dataclass(frozen=True)
class _CutTrajectories:
    descent: tuple[TrajectoryPoint, ...]
    retract: tuple[TrajectoryPoint, ...]
    shift: tuple[TrajectoryPoint, ...]


@dataclass(frozen=True)
class _LinePreflight:
    initial_safe: TrajectoryPoint
    cuts: tuple[_CutTrajectories, ...]
    cut_points_xy: np.ndarray


def run_line_chop(
    config: LineChopConfig,
    *,
    log_path: Path,
    plot_path: Path,
    viewer: bool = False,
) -> LineChopResult:
    with CsvLogger(log_path) as csv_logger:
        _validate_line_config(config)
        robot = RightArmRobot(viewer=viewer)
        try:
            kinematics = Kinematics(robot.sim)
            preflight = _preflight_line_chop(robot, kinematics, config)
            targets = [
                point.joints_rad
                for cut in preflight.cuts
                for points in (cut.descent, cut.retract, cut.shift)
                for point in points
            ]
            robot.validate_targets(targets)
            robot.reset(preflight.initial_safe.joints_rad)
            trail = ViewerTrajectoryTrail(robot._viewer, sample_stride=10)
            trail.draw_target_plan(_target_trail_positions(preflight, config))
            _prepare_viewer_presentation(robot, config.chop)
            force_monitor = ForceMonitor(
                robot.sim,
                alpha=config.chop.force_filter_alpha,
                warning_threshold_n=config.chop.force_warning_threshold_n,
            )
            samples: list[SimulationSample] = []
            warning_count = 0
            was_over_threshold = False
            trail_step = 0

            def execute(
                phase: str,
                cut_index: int,
                points: Iterable[TrajectoryPoint],
            ) -> None:
                nonlocal warning_count, was_over_threshold, trail_step
                for point in points:
                    robot.command(point.joints_rad)
                    robot.step(config.chop.control_dt_s)
                    force = force_monitor.sample()
                    if force.over_threshold and not was_over_threshold:
                        warning_count += 1
                    was_over_threshold = force.over_threshold
                    actual_pose = kinematics.fk(robot.joint_positions)
                    sample = SimulationSample(
                        time_s=float(robot.sim.data.time),
                        phase=phase,
                        target_joints_rad=point.joints_rad.copy(),
                        actual_joints_rad=robot.joint_positions,
                        target_pose=point.target_pose.copy(),
                        actual_pose=actual_pose,
                        raw_force_n=force.raw_force_n,
                        filtered_force_n=force.filtered_force_n,
                        force_over_threshold=force.over_threshold,
                        cut_index=cut_index,
                    )
                    samples.append(sample)
                    csv_logger.write(sample)
                    trail.record_actual(actual_pose[:3, 3], trail_step)
                    trail_step += 1

            execute("READY", 0, (preflight.initial_safe,))
            hold_count = int(
                round(config.chop.hold_duration_s / config.chop.control_dt_s)
            )
            for index, cut in enumerate(preflight.cuts, start=1):
                execute("DESCEND", index, cut.descent[1:])
                execute("HOLD", index, (cut.descent[-1],) * hold_count)
                execute("RETRACT", index, cut.retract[1:])
                if cut.shift:
                    execute("SHIFT", index, cut.shift[1:])

            final_target = preflight.cuts[-1].retract[-1]
            final_force = force_monitor.sample()
            complete = SimulationSample(
                time_s=float(robot.sim.data.time),
                phase="COMPLETE",
                target_joints_rad=final_target.joints_rad.copy(),
                actual_joints_rad=robot.joint_positions,
                target_pose=final_target.target_pose.copy(),
                actual_pose=kinematics.fk(robot.joint_positions),
                raw_force_n=final_force.raw_force_n,
                filtered_force_n=final_force.filtered_force_n,
                force_over_threshold=final_force.over_threshold,
                cut_index=config.cuts,
            )
            samples.append(complete)
            csv_logger.write(complete)
            write_trajectory_svg(plot_path, samples)
            _finish_viewer_presentation(robot, config.chop)
            return LineChopResult(
                completed=True,
                samples=tuple(samples),
                warning_count=warning_count,
                cut_points_xy=preflight.cut_points_xy.copy(),
            )
        finally:
            robot.close()


def _preflight_line_chop(
    robot: RightArmRobot,
    kinematics: Kinematics,
    config: LineChopConfig,
) -> _LinePreflight:
    base, _ = _preflight(robot, kinematics, config.chop)
    initial_safe = base["APPROACH"][-1]
    data = robot.sim.data
    top_id = robot.sim.require_site("right_blade_edge_top")
    tip_id = robot.sim.require_site("right_tool_tip_site")
    marker_ids = (
        top_id,
        robot.sim.require_site("right_blade_edge_bot"),
        tip_id,
    )
    with kinematics._configuration(CHOP_READY_RAD):
        marker_xy = data.site_xpos[np.asarray(marker_ids), :2].copy()
    blade_normal_xy = initial_safe.target_pose[:2, 0]
    normal_norm = float(np.linalg.norm(blade_normal_xy))
    if not np.isfinite(normal_norm) or normal_norm <= 1e-9:
        raise ValueError("blade normal has no finite XY projection")
    direction_xy = -blade_normal_xy / normal_norm
    offsets = (
        np.arange(config.cuts, dtype=float)[:, None]
        * config.spacing_m
        * direction_xy
    )
    cut_points_xy = marker_xy[1] + offsets
    _validate_board_bounds(robot, marker_xy, offsets)

    safe_pose = initial_safe.target_pose.copy()
    current_safe = initial_safe
    cut_trajectories: list[_CutTrajectories] = []
    for index in range(config.cuts):
        if index:
            safe_pose = initial_safe.target_pose.copy()
            safe_pose[:2, 3] += offsets[index]
        contact_pose = safe_pose.copy()
        contact_pose[2, 3] -= (
            config.chop.safe_clearance_m + config.chop.penetration_m
        )
        descent = cartesian_trajectory(
            kinematics,
            safe_pose,
            contact_pose,
            current_safe.joints_rad,
            config.chop.descent_duration_s,
            config.chop.control_dt_s,
        )
        retract = cartesian_trajectory(
            kinematics,
            contact_pose,
            safe_pose,
            descent[-1].joints_rad,
            config.chop.retract_duration_s,
            config.chop.control_dt_s,
        )
        shift: list[TrajectoryPoint] = []
        current_safe = retract[-1]
        if index + 1 < config.cuts:
            next_safe = initial_safe.target_pose.copy()
            next_safe[:2, 3] += offsets[index + 1]
            shift = cartesian_trajectory(
                kinematics,
                safe_pose,
                next_safe,
                retract[-1].joints_rad,
                config.shift_duration_s,
                config.chop.control_dt_s,
            )
            current_safe = shift[-1]
        cut_trajectories.append(
            _CutTrajectories(
                tuple(descent),
                tuple(retract),
                tuple(shift),
            )
        )
    return _LinePreflight(
        initial_safe,
        tuple(cut_trajectories),
        cut_points_xy,
    )


def _validate_line_config(config: LineChopConfig) -> None:
    _validate_config(config.chop)
    if (
        isinstance(config.cuts, bool)
        or not isinstance(config.cuts, (int, np.integer))
        or config.cuts <= 0
    ):
        raise ValueError("cuts must be a positive integer")
    values = (config.spacing_m, config.shift_duration_s)
    if not all(np.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("spacing_m and shift_duration_s must be positive and finite")
    ratio = config.shift_duration_s / config.chop.control_dt_s
    if ratio != round(ratio):
        raise ValueError(
            "shift_duration_s must be an integer multiple of control_dt_s"
        )


def _validate_board_bounds(
    robot: RightArmRobot,
    marker_xy: np.ndarray,
    offsets: np.ndarray,
) -> None:
    board_id = robot.sim.require_geom("chopping_board")
    center = robot.sim.data.geom_xpos[board_id, :2]
    half_size = robot.sim.model.geom_size[board_id, :2]
    points = marker_xy[None, :, :] + offsets[:, None, :]
    if np.any(points < center - half_size) or np.any(points > center + half_size):
        raise ValueError("line-chop path leaves chopping board bounds")


def _target_trail_positions(
    preflight: _LinePreflight,
    config: LineChopConfig,
) -> list[np.ndarray]:
    positions = [preflight.initial_safe.target_pose[:3, 3].copy()]
    hold_count = int(
        round(config.chop.hold_duration_s / config.chop.control_dt_s)
    )
    for cut in preflight.cuts:
        positions.extend(point.target_pose[:3, 3].copy() for point in cut.descent[1:])
        positions.extend(
            cut.descent[-1].target_pose[:3, 3].copy()
            for _ in range(hold_count)
        )
        positions.extend(point.target_pose[:3, 3].copy() for point in cut.retract[1:])
        positions.extend(point.target_pose[:3, 3].copy() for point in cut.shift[1:])
    return positions
