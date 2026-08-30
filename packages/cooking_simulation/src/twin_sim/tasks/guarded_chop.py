from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, Sequence

import mujoco
import numpy as np

from twin_sim import guarded_chop_recording
from twin_sim.guarded_chop_safety import (
    CAT_PAW_RAD,
    GUARD_RELAXED_RAD,
    GUARD_RETRACTED_RAD,
    SafetyCoordinator,
    SafetyObservation,
)
from twin_sim.kinematics import PathIkError
from twin_sim.robot import RightArmRobot
from twin_sim.tasks.chop import ChopConfig
from twin_sim.tasks.line_chop import (
    LineChopConfig,
    _CutTrajectories,
    _preflight_line_chop,
)
from twin_sim.tasks.pick_place import LEFT_GRASP_READY_RAD
from twin_sim.trajectory import (
    TrajectoryPoint,
    _validate_motion,
    cartesian_trajectory,
    minimum_jerk,
)


_GUARDED_CHOP_VIEW_AZIMUTH_DEG = 135.0
_GUARDED_CHOP_VIEW_ELEVATION_DEG = -20.0
_GUARDED_CHOP_VIEW_DISTANCE_M = 1.6
_GUARDED_CHOP_VIEW_LOOKAT = (0.48, 0.0, 0.48)
_VIEWER_REFRESH_PERIOD_S = 0.03
_KNIFE_SAFE_TRACKING_HEADROOM_M = 0.006
_KNIFE_CLEAR_TARGET_HEADROOM_M = 0.005


def _viewer_sync_stride(control_dt_s: float) -> int:
    return max(
        1,
        int(round(_VIEWER_REFRESH_PERIOD_S / control_dt_s)),
    )


class GuardedChopPhase(str, Enum):
    INITIALIZE = "initialize"
    GUARD_READY = "guard_ready"
    CUT_DOWN = "cut_down"
    LOW_GUARD_OPEN = "low_guard_open"
    LOW_GUARD_SHIFT = "low_guard_shift"
    LOW_GUARD_CLOSE = "low_guard_close"
    GUARD_SETTLE = "guard_settle"
    FINGER_RETRACT = "finger_retract"
    ARM_RESET_RELAX = "arm_reset_relax"
    KNIFE_LIFT_SHIFT = "knife_lift_shift"
    KNIFE_CLEAR = "knife_clear"
    COMPLETE = "complete"
    ABORTED = "aborted"


@dataclass(frozen=True)
class GuardedChopConfig:
    scene_mode: Literal["plane", "object"] = "plane"
    cuts: int = 5
    hand_shift_m: float = 0.02
    minimum_distance_m: float = 0.02
    maximum_guard_penetration_m: float = 0.003
    maximum_guard_force_n: float = 35.0
    control_dt_s: float = 0.01
    guard_ready_duration_s: float = 1.2
    cut_duration_s: float = 0.7
    knife_up_duration_s: float = 0.7
    hand_open_duration_s: float = 0.4
    hand_shift_duration_s: float = 2.0
    hand_close_duration_s: float = 0.4
    finger_motion_duration_s: float = 2.5
    interlock_settle_s: float = 0.3
    stability_timeout_s: float = 4.0
    final_hold_s: float = 3.0

    def validated(self) -> "GuardedChopConfig":
        if self.scene_mode not in {"plane", "object"}:
            raise ValueError("scene_mode must be 'plane' or 'object'")
        if (
            isinstance(self.cuts, bool)
            or not isinstance(self.cuts, (int, np.integer))
            or self.cuts != 5
        ):
            raise ValueError("cuts must equal 5 for guarded chopping")
        if not np.isclose(self.hand_shift_m, 0.02, rtol=0.0, atol=1e-12):
            raise ValueError("hand_shift_m must equal 0.02")
        if not np.isclose(
            self.minimum_distance_m, 0.02, rtol=0.0, atol=1e-12
        ):
            raise ValueError("minimum_distance_m must equal 0.02")
        if (
            not np.isfinite(self.maximum_guard_penetration_m)
            or self.maximum_guard_penetration_m <= 0.0
            or not np.isfinite(self.maximum_guard_force_n)
            or self.maximum_guard_force_n <= 0.0
        ):
            raise ValueError("guard contact limits must be positive and finite")
        durations = (
            self.control_dt_s,
            self.guard_ready_duration_s,
            self.cut_duration_s,
            self.knife_up_duration_s,
            self.hand_open_duration_s,
            self.hand_shift_duration_s,
            self.hand_close_duration_s,
            self.finger_motion_duration_s,
            self.interlock_settle_s,
            self.stability_timeout_s,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in durations):
            raise ValueError("control durations must be positive and finite")
        if not np.isfinite(self.final_hold_s) or self.final_hold_s < 0.0:
            raise ValueError("final_hold_s must be non-negative and finite")
        for duration in durations[1:] + (self.final_hold_s,):
            ratio = duration / self.control_dt_s
            if not np.isclose(ratio, round(ratio), rtol=0.0, atol=1e-9):
                raise ValueError(
                    "durations must be integer multiples of control_dt_s"
                )
        return self


@dataclass(frozen=True)
class GuardedChopSample:
    time_s: float
    phase: GuardedChopPhase
    cut_index: int
    left_target_rad: np.ndarray
    left_actual_rad: np.ndarray
    right_target_rad: np.ndarray
    right_actual_rad: np.ndarray
    hand_target_rad: np.ndarray
    knife_position: np.ndarray
    guard_position: np.ndarray
    knife_guard_distance_m: float
    knife_height_m: float
    cut_allowed: bool
    safety_allowed: bool
    guard_cube_contact_count: int
    guard_cube_penetration_m: float
    guard_cube_normal_force_n: float
    blade_hand_contact_count: int
    left_speed_rad_s: float
    right_speed_rad_s: float
    hand_speed_rad_s: float

    def __post_init__(self) -> None:
        for name in (
            "left_target_rad",
            "left_actual_rad",
            "right_target_rad",
            "right_actual_rad",
            "hand_target_rad",
            "knife_position",
            "guard_position",
        ):
            object.__setattr__(
                self, name, np.asarray(getattr(self, name), dtype=float).copy()
            )


@dataclass(frozen=True)
class GuardedChopResult:
    success: bool
    final_phase: GuardedChopPhase
    reason: str
    samples: tuple[GuardedChopSample, ...]
    completed_cuts: int
    completed_shifts: int
    total_shift_m: float
    minimum_distance_m: float


@dataclass(frozen=True)
class _GuardedChopPlan:
    right_ready_rad: np.ndarray
    left_ready_rad: np.ndarray
    cuts: tuple[_CutTrajectories, ...]
    knife_lift_shifts: tuple[tuple[TrajectoryPoint, ...], ...]
    cut_points_xy: np.ndarray
    guard_shifts: tuple[tuple[TrajectoryPoint, ...], ...]
    guard_targets: np.ndarray
    plane_guard_motions: tuple["_PlaneGuardMotion", ...]
    minimum_planned_distances_m: np.ndarray
    safe_knife_height_m: float


@dataclass(frozen=True)
class _PlaneGuardMotion:
    phase: GuardedChopPhase
    left: tuple[np.ndarray, ...]
    hand: tuple[np.ndarray, ...]


def _point_to_box_distance(
    point: Sequence[float],
    center: Sequence[float],
    rotation: np.ndarray,
    half_size: Sequence[float],
) -> float:
    point_array = np.asarray(point, dtype=float)
    center_array = np.asarray(center, dtype=float)
    rotation_array = np.asarray(rotation, dtype=float)
    size_array = np.asarray(half_size, dtype=float)
    if (
        point_array.shape != (3,)
        or center_array.shape != (3,)
        or rotation_array.shape != (3, 3)
        or size_array.shape != (3,)
        or not all(
            np.isfinite(value).all()
            for value in (
                point_array,
                center_array,
                rotation_array,
                size_array,
            )
        )
        or np.any(size_array < 0.0)
    ):
        raise ValueError("box distance inputs must be finite and correctly shaped")
    local = rotation_array.T @ (point_array - center_array)
    closest = np.clip(local, -size_array, size_array)
    return float(np.linalg.norm(local - closest))


def _joint_trajectory(
    start_rad: Sequence[float],
    goal_rad: Sequence[float],
    duration_s: float,
    dt_s: float,
) -> tuple[np.ndarray, ...]:
    steps = int(round(duration_s / dt_s))
    blend = np.linspace(0.0, 1.0, steps + 1)
    smooth = blend * blend * (3.0 - 2.0 * blend)
    start = np.asarray(start_rad, dtype=float)
    goal = np.asarray(goal_rad, dtype=float)
    return tuple(start + value * (goal - start) for value in smooth)


def _resample_trajectory(
    points: Sequence[np.ndarray], count: int
) -> tuple[np.ndarray, ...]:
    if count <= 0:
        raise ValueError("count must be positive")
    values = tuple(np.asarray(point, dtype=float) for point in points)
    if not values:
        raise ValueError("trajectory must not be empty")
    if len(values) == 1:
        return tuple(values[0].copy() for _ in range(count))
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, count)
    stacked = np.asarray(values)
    return tuple(
        np.asarray(
            [
                np.interp(progress, source, stacked[:, axis])
                for axis in range(stacked.shape[1])
            ]
        )
        for progress in target
    )


def _preflight_guarded_chop(
    robot: RightArmRobot,
    config: GuardedChopConfig,
) -> _GuardedChopPlan:
    config = config.validated()
    robot.sim.require_geom("guarded_chop_cube")
    robot.sim.require_site("guarded_chop_cube_center")
    robot.sim.require_site("left_guard_knuckle_site")
    robot.sim.require_site("left_palm_tcp_site")
    saved = mujoco.MjData(robot.sim.model)
    mujoco.mj_copyData(saved, robot.sim.model, robot.sim.data)
    try:
        line = _preflight_line_chop(
            robot,
            robot.right_kinematics,
            LineChopConfig(
                chop=ChopConfig(
                    control_dt_s=config.control_dt_s,
                    descent_duration_s=config.cut_duration_s,
                    retract_duration_s=config.knife_up_duration_s,
                ),
                cuts=config.cuts,
                spacing_m=config.hand_shift_m,
                shift_duration_s=config.hand_shift_duration_s,
            ),
        )
        board = robot.sim.require_geom("chopping_board")
        cube = robot.sim.require_geom("guarded_chop_cube")
        board_top = float(
            robot.sim.data.geom_xpos[board, 2]
            + robot.sim.model.geom_size[board, 2]
        )
        cube_top = float(
            robot.sim.data.geom_xpos[cube, 2]
            + robot.sim.model.geom_size[cube, 2]
        )
        ordered_indices = _right_to_left_indices(line.cut_points_xy)
        cut_points_xy = line.cut_points_xy[ordered_indices].copy()
        ordered_line_cuts = tuple(line.cuts[index] for index in ordered_indices)
        cuts = _translate_right_cuts(
            robot,
            ordered_line_cuts,
            cube_top - board_top + 0.008,
            config,
        )
        knife_lift_shifts = _diagonal_lift_shifts(robot, cuts, config)
        left_ready, guard_targets, guard_shifts = _guard_plan(
            robot,
            cut_points_xy,
            config,
            guard_offset_direction=line.cut_points_xy[1]
            - line.cut_points_xy[0],
        )
        plane_guard_motions = (
            _plane_guard_motions(
                robot,
                left_ready,
                guard_targets,
                config,
            )
            if config.scene_mode == "plane"
            else ()
        )
        distances = _planned_clearances(
            robot,
            cuts,
            knife_lift_shifts,
            guard_targets,
            guard_shifts,
            plane_guard_motions,
            left_ready,
            config,
        )
        if np.any(distances < config.minimum_distance_m):
            raise ValueError(
                "guarded-chop plan violates knife-guard clearance: "
                f"{float(np.min(distances)):.6f} m"
            )
        right_ready = cuts[0].descent[0].joints_rad
        safe_height = (
            _blade_bottom_height(robot, right_ready)
            - _KNIFE_SAFE_TRACKING_HEADROOM_M
        )
        return _GuardedChopPlan(
            right_ready_rad=right_ready.copy(),
            left_ready_rad=left_ready.copy(),
            cuts=cuts,
            knife_lift_shifts=knife_lift_shifts,
            cut_points_xy=cut_points_xy,
            guard_shifts=guard_shifts,
            guard_targets=guard_targets,
            plane_guard_motions=plane_guard_motions,
            minimum_planned_distances_m=distances,
            safe_knife_height_m=safe_height,
        )
    finally:
        mujoco.mj_copyData(robot.sim.data, robot.sim.model, saved)


def _translate_right_cuts(
    robot: RightArmRobot,
    cuts: tuple[_CutTrajectories, ...],
    dz_m: float,
    config: GuardedChopConfig,
) -> tuple[_CutTrajectories, ...]:
    translated = []
    seed = cuts[0].descent[0].joints_rad
    for index, cut in enumerate(cuts):
        safe = cut.descent[0].target_pose.copy()
        contact = cut.descent[-1].target_pose.copy()
        safe[2, 3] += dz_m + _KNIFE_SAFE_TRACKING_HEADROOM_M
        contact[2, 3] += dz_m
        safe_result = robot.right_kinematics.ik(
            safe, seed, max_iterations=1000, damping=0.003
        )
        if not safe_result.success:
            raise PathIkError(
                "raised knife-safe IK failed "
                f"(residual {safe_result.residual:.6g})"
            )
        descent = cartesian_trajectory(
            robot.right_kinematics,
            safe,
            contact,
            safe_result.joints_rad,
            config.cut_duration_s,
            config.control_dt_s,
        )
        retract = cartesian_trajectory(
            robot.right_kinematics,
            contact,
            safe,
            descent[-1].joints_rad,
            config.knife_up_duration_s,
            config.control_dt_s,
        )
        shift = ()
        seed = retract[-1].joints_rad
        if index + 1 < len(cuts):
            next_safe = cuts[index + 1].descent[0].target_pose.copy()
            next_safe[2, 3] += (
                dz_m + _KNIFE_SAFE_TRACKING_HEADROOM_M
            )
            shift = tuple(
                cartesian_trajectory(
                    robot.right_kinematics,
                    safe,
                    next_safe,
                    seed,
                    config.hand_shift_duration_s,
                    config.control_dt_s,
                )
            )
            seed = shift[-1].joints_rad
        translated.append(
            _CutTrajectories(tuple(descent), tuple(retract), shift)
        )
    return tuple(translated)


def _diagonal_lift_shifts(
    robot: RightArmRobot,
    cuts: tuple[_CutTrajectories, ...],
    config: GuardedChopConfig,
) -> tuple[tuple[TrajectoryPoint, ...], ...]:
    duration_s = config.knife_up_duration_s + config.hand_shift_duration_s
    paths = []
    for current, following in zip(cuts[:-1], cuts[1:], strict=True):
        path = cartesian_trajectory(
            robot.right_kinematics,
            current.descent[-1].target_pose,
            following.descent[0].target_pose,
            current.descent[-1].joints_rad,
            duration_s,
            config.control_dt_s,
        )
        paths.append(
            _retarget_trajectory_endpoint(
                path,
                following.descent[0].joints_rad,
                config.control_dt_s,
            )
        )
    return tuple(paths)


def _retarget_trajectory_endpoint(
    path: Sequence[TrajectoryPoint],
    goal_joints_rad: Sequence[float],
    control_dt_s: float,
) -> tuple[TrajectoryPoint, ...]:
    points = tuple(path)
    if len(points) < 2:
        raise ValueError("trajectory must contain at least two points")
    joints = np.asarray([point.joints_rad for point in points])
    goal = np.asarray(goal_joints_rad, dtype=float)
    if goal.shape != joints.shape[1:] or not np.isfinite(goal).all():
        raise ValueError("goal joints must match trajectory shape")
    phase = np.linspace(0.0, 1.0, len(points))
    blend, _ = minimum_jerk(phase)
    joints = joints + blend[:, None] * (goal - joints[-1])
    joints[-1] = goal
    _validate_motion(joints, control_dt_s)
    velocities = np.gradient(joints, control_dt_s, axis=0, edge_order=2)
    velocities[[0, -1]] = 0.0
    return tuple(
        TrajectoryPoint(
            point.time_s,
            joint.copy(),
            velocity.copy(),
            point.target_pose.copy(),
        )
        for point, joint, velocity in zip(
            points, joints, velocities, strict=True
        )
    )


def _validate_nonapproaching_retreat(
    distances_m: Sequence[float], *, tolerance_m: float = 1e-6
) -> None:
    values = np.asarray(distances_m, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("guard retreat distances must be finite and 1-D")
    if np.any(np.diff(values) < -tolerance_m):
        raise ValueError("guard retreat moved closer to knife")


def _right_to_left_indices(points_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=float)
    indices = np.arange(len(points))
    screen_x = _default_view_screen_x(points)
    if screen_x[0] < screen_x[-1]:
        indices = indices[::-1]
    return indices


def _ordered_right_to_left(points_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=float)
    return points[_right_to_left_indices(points)].copy()


def _default_view_screen_x(points_xy: np.ndarray) -> np.ndarray:
    azimuth = np.deg2rad(_GUARDED_CHOP_VIEW_AZIMUTH_DEG)
    screen_right_xy = np.asarray((-np.sin(azimuth), np.cos(azimuth)))
    return np.asarray(points_xy, dtype=float) @ screen_right_xy


def _prepare_guarded_chop_viewer(robot: RightArmRobot) -> None:
    viewer = robot._viewer
    if viewer is None or not viewer.is_running():
        return
    viewer.cam.azimuth = _GUARDED_CHOP_VIEW_AZIMUTH_DEG
    viewer.cam.elevation = _GUARDED_CHOP_VIEW_ELEVATION_DEG
    viewer.cam.distance = _GUARDED_CHOP_VIEW_DISTANCE_M
    viewer.cam.lookat[:] = _GUARDED_CHOP_VIEW_LOOKAT
    viewer.sync()


def _blade_bottom_height(robot: RightArmRobot, joints_rad: np.ndarray) -> float:
    blade = robot.sim.require_geom("right_knife_blade")
    with robot.right_kinematics._configuration(joints_rad):
        rotation = robot.sim.data.geom_xmat[blade].reshape(3, 3)
        radius = float(
            np.abs(rotation[2]) @ robot.sim.model.geom_size[blade]
        )
        return float(robot.sim.data.geom_xpos[blade, 2] - radius)


def _first_safe_retract_index(
    robot: RightArmRobot,
    retract: Sequence[TrajectoryPoint],
    safe_height_m: float,
) -> int:
    for index, point in enumerate(retract):
        if _blade_bottom_height(robot, point.joints_rad) >= safe_height_m:
            return index
    raise ValueError("retract never reaches safe knife height")


def _guard_plan(
    robot: RightArmRobot,
    cut_points_xy: np.ndarray,
    config: GuardedChopConfig,
    *,
    guard_offset_direction: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[TrajectoryPoint, ...], ...]]:
    model, data = robot.sim.model, robot.sim.data
    knuckle = robot.sim.require_site("left_guard_knuckle_site")
    data.qpos[robot.sim.left.qpos_ids] = LEFT_GRASP_READY_RAD
    data.qpos[robot.sim.hand.qpos_ids] = CAT_PAW_RAD
    mujoco.mj_forward(model, data)
    palm_pose = robot.left_kinematics.fk(LEFT_GRASP_READY_RAD)
    knuckle_local = palm_pose[:3, :3].T @ (
        data.site_xpos[knuckle] - palm_pose[:3, 3]
    )

    direction = cut_points_xy[1] - cut_points_xy[0]
    direction /= np.linalg.norm(direction)
    offset_direction = (
        direction
        if guard_offset_direction is None
        else np.asarray(guard_offset_direction, dtype=float)
    )
    offset_direction /= np.linalg.norm(offset_direction)
    desired_knuckle = np.asarray(
        (cut_points_xy[0, 0], cut_points_xy[0, 1], 0.367), dtype=float
    )
    desired_knuckle[:2] -= (
        config.minimum_distance_m + 0.080
    ) * offset_direction
    ready_pose = palm_pose.copy()
    ready_pose[:3, 3] = (
        desired_knuckle - ready_pose[:3, :3] @ knuckle_local
    )
    result = robot.left_kinematics.ik(
        ready_pose,
        LEFT_GRASP_READY_RAD,
        max_iterations=1000,
        damping=0.003,
    )
    if not result.success:
        raise PathIkError(
            f"guard-ready IK failed (residual {result.residual:.6g})"
        )

    targets = []
    for index in range(config.cuts):
        pose = ready_pose.copy()
        pose[:2, 3] += index * config.hand_shift_m * direction
        targets.append(pose)
    guard_targets = np.asarray(targets)
    shifts = []
    seed = result.joints_rad
    for start, goal in zip(
        guard_targets[:-1], guard_targets[1:], strict=True
    ):
        points = cartesian_trajectory(
            robot.left_kinematics,
            start,
            goal,
            seed,
            config.hand_shift_duration_s,
            config.control_dt_s,
        )
        shifts.append(tuple(points))
        seed = points[-1].joints_rad
    return result.joints_rad, guard_targets, tuple(shifts)


def _finger_pad_position(
    robot: RightArmRobot,
    hand_rad: Sequence[float],
    left_rad: Sequence[float],
) -> np.ndarray:
    saved = mujoco.MjData(robot.sim.model)
    mujoco.mj_copyData(saved, robot.sim.model, robot.sim.data)
    try:
        robot.sim.data.qpos[robot.sim.left.qpos_ids] = np.asarray(
            left_rad, dtype=float
        )
        robot.sim.data.qpos[robot.sim.hand.qpos_ids] = np.asarray(
            hand_rad, dtype=float
        )
        mujoco.mj_forward(robot.sim.model, robot.sim.data)
        pad = robot.sim.require_geom("left_finger3_pad")
        return robot.sim.data.geom_xpos[pad].copy()
    finally:
        mujoco.mj_copyData(robot.sim.data, robot.sim.model, saved)


def _minimum_jerk_joint_trajectory(
    start_rad: Sequence[float],
    goal_rad: Sequence[float],
    duration_s: float,
    dt_s: float,
) -> tuple[np.ndarray, ...]:
    steps = int(round(duration_s / dt_s))
    if steps < 1:
        raise ValueError("trajectory duration must include at least one step")
    progress = np.linspace(0.0, 1.0, steps + 1)
    blend, _ = minimum_jerk(progress)
    start = np.asarray(start_rad, dtype=float)
    goal = np.asarray(goal_rad, dtype=float)
    return tuple(start + value * (goal - start) for value in blend)


def _plane_guard_motions(
    robot: RightArmRobot,
    left_ready_rad: np.ndarray,
    guard_targets: np.ndarray,
    config: GuardedChopConfig,
) -> tuple[_PlaneGuardMotion, ...]:
    hand_retract = _minimum_jerk_joint_trajectory(
        GUARD_RELAXED_RAD,
        GUARD_RETRACTED_RAD,
        config.finger_motion_duration_s,
        config.control_dt_s,
    )
    hand_relax = tuple(reversed(hand_retract))
    left = np.asarray(left_ready_rad, dtype=float).copy()
    motions = []
    for logical_shift in range(4):
        if logical_shift % 2 == 0:
            left_path = tuple(left.copy() for _ in hand_retract)
            hand_path = hand_retract
            phase = GuardedChopPhase.FINGER_RETRACT
        else:
            start_target = guard_targets[logical_shift - 1]
            goal_target = guard_targets[logical_shift + 1]
            arm_points = cartesian_trajectory(
                robot.left_kinematics,
                start_target,
                goal_target,
                left,
                config.finger_motion_duration_s,
                config.control_dt_s,
            )
            left_path = tuple(point.joints_rad.copy() for point in arm_points)
            hand_path = hand_relax
            left = left_path[-1].copy()
            phase = GuardedChopPhase.ARM_RESET_RELAX
        motions.append(_PlaneGuardMotion(phase, left_path, hand_path))
    return tuple(motions)


def _planned_clearances(
    robot: RightArmRobot,
    cuts: tuple[_CutTrajectories, ...],
    knife_lift_shifts: tuple[tuple[TrajectoryPoint, ...], ...],
    guard_targets: np.ndarray,
    guard_shifts: tuple[tuple[TrajectoryPoint, ...], ...],
    plane_guard_motions: tuple[_PlaneGuardMotion, ...],
    left_ready_rad: np.ndarray,
    config: GuardedChopConfig,
) -> np.ndarray:
    model, data = robot.sim.model, robot.sim.data
    distances = []

    def measure(
        right: np.ndarray, left: np.ndarray, hand: np.ndarray = CAT_PAW_RAD
    ) -> float:
        data.qpos[robot.sim.right.qpos_ids] = right
        data.qpos[robot.sim.left.qpos_ids] = left
        data.qpos[robot.sim.hand.qpos_ids] = hand
        mujoco.mj_forward(model, data)
        distance = _blade_hand_distance(robot)
        distances.append(distance)
        return distance

    left = left_ready_rad
    hand = (
        GUARD_RELAXED_RAD
        if config.scene_mode == "plane"
        else CAT_PAW_RAD
    )
    for index, cut in enumerate(cuts):
        for point in cut.descent:
            measure(point.joints_rad, left, hand)
        if config.scene_mode == "object":
            for point in (*cut.retract, *cut.shift):
                measure(point.joints_rad, left)
            if index < len(guard_shifts):
                right = cut.shift[-1].joints_rad
                for point in guard_shifts[index]:
                    measure(right, point.joints_rad)
                left = guard_shifts[index][-1].joints_rad
            continue
        if index >= len(plane_guard_motions):
            for point in cut.retract:
                measure(point.joints_rad, left, hand)
            continue

        contact_right = cut.descent[-1].joints_rad
        motion = plane_guard_motions[index]
        retreat_distances = []
        for left_target, hand_target in zip(
            motion.left, motion.hand, strict=True
        ):
            retreat_distances.append(
                measure(contact_right, left_target, hand_target)
            )
        left = motion.left[-1]
        hand = motion.hand[-1]
        _validate_nonapproaching_retreat(
            retreat_distances, tolerance_m=2e-6
        )
        for point in knife_lift_shifts[index]:
            measure(point.joints_rad, left, hand)
    return np.asarray(distances)


def run_guarded_chop(
    config: GuardedChopConfig = GuardedChopConfig(),
    *,
    viewer: bool = False,
    trace=None,
    coordinator: SafetyCoordinator | None = None,
    replay_rate: float | None = None,
    record_path: Path | None = None,
) -> GuardedChopResult:
    config = config.validated()
    playback_rate = (
        guarded_chop_recording.validate_replay_rate(replay_rate)
        if replay_rate is not None
        else None
    )
    if playback_rate is not None and not viewer:
        raise ValueError("replay requires a Viewer")
    capture_requested = playback_rate is not None or record_path is not None
    safety = coordinator or SafetyCoordinator(
        config.minimum_distance_m,
        config.maximum_guard_penetration_m,
        config.maximum_guard_force_n,
    )
    robot = RightArmRobot(viewer=False)
    samples: list[GuardedChopSample] = []
    phase = GuardedChopPhase.INITIALIZE
    completed_cuts = 0
    completed_shifts = 0
    recorded_frames = [] if capture_requested else None
    viewer_starting = False
    try:
        plan = _preflight_guarded_chop(robot, config)
        _configure_guarded_scene(robot, config.scene_mode)
        robot.reset(plan.right_ready_rad)
        robot.sim.data.qpos[robot.sim.left.qpos_ids] = plan.left_ready_rad
        robot.sim.data.qvel[robot.sim.left.dof_ids] = 0.0
        robot.command_left(plan.left_ready_rad)
        robot.sim.data.qpos[robot.sim.hand.qpos_ids] = CAT_PAW_RAD
        robot.sim.data.qvel[robot.sim.hand.dof_ids] = 0.0
        robot.hand.command(CAT_PAW_RAD)
        robot.sim.data.ctrl[robot.sim.left.actuator_ids] = plan.left_ready_rad
        robot.sim.data.ctrl[robot.sim.hand.actuator_ids] = CAT_PAW_RAD
        mujoco.mj_forward(robot.sim.model, robot.sim.data)

        if viewer:
            viewer_starting = True
            robot.open_viewer()
            _prepare_guarded_chop_viewer(robot)
            if trace is None:
                from twin_sim.guarded_chop_visualization import (
                    GuardedChopTrace,
                )

                trace = GuardedChopTrace(robot._viewer)
            viewer_starting = False
        if trace is not None:
            blade_contact_heights = np.asarray(
                [
                    _blade_center_for_joints(
                        robot, cut.descent[-1].joints_rad
                    )[2]
                    for cut in plan.cuts
                ]
            )
            trace.set_plan(
                np.column_stack(
                    (plan.cut_points_xy, blade_contact_heights)
                )
            )

        cube_pose = None
        if config.scene_mode == "object":
            cube_site = robot.sim.require_site("guarded_chop_cube_center")
            cube_pose = (
                cube_site,
                robot.sim.data.site_xpos[cube_site].copy(),
                robot.sim.data.site_xmat[cube_site].copy(),
            )
        last_left_target = robot._left_target.copy()
        last_right_target = robot._right_target.copy()
        viewer_step_index = 0
        viewer_sync_stride = _viewer_sync_stride(config.control_dt_s)

        def step_and_record(
            current_phase: GuardedChopPhase,
            cut_index: int,
        ) -> None:
            nonlocal last_left_target, last_right_target, viewer_step_index
            left_stationary = np.array_equal(
                robot._left_target, last_left_target
            )
            right_stationary = np.array_equal(
                robot._right_target, last_right_target
            )
            robot.step(
                config.control_dt_s,
                sync_viewer=(
                    viewer_step_index % viewer_sync_stride == 0
                ),
            )
            viewer_step_index += 1
            sample = _observe_sample(
                robot,
                current_phase,
                cut_index,
                plan.safe_knife_height_m,
                left_stationary=left_stationary,
                right_stationary=right_stationary,
                safety=safety,
                trace=trace,
            )
            samples.append(sample)
            if recorded_frames is not None:
                recorded_frames.append(
                    guarded_chop_recording.capture_frame(robot, sample)
                )
            last_left_target = robot._left_target.copy()
            last_right_target = robot._right_target.copy()
            if not sample.safety_allowed:
                observation = _safety_observation(
                    robot,
                    current_phase,
                    plan.safe_knife_height_m,
                    sample.knife_guard_distance_m,
                    left_stationary,
                    right_stationary,
                )
                decision = safety.evaluate(observation)
                raise RuntimeError(decision.reason)

        def latch_hand_pose() -> None:
            hand_position = robot.sim.data.qpos[
                robot.sim.hand.qpos_ids
            ].copy()
            hand_ranges = robot.sim.model.actuator_ctrlrange[
                robot.sim.hand.actuator_ids
            ]
            robot.hand.command(
                np.clip(hand_position, hand_ranges[:, 0], hand_ranges[:, 1])
            )

        def wait_for_guard_stability(
            current_phase: GuardedChopPhase,
            cut_index: int,
            *,
            latch_contact: bool,
        ) -> None:
            if latch_contact:
                latch_hand_pose()
            stable_time = 0.0
            maximum_steps = int(
                round(config.stability_timeout_s / config.control_dt_s)
            )
            for _ in range(maximum_steps):
                step_and_record(current_phase, cut_index)
                hand_speed = float(
                    np.linalg.norm(
                        robot.sim.data.qvel[robot.sim.hand.dof_ids]
                    )
                )
                if (
                    np.linalg.norm(robot.left_joint_velocities) <= 0.05
                    and hand_speed <= 0.05
                ):
                    stable_time += config.control_dt_s
                    if stable_time + 1e-12 >= config.interlock_settle_s:
                        return
                else:
                    stable_time = 0.0
            raise RuntimeError("left guard did not settle before cut")

        def wait_for_knife_stability(
            current_phase: GuardedChopPhase,
            cut_index: int,
        ) -> None:
            stable_time = 0.0
            maximum_steps = int(
                round(config.stability_timeout_s / config.control_dt_s)
            )
            for _ in range(maximum_steps):
                step_and_record(current_phase, cut_index)
                if np.linalg.norm(robot.joint_velocities) <= 0.05:
                    stable_time += config.control_dt_s
                    if stable_time + 1e-12 >= config.interlock_settle_s:
                        return
                else:
                    stable_time = 0.0
            raise RuntimeError("knife did not settle before guard shift")

        def wait_for_knife_clearance(cut_index: int) -> None:
            maximum_steps = int(
                round(config.stability_timeout_s / config.control_dt_s)
            )
            for _ in range(maximum_steps):
                if _current_blade_bottom(robot) >= plan.safe_knife_height_m:
                    return
                step_and_record(GuardedChopPhase.KNIFE_CLEAR, cut_index)
            raise RuntimeError("knife did not reach safe height")

        phase = GuardedChopPhase.GUARD_READY
        for _ in range(
            int(round(config.guard_ready_duration_s / config.control_dt_s))
        ):
            step_and_record(phase, 0)
        wait_for_guard_stability(
            phase,
            0,
            latch_contact=config.scene_mode == "object",
        )

        for index, cut in enumerate(plan.cuts, start=1):
            phase = GuardedChopPhase.CUT_DOWN
            for point in cut.descent[1:]:
                robot.command(point.joints_rad)
                step_and_record(phase, index)
            completed_cuts += 1

            if index > len(plan.guard_shifts):
                phase = GuardedChopPhase.KNIFE_CLEAR
                for point in cut.retract[1:]:
                    robot.command(point.joints_rad)
                    step_and_record(phase, index)
                continue

            if config.scene_mode == "plane":
                wait_for_knife_stability(phase, index)
                motion = plan.plane_guard_motions[index - 1]
                phase = motion.phase
                for left_target, hand_target in zip(
                    motion.left[1:], motion.hand[1:], strict=True
                ):
                    robot.command_left(left_target)
                    robot.hand.command(hand_target)
                    step_and_record(phase, index)
                phase = GuardedChopPhase.GUARD_SETTLE
                wait_for_guard_stability(
                    phase, index, latch_contact=False
                )

                phase = GuardedChopPhase.KNIFE_LIFT_SHIFT
                for point in plan.knife_lift_shifts[index - 1][1:]:
                    robot.command(point.joints_rad)
                    step_and_record(phase, index)
            else:
                phase = GuardedChopPhase.KNIFE_CLEAR
                safe_index = _first_safe_retract_index(
                    robot,
                    cut.retract,
                    plan.safe_knife_height_m
                    + _KNIFE_CLEAR_TARGET_HEADROOM_M,
                )
                for point in cut.retract[1 : safe_index + 1]:
                    robot.command(point.joints_rad)
                    step_and_record(phase, index)
                wait_for_knife_clearance(index)
                for point in (
                    *cut.retract[safe_index + 1 :],
                    *cut.shift[1:],
                ):
                    robot.command(point.joints_rad)
                    step_and_record(phase, index)
                wait_for_knife_stability(phase, index)
                phase = GuardedChopPhase.LOW_GUARD_SHIFT
                for point in plan.guard_shifts[index - 1][1:]:
                    robot.command_left(point.joints_rad)
                    latch_hand_pose()
                    step_and_record(phase, index)
                phase = GuardedChopPhase.GUARD_SETTLE
                wait_for_guard_stability(
                    phase, index, latch_contact=True
                )
            completed_shifts += 1

        phase = GuardedChopPhase.COMPLETE
        complete_steps = max(
            1, int(round(config.final_hold_s / config.control_dt_s))
        )
        for _ in range(complete_steps):
            step_and_record(phase, config.cuts)

        if cube_pose is not None:
            cube_site, cube_position, cube_rotation = cube_pose
            np.testing.assert_array_equal(
                robot.sim.data.site_xpos[cube_site], cube_position
            )
            np.testing.assert_array_equal(
                robot.sim.data.site_xmat[cube_site], cube_rotation
            )
        minimum = min(
            sample.knife_guard_distance_m for sample in samples
        )
        if completed_cuts != 5 or completed_shifts != 4:
            raise RuntimeError("guarded chopping did not complete 5 cuts/4 shifts")
        if minimum < config.minimum_distance_m:
            raise RuntimeError("knife-guard distance below final limit")
        recording = (
            guarded_chop_recording.GuardedChopRecording(recorded_frames)
            if recorded_frames is not None
            else None
        )
        if record_path is not None and recording is not None:
            guarded_chop_recording.save_recording(recording, record_path)
        if playback_rate is not None and recording is not None:
            guarded_chop_recording.replay_recording(
                robot,
                recording,
                rate=playback_rate,
                trace=trace,
            )
        return GuardedChopResult(
            success=True,
            final_phase=phase,
            reason="",
            samples=tuple(samples),
            completed_cuts=completed_cuts,
            completed_shifts=completed_shifts,
            total_shift_m=completed_shifts * config.hand_shift_m,
            minimum_distance_m=minimum,
        )
    except (AssertionError, PathIkError, RuntimeError, ValueError) as error:
        if viewer_starting:
            raise
        if trace is not None and hasattr(trace, "set_abort"):
            trace.set_abort(str(error))
        if record_path is not None and recorded_frames:
            guarded_chop_recording.save_recording(
                guarded_chop_recording.GuardedChopRecording(recorded_frames),
                record_path,
            )
        minimum = min(
            (sample.knife_guard_distance_m for sample in samples),
            default=float("inf"),
        )
        return GuardedChopResult(
            success=False,
            final_phase=GuardedChopPhase.ABORTED,
            reason=str(error),
            samples=tuple(samples),
            completed_cuts=completed_cuts,
            completed_shifts=completed_shifts,
            total_shift_m=completed_shifts * config.hand_shift_m,
            minimum_distance_m=minimum,
        )
    finally:
        robot.close()


def _configure_guarded_scene(
    robot: RightArmRobot, scene_mode: Literal["plane", "object"]
) -> None:
    cube = robot.sim.require_geom("guarded_chop_cube")
    cube_body = int(robot.sim.model.geom_bodyid[cube])
    guard_contact_bit = 8
    if scene_mode == "object":
        robot.sim.model.geom_rgba[cube] = (0.75, 0.18, 0.12, 1.0)
        robot.sim.model.geom_contype[cube] = guard_contact_bit
        robot.sim.model.geom_conaffinity[cube] = guard_contact_bit
        robot.sim.model.body_contype[cube_body] = guard_contact_bit
        robot.sim.model.body_conaffinity[cube_body] = guard_contact_bit
    elif scene_mode == "plane":
        robot.sim.model.geom_rgba[cube, 3] = 0.0
        robot.sim.model.geom_contype[cube] = 0
        robot.sim.model.geom_conaffinity[cube] = 0
        robot.sim.model.body_contype[cube_body] = 0
        robot.sim.model.body_conaffinity[cube_body] = 0
    else:
        raise ValueError("scene_mode must be 'plane' or 'object'")

    guarded_geoms = ("left_finger3_pad", "right_knife_blade")
    for name in guarded_geoms:
        geom = robot.sim.require_geom(name)
        if scene_mode == "object":
            robot.sim.model.geom_contype[geom] |= guard_contact_bit
            robot.sim.model.geom_conaffinity[geom] |= guard_contact_bit
        else:
            robot.sim.model.geom_contype[geom] &= ~guard_contact_bit
            robot.sim.model.geom_conaffinity[geom] &= ~guard_contact_bit
        if name == "right_knife_blade":
            robot.sim.model.geom_contype[geom] |= 2
            robot.sim.model.geom_conaffinity[geom] |= 2
        body = int(robot.sim.model.geom_bodyid[geom])
        while body:
            if scene_mode == "object":
                robot.sim.model.body_contype[body] |= guard_contact_bit
                robot.sim.model.body_conaffinity[body] |= guard_contact_bit
            else:
                robot.sim.model.body_contype[body] &= ~guard_contact_bit
                robot.sim.model.body_conaffinity[body] &= ~guard_contact_bit
            if name == "right_knife_blade":
                robot.sim.model.body_contype[body] |= 2
                robot.sim.model.body_conaffinity[body] |= 2
            body = int(robot.sim.model.body_parentid[body])
    for name in (
        "pick_source_pedestal",
        "pick_target_pedestal",
        "pick_cube_geom",
        "pick_target_region",
    ):
        geom = robot.sim.require_geom(name)
        robot.sim.model.geom_rgba[geom, 3] = 0.0
        robot.sim.model.geom_contype[geom] = 0
        robot.sim.model.geom_conaffinity[geom] = 0
    palm = robot.sim.require_body("left_palm_link")
    for geom in range(robot.sim.model.ngeom):
        if (
            _geom_belongs_to_body_tree(robot, geom, palm)
            and mujoco.mj_id2name(
                robot.sim.model, mujoco.mjtObj.mjOBJ_GEOM, geom
            )
            is None
        ):
            robot.sim.model.geom_contype[geom] = 0
            robot.sim.model.geom_conaffinity[geom] = 0


def _activate_guarded_scene(robot: RightArmRobot) -> None:
    _configure_guarded_scene(robot, "object")


def _safety_observation(
    robot: RightArmRobot,
    phase: GuardedChopPhase,
    safe_height_m: float,
    distance_m: float,
    left_stationary: bool,
    right_stationary: bool,
) -> SafetyObservation:
    _, penetration_m, normal_force_n = _hand_cube_contact_metrics(robot)
    finite = all(
        np.isfinite(values).all()
        for values in (
            robot.sim.data.qpos,
            robot.sim.data.qvel,
            robot.sim.data.ctrl,
        )
    )
    return SafetyObservation(
        phase=phase.value,
        knife_height_m=_current_blade_bottom(robot),
        safe_knife_height_m=safe_height_m,
        knife_guard_distance_m=distance_m,
        left_target_stationary=left_stationary,
        right_target_stationary=right_stationary,
        left_speed_rad_s=float(
            np.linalg.norm(robot.left_joint_velocities)
        ),
        right_speed_rad_s=float(np.linalg.norm(robot.joint_velocities)),
        hand_speed_rad_s=float(
            np.linalg.norm(
                robot.sim.data.qvel[robot.sim.hand.dof_ids]
            )
        ),
        guard_cube_penetration_m=penetration_m,
        guard_cube_normal_force_n=normal_force_n,
        finite_state=finite,
    )


def _observe_sample(
    robot: RightArmRobot,
    phase: GuardedChopPhase,
    cut_index: int,
    safe_height_m: float,
    *,
    left_stationary: bool,
    right_stationary: bool,
    safety: SafetyCoordinator,
    trace=None,
) -> GuardedChopSample:
    blade = robot.sim.require_geom("right_knife_blade")
    finger_pad = robot.sim.require_geom("left_finger3_pad")
    guard_position = robot.sim.data.geom_xpos[finger_pad].copy()
    distance = _blade_hand_distance(robot)
    observation = _safety_observation(
        robot,
        phase,
        safe_height_m,
        distance,
        left_stationary,
        right_stationary,
    )
    decision = safety.evaluate(observation)
    sample = GuardedChopSample(
        time_s=float(robot.sim.data.time),
        phase=phase,
        cut_index=cut_index,
        left_target_rad=robot._left_target,
        left_actual_rad=robot.left_joint_positions,
        right_target_rad=robot._right_target,
        right_actual_rad=robot.joint_positions,
        hand_target_rad=robot.hand.target,
        knife_position=robot.sim.data.geom_xpos[blade],
        guard_position=guard_position,
        knife_guard_distance_m=distance,
        knife_height_m=observation.knife_height_m,
        cut_allowed=(
            decision.allowed and phase is GuardedChopPhase.CUT_DOWN
        ),
        safety_allowed=decision.allowed,
        guard_cube_contact_count=_hand_cube_contact_count(robot),
        guard_cube_penetration_m=observation.guard_cube_penetration_m,
        guard_cube_normal_force_n=observation.guard_cube_normal_force_n,
        blade_hand_contact_count=_blade_hand_contact_count(robot),
        left_speed_rad_s=observation.left_speed_rad_s,
        right_speed_rad_s=observation.right_speed_rad_s,
        hand_speed_rad_s=observation.hand_speed_rad_s,
    )
    if trace is not None:
        trace.append(
            actual_knife=sample.knife_position,
            actual_guard=sample.guard_position,
            phase=phase.value,
            cut_index=cut_index,
            minimum_distance_m=distance,
            cut_allowed=sample.cut_allowed,
        )
    return sample


def _current_blade_bottom(robot: RightArmRobot) -> float:
    blade = robot.sim.require_geom("right_knife_blade")
    rotation = robot.sim.data.geom_xmat[blade].reshape(3, 3)
    radius = float(
        np.abs(rotation[2]) @ robot.sim.model.geom_size[blade]
    )
    return float(robot.sim.data.geom_xpos[blade, 2] - radius)


def _blade_center_for_joints(
    robot: RightArmRobot, joints_rad: np.ndarray
) -> np.ndarray:
    blade = robot.sim.require_geom("right_knife_blade")
    with robot.right_kinematics._configuration(joints_rad):
        return robot.sim.data.geom_xpos[blade].copy()


def _hand_cube_contact_count(robot: RightArmRobot) -> int:
    return _hand_cube_contact_metrics(robot)[0]


def _hand_cube_contact_metrics(
    robot: RightArmRobot,
) -> tuple[int, float, float]:
    cube = robot.sim.require_geom("guarded_chop_cube")
    palm = robot.sim.require_body("left_palm_link")

    count = 0
    maximum_penetration = 0.0
    maximum_normal_force = 0.0
    for index in range(robot.sim.data.ncon):
        contact = robot.sim.data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if (
            geom1 == cube
            and _geom_belongs_to_body_tree(robot, geom2, palm)
        ) or (
            geom2 == cube
            and _geom_belongs_to_body_tree(robot, geom1, palm)
        ):
            count += 1
            maximum_penetration = max(
                maximum_penetration, max(0.0, -float(contact.dist))
            )
            contact_force = np.zeros(6)
            mujoco.mj_contactForce(
                robot.sim.model, robot.sim.data, index, contact_force
            )
            maximum_normal_force = max(
                maximum_normal_force, abs(float(contact_force[0]))
            )
    return count, maximum_penetration, maximum_normal_force


def _blade_hand_distance(robot: RightArmRobot) -> float:
    blade = robot.sim.require_geom("right_knife_blade")
    palm = robot.sim.require_body("left_palm_link")
    distances = []
    for geom in range(robot.sim.model.ngeom):
        if geom == blade or robot.sim.model.geom_contype[geom] == 0:
            continue
        name = mujoco.mj_id2name(
            robot.sim.model, mujoco.mjtObj.mjOBJ_GEOM, geom
        )
        if name is None or not _geom_belongs_to_body_tree(
            robot, geom, palm
        ):
            continue
        from_to = np.zeros(6)
        distance = mujoco.mj_geomDistance(
            robot.sim.model,
            robot.sim.data,
            blade,
            geom,
            1.0,
            from_to,
        )
        distances.append(float(distance))
    if not distances:
        raise RuntimeError("no collidable left-hand geometry found")
    return min(distances)


def _blade_hand_contact_count(robot: RightArmRobot) -> int:
    blade = robot.sim.require_geom("right_knife_blade")
    palm = robot.sim.require_body("left_palm_link")
    count = 0
    for index in range(robot.sim.data.ncon):
        contact = robot.sim.data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if (
            geom1 == blade
            and _geom_belongs_to_body_tree(robot, geom2, palm)
        ) or (
            geom2 == blade
            and _geom_belongs_to_body_tree(robot, geom1, palm)
        ):
            count += 1
    return count


def _geom_belongs_to_body_tree(
    robot: RightArmRobot, geom: int, root_body: int
) -> bool:
    body = int(robot.sim.model.geom_bodyid[geom])
    while body:
        if body == root_body:
            return True
        body = int(robot.sim.model.body_parentid[body])
    return False
