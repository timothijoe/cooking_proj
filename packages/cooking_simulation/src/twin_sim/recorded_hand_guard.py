"""Adapt recorded Wuji tabletop motion for the combined arm task."""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from tianji_robotics.data.mcap import (
    JointStateMcapSource,
    StudioMcapSkeletonSource,
    detect_hand_mcap_kind,
)
from tianji_robotics.simulation.tabletop_wuji_hand import TabletopWujiHand
from tianji_robotics.workflows.wuji_glove_replay import retarget_recording
from tianji_robotics.workflows.wuji_table_retreat import build_recorded_table_retreat
from tianji_robotics.wuji_hand.models import HandTrajectory


@dataclass(frozen=True)
class RecordedGuardCycle:
    timestamps_s: np.ndarray
    hand_positions_rad: np.ndarray
    relative_palm_transforms: np.ndarray
    initial_palm_transform: np.ndarray
    phases: tuple[str, ...]
    source_kind: str
    source_frame_count: int
    source_duration_s: float
    motion_start_frame: int
    motion_end_frame: int
    retreat_distance_m: float
    maximum_joint_correction_rad: float


def shape_pip_led_guard_hand(
    positions_rad: np.ndarray,
    phases: tuple[str, ...],
    control_ranges: np.ndarray,
) -> np.ndarray:
    positions = np.asarray(positions_rad, dtype=float)
    limits = np.asarray(control_ranges, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 20:
        raise ValueError("positions_rad must have shape (N, 20)")
    if len(phases) != len(positions):
        raise ValueError("phases must align with positions_rad")
    if limits.shape != (20, 2) or not np.isfinite(limits).all():
        raise ValueError("control_ranges must have shape (20, 2)")
    shaped = positions.copy()
    for mcp, pip in zip((4, 8, 12, 16), (6, 10, 14, 18), strict=True):
        original_mcp = positions[:, mcp]
        shaped[:, mcp] = np.minimum(original_mcp, 0.30)
        transferred = np.maximum(0.0, original_mcp - shaped[:, mcp])
        shaped[:, pip] = np.maximum(
            positions[:, pip] + transferred,
            shaped[:, mcp] + 0.25,
        )
        shaped[:, mcp] = np.clip(
            shaped[:, mcp], limits[mcp, 0], limits[mcp, 1]
        )
        shaped[:, pip] = np.clip(
            shaped[:, pip], limits[pip, 0], limits[pip, 1]
        )
    if np.max(np.abs(np.diff(shaped, axis=0)), initial=0.0) > 0.12:
        raise ValueError("PIP-led shaping exceeds 0.12 rad joint-step limit")
    return shaped


def _load_trajectory(path: Path) -> tuple[HandTrajectory, str]:
    kind = detect_hand_mcap_kind(path)
    if kind == "joint_states":
        return JointStateMcapSource(path).trajectory(), kind
    from tianji_robotics.wuji_sdk.retargeter import OfficialWujiRetargeter

    return (
        retarget_recording(
            StudioMcapSkeletonSource(path),
            OfficialWujiRetargeter.create_left_first_generation(),
        ),
        kind,
    )


def _pose_transform(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(quaternion, dtype=float))
    transform = np.eye(4)
    transform[:3, :3] = rotation.reshape(3, 3)
    transform[:3, 3] = position
    return transform


def _resample(values: np.ndarray, source_s: np.ndarray, target_s: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [np.interp(target_s, source_s, values[:, axis]) for axis in range(values.shape[1])]
    )


def load_recorded_guard_cycle(
    path: Path,
    *,
    control_dt_s: float = 0.01,
) -> RecordedGuardCycle:
    if not np.isfinite(control_dt_s) or control_dt_s <= 0.0:
        raise ValueError("control_dt_s must be positive and finite")
    trajectory, source_kind = _load_trajectory(Path(path))
    backend = TabletopWujiHand(viewer=False)
    try:
        corrected, report = build_recorded_table_retreat(
            trajectory, backend, source_kind=source_kind
        )
    finally:
        backend.close()

    source_s = (corrected.timestamps_ns - corrected.timestamps_ns[0]) / 1e9
    duration_s = float(source_s[-1])
    sample_count = max(2, int(round(duration_s / control_dt_s)) + 1)
    target_s = np.linspace(0.0, duration_s, sample_count)
    hand = _resample(corrected.positions_rad, source_s, target_s)
    palms = _resample(corrected.palm_positions_m, source_s, target_s)
    quaternions = _resample(corrected.palm_quaternions_wxyz, source_s, target_s)
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    transforms = np.asarray(
        [_pose_transform(position, quaternion) for position, quaternion in zip(palms, quaternions, strict=True)]
    )
    first_inverse = np.linalg.inv(transforms[0])
    relative = np.asarray([first_inverse @ transform for transform in transforms])
    source_indices = np.searchsorted(source_s, target_s, side="right") - 1
    source_indices = np.clip(source_indices, 0, len(corrected.phases) - 1)
    phases = tuple(corrected.phases[index] for index in source_indices)
    return RecordedGuardCycle(
        timestamps_s=target_s,
        hand_positions_rad=hand,
        relative_palm_transforms=relative,
        initial_palm_transform=transforms[0],
        phases=phases,
        source_kind=source_kind,
        source_frame_count=len(trajectory.timestamps_ns),
        source_duration_s=float(
            trajectory.timestamps_ns[-1] - trajectory.timestamps_ns[0]
        )
        / 1e9,
        motion_start_frame=report.motion_start_frame,
        motion_end_frame=report.motion_end_frame,
        retreat_distance_m=report.actual_retreat_m,
        maximum_joint_correction_rad=report.maximum_joint_correction_rad,
    )
