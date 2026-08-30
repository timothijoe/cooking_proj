#!/usr/bin/env python3
"""Offline: translate every right-arm TCP in its base frame, then solve IK.

The default input is the current dual-arm recording.  This tool never connects
to the robot.  It preserves every original NPZ array, replacing only
``right_arm_target_rad`` and adding transformation metadata to the output.
Coordinates follow the right-arm kinematics SDK base frame.  The current
defaults use +X for forward and +Z for right.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np


from robot_core.paths import local_root
ROOT = local_root()
DEFAULT_SOURCE = ROOT / "recordings" / "recorded_hand_guarded_chop_200hz_official_right_retarget_candidate.npz"
DEFAULT_OUTPUT = ROOT / "recordings" / "recorded_hand_guarded_chop_200hz_official_right_retarget_candidate_forward_x_plus_30mm_right_z_plus_100mm.npz"
DEFAULT_KINE_CONFIG = ROOT / "test" / "ccs_m6_40.MvKDCfg"
ARM_JOINT_LIMITS_DEG = np.array(
    [(-170, 170), (-100, 120), (-170, 170), (-130, 130), (-170, 170), (-90, 220), (-170, 170)],
    dtype=float,
)


class TrajectoryShiftError(RuntimeError):
    """The requested Cartesian offset cannot be converted into a safe joint path."""


class Kinematics(Protocol):
    def fk(self, joints: list[float]) -> Any: ...

    def solve_ik(self, target_matrix: np.ndarray, reference_joints: np.ndarray) -> np.ndarray | None: ...


@dataclass(frozen=True)
class ShiftReport:
    frames: int
    shift_base_mm: np.ndarray
    max_tcp_position_error_mm: float
    max_joint_step_deg: float
    joint_min_deg: np.ndarray
    joint_max_deg: np.ndarray


def _right_joints_deg(source: Mapping[str, np.ndarray]) -> np.ndarray:
    try:
        right_rad = np.asarray(source["right_arm_target_rad"], dtype=float)
        time_s = np.asarray(source["time_s"], dtype=float)
    except KeyError as exc:
        raise ValueError(f"input trajectory is missing {exc.args[0]}") from exc
    if time_s.ndim != 1 or time_s.size < 2:
        raise ValueError("time_s must have at least two frames")
    if right_rad.shape != (time_s.size, 7) or not np.all(np.isfinite(right_rad)):
        raise ValueError("right_arm_target_rad must be finite with shape (N, 7)")
    return np.rad2deg(right_rad)


def _target_matrix(kine: Kinematics, joints_deg: np.ndarray, shift_base_mm: np.ndarray) -> np.ndarray:
    matrix = np.asarray(kine.fk(joints_deg.tolist()), dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise TrajectoryShiftError("FK returned an invalid 4x4 TCP matrix")
    result = matrix.copy()
    result[:3, 3] += shift_base_mm
    return result


def shift_right_arm_trajectory(
    source: Mapping[str, np.ndarray],
    kine: Kinematics,
    *,
    shift_mm: float | None = None,
    shift_base_mm: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], ShiftReport]:
    """Solve a continuous right-arm IK path for a fixed base-frame offset."""
    if shift_base_mm is not None and shift_mm is not None:
        raise ValueError("specify either shift_mm or shift_base_mm, not both")
    if shift_mm is not None:
        offset = np.array([float(shift_mm), 0.0, 0.0])
    elif shift_base_mm is not None:
        offset = np.asarray(shift_base_mm, dtype=float)
    else:
        offset = np.array([30.0, 0.0, 100.0])
    if offset.shape != (3,) or not np.all(np.isfinite(offset)) or np.allclose(offset, 0.0):
        raise ValueError("base-frame shift must be three finite values and non-zero")
    original_deg = _right_joints_deg(source)
    solved_deg = np.empty_like(original_deg)
    max_position_error_mm = 0.0
    reference = original_deg[0].copy()

    for frame, original in enumerate(original_deg):
        target = _target_matrix(kine, original, offset)
        solved = kine.solve_ik(target, reference)
        if solved is None:
            raise TrajectoryShiftError(f"IK found no valid solution at frame {frame}")
        joints = np.asarray(solved, dtype=float)
        if joints.shape != (7,) or not np.all(np.isfinite(joints)):
            raise TrajectoryShiftError(f"IK returned invalid joints at frame {frame}")
        if np.any(joints < ARM_JOINT_LIMITS_DEG[:, 0] + 1.0) or np.any(joints > ARM_JOINT_LIMITS_DEG[:, 1] - 1.0):
            raise TrajectoryShiftError(f"IK result violates the 1 deg joint margin at frame {frame}: {joints.tolist()}")
        reached = np.asarray(kine.fk(joints.tolist()), dtype=float)
        if reached.shape != (4, 4) or not np.all(np.isfinite(reached)):
            raise TrajectoryShiftError(f"FK validation failed at frame {frame}")
        position_error_mm = float(np.linalg.norm(reached[:3, 3] - target[:3, 3]))
        if position_error_mm > 0.1:
            raise TrajectoryShiftError(f"IK TCP position error exceeds 0.1 mm at frame {frame}: {position_error_mm:.4f}")
        max_position_error_mm = max(max_position_error_mm, position_error_mm)
        solved_deg[frame] = joints
        reference = joints

    result = {name: np.asarray(values).copy() for name, values in source.items()}
    result["right_arm_target_rad"] = np.deg2rad(solved_deg)
    result["right_arm_tcp_shift_base_mm"] = offset.copy()
    result["right_arm_tcp_shift_frame"] = np.asarray("right_arm_kinematics_base")
    report = ShiftReport(
        frames=len(solved_deg),
        shift_base_mm=offset.copy(),
        max_tcp_position_error_mm=max_position_error_mm,
        max_joint_step_deg=float(np.abs(np.diff(solved_deg, axis=0)).max()),
        joint_min_deg=solved_deg.min(axis=0),
        joint_max_deg=solved_deg.max(axis=0),
    )
    return result, report


class SdkRightArmKinematics:
    """Small adapter around the vendor SDK, initialized only for arm B."""

    def __init__(self, config_path: Path):
        from SDK_PYTHON.fx_kine import FX_InvKineSolvePara, Marvin_Kine

        self._ik_parameter_type = FX_InvKineSolvePara
        self._kine = Marvin_Kine()
        self._kine.log_switch(0)
        config = self._kine.load_config(arm_type=1, config_path=str(config_path))
        if not self._kine.initial_kine(
            robot_type=config["TYPE"][1], dh=config["DH"][1], pnva=config["PNVA"][1], j67=config["BD"][1]
        ):
            raise TrajectoryShiftError("failed to initialize right-arm SDK kinematics")

    def fk(self, joints: list[float]) -> Any:
        return self._kine.fk(joints)

    def solve_ik(self, target_matrix: np.ndarray, reference_joints: np.ndarray) -> np.ndarray | None:
        parameter = self._ik_parameter_type()
        parameter.set_input_ik_target_tcp(self._kine.mat4x4_to_mat1x16(target_matrix.tolist()))
        parameter.set_input_ik_ref_joint(reference_joints.tolist())
        parameter.set_input_ik_zsp_type(0)
        result = self._kine.ik(parameter)
        if not result or result.get_output_result_num() < 1:
            return None
        joints = np.asarray(list(result.get_output_ret_joint())[:7], dtype=float)
        return joints if joints.shape == (7,) else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-npz", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-npz", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--kine-config", type=Path, default=DEFAULT_KINE_CONFIG)
    parser.add_argument("--shift-x-mm", type=float, default=30.0, help="Base +X: forward offset (mm)")
    parser.add_argument("--shift-y-mm", type=float, default=0.0, help="Base Y offset (mm)")
    parser.add_argument("--shift-z-mm", type=float, default=100.0, help="Base +Z: right offset (mm)")
    args = parser.parse_args(argv)
    if not args.source_npz.is_file():
        raise ValueError(f"source NPZ not found: {args.source_npz}")
    if not args.kine_config.is_file():
        raise ValueError(f"kinematics config not found: {args.kine_config}")
    if args.output_npz.resolve() == args.source_npz.resolve():
        raise ValueError("output-npz must differ from source-npz")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        with np.load(args.source_npz, allow_pickle=False) as archive:
            source = {name: archive[name] for name in archive.files}
        transformed, report = shift_right_arm_trajectory(
            source,
            SdkRightArmKinematics(args.kine_config),
            shift_base_mm=np.array([args.shift_x_mm, args.shift_y_mm, args.shift_z_mm]),
        )
        np.savez(args.output_npz, **transformed)
    except (ValueError, TrajectoryShiftError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"WROTE {args.output_npz}")
    print(f"frames={report.frames} base_xyz_shift_mm={np.round(report.shift_base_mm, 3).tolist()}")
    print(f"max_tcp_position_error_mm={report.max_tcp_position_error_mm:.4f}")
    print(f"max_joint_step_deg={report.max_joint_step_deg:.4f}")
    print(f"right_joint_min_deg={np.round(report.joint_min_deg, 3).tolist()}")
    print(f"right_joint_max_deg={np.round(report.joint_max_deg, 3).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
