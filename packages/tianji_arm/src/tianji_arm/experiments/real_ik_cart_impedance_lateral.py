#!/usr/bin/env python3
"""Real left-arm IK + Cartesian impedance lateral down-up debug script.

The script connects to the real controller, optionally moves the SDK A arm
(left arm on this robot) to an initialization joint pose, then executes a
down-up chopping motion from the current FK pose. The default command path is
planned Cartesian position mode (MOVLA + setPln_Cart); the lower-level
position and Cartesian impedance paths are kept for comparison/debugging.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


from robot_core.paths import local_root
ROOT = local_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def mirror_right_to_left_joints(joints: Sequence[float]) -> tuple[float, float, float, float, float, float, float]:
    """Convert a right-arm joint pose into the matching left-arm joint pose.

    Joint numbering in the robot docs starts at 1. For this robot, joints
    1, 3, 5, and 7 are mirrored by changing sign, while joints 2, 4, and 6
    keep the same value. The function validates the input because these
    values may be used as real-machine initialization targets.
    """
    values = tuple(float(joint) for joint in joints)
    if len(values) != 7:
        raise ValueError("joints must contain 7 values")
    if not all(math.isfinite(joint) for joint in values):
        raise ValueError("joints must contain 7 finite values")
    return tuple(-joint if index % 2 == 0 else joint for index, joint in enumerate(values))  # type: ignore[return-value]


RIGHT_ARM_REFERENCE_INIT_JOINTS: tuple[float, float, float, float, float, float, float] = (
    0.0,
    0.0,
    0.0,
    -5.0,
    0.0,
    0.0,
    0.0,
)
LEFT_ARM_DEFAULT_INIT_JOINTS = mirror_right_to_left_joints(RIGHT_ARM_REFERENCE_INIT_JOINTS)
DEFAULT_JOINT_IMPEDANCE_K: tuple[float, float, float, float, float, float, float] = (
    8.0,
    8.0,
    8.0,
    4.0,
    2.0,
    1.5,
    1.0,
)
DEFAULT_JOINT_IMPEDANCE_D: tuple[float, float, float, float, float, float, float] = (
    0.8,
    0.8,
    0.8,
    0.6,
    0.4,
    0.3,
    0.2,
)
DCSS_CMD_ARM0_GET_DATA_6FT = 116


@dataclass(frozen=True)
class MotionConfig:
    """Runtime configuration for one real-robot chopping run.

    The defaults are intentionally conservative: SDK arm A, dry-run mode, low
    speed/acceleration ratios, and bounded vertical/lateral motion. Passing
    ``--execute`` is the switch that changes the script from planning/logging
    into sending commands to the real robot.
    """

    arm: str = "A"
    robot_ip: str = "192.168.1.190"
    sdk_root: Path = ROOT
    kine_config: Path = ROOT / "test" / "ccs_m6_40.MvKDCfg"
    control_hz: float = 250.0
    dz_mm: float = -20.0
    hold_s: float = 2.0
    cycles: int = 5
    lateral: bool = True
    lateral_mm: float = 10.0
    chop_axis: str = "y"
    lateral_axis: str = "x"
    lateral_phase: str = "separate"
    vel_ratio: int = 100
    acc_ratio: int = 100
    joint_k: tuple[float, float, float, float, float, float, float] = DEFAULT_JOINT_IMPEDANCE_K
    joint_d: tuple[float, float, float, float, float, float, float] = DEFAULT_JOINT_IMPEDANCE_D
    init_joints: tuple[float, float, float, float, float, float, float] | None = LEFT_ARM_DEFAULT_INIT_JOINTS
    init_timeout_s: float = 10.0
    init_tolerance_deg: float = 0.5
    command_mode: str = "pln-cart"
    execute: bool = False
    keep_enabled: bool = False
    trace_csv: Path | None = None
    print_trajectory: bool = False
    trajectory_stride: int = 1
    print_feedback: bool = False
    feedback_stride: int = 1
    force_feedback: bool = True
    print_force_feedback: bool = False
    force_feedback_stride: int = 1


@dataclass(frozen=True)
class PlannedSegment:
    """One planned Cartesian segment in xyzabc space.

    ``label`` is only for diagnostics. ``start_xyzabc`` and ``end_xyzabc`` are
    the Cartesian poses passed to the SDK MOVLA planner.
    """

    label: str
    start_xyzabc: np.ndarray
    end_xyzabc: np.ndarray


def parse_seven_floats(value: str, flag_name: str) -> tuple[float, float, float, float, float, float, float]:
    """Parse a strict seven-element comma-separated float tuple."""
    try:
        values = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"{flag_name} must be seven comma-separated floats") from exc
    if len(values) != 7:
        raise ValueError(f"{flag_name} must contain 7 values")
    if not all(math.isfinite(item) for item in values):
        raise ValueError(f"{flag_name} must contain 7 finite values")
    return values  # type: ignore[return-value]


def parse_joints(value: str) -> tuple[float, float, float, float, float, float, float] | None:
    """Parse the ``--init-joints`` CLI value.

    Returns seven finite joint angles in degrees, or ``None`` when the user
    passes ``none``/``skip`` to avoid initialization. This parser is strict so
    malformed commands fail before any robot connection or motion command.
    """
    if str(value).strip().lower() in ("", "none", "skip"):
        return None
    return parse_seven_floats(value, "--init-joints")


def parse_joint_k(value: str) -> tuple[float, float, float, float, float, float, float]:
    """Parse joint impedance stiffness values for ``--joint-k``."""
    return parse_seven_floats(value, "--joint-k")


def parse_joint_d(value: str) -> tuple[float, float, float, float, float, float, float]:
    """Parse joint impedance damping values for ``--joint-d``."""
    return parse_seven_floats(value, "--joint-d")


def parse_args(argv: list[str] | None = None) -> MotionConfig:
    """Build and validate a ``MotionConfig`` from command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("A", "B"), default="A", help="SDK arm selector; A is the left arm on this robot")
    parser.add_argument("--robot-ip", default="192.168.1.190")
    parser.add_argument("--sdk-root", type=Path, default=ROOT)
    parser.add_argument("--kine-config", type=Path, default=ROOT / "test" / "ccs_m6_40.MvKDCfg")
    parser.add_argument("--control-hz", type=float, default=250.0)
    parser.add_argument("--dz-mm", type=float, default=-20.0)
    parser.add_argument("--hold-s", type=float, default=2.0)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--lateral", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lateral-mm", type=float, default=10.0)
    parser.add_argument("--chop-axis", choices=("x", "y", "z"), default="y", help="Cartesian axis used for the down-up chopping stroke")
    parser.add_argument("--lateral-axis", choices=("x", "y", "z"), default="x", help="Cartesian axis used for lateral shift between cuts")
    parser.add_argument("--lateral-phase", choices=("separate", "retract"), default="separate", help="When to apply lateral shift: after retracting, or during retract like the MuJoCo demo")
    parser.add_argument("--vel-ratio", type=int, default=10)
    parser.add_argument("--acc-ratio", type=int, default=10)
    parser.add_argument("--joint-k", type=parse_joint_k, default=DEFAULT_JOINT_IMPEDANCE_K, help="Seven joint impedance stiffness values for --command-mode joint-impedance")
    parser.add_argument("--joint-d", type=parse_joint_d, default=DEFAULT_JOINT_IMPEDANCE_D, help="Seven joint impedance damping values in [0, 1] for --command-mode joint-impedance")
    parser.add_argument("--init-joints", type=parse_joints, default=LEFT_ARM_DEFAULT_INIT_JOINTS, help="Seven comma-separated left-arm initialization joints in degrees; use 'none' to skip")
    parser.add_argument("--init-timeout-s", type=float, default=10.0)
    parser.add_argument("--init-tolerance-deg", type=float, default=0.5)
    parser.add_argument("--command-mode", choices=("pln-cart", "position", "cart-impedance", "joint-impedance"), default="pln-cart")
    parser.add_argument("--execute", action="store_true", help="Send joint commands to the real robot")
    parser.add_argument("--keep-enabled", action="store_true", help="Do not disable the arm at the end")
    parser.add_argument("--trace-csv", type=Path, default=None)
    parser.add_argument("--print-trajectory", action="store_true", help="Print planned Cartesian and joint trajectory points to terminal")
    parser.add_argument("--trajectory-stride", type=int, default=1, help="Print every Nth planned trajectory point")
    parser.add_argument("--print-feedback", action="store_true", help="Print target/actual Cartesian feedback while running")
    parser.add_argument("--feedback-stride", type=int, default=1, help="Print every Nth feedback sample")
    parser.add_argument("--force-feedback", action=argparse.BooleanOptionalAction, default=True, help="Subscribe and record DCSS user-specified 6-axis force feedback")
    parser.add_argument("--print-force-feedback", action="store_true", help="Print DCSS 6-axis force feedback while running")
    parser.add_argument("--force-feedback-stride", type=int, default=1, help="Print every Nth force feedback sample")
    config = MotionConfig(**vars(parser.parse_args(argv)))
    validate_config(config)
    return config


def validate_config(config: MotionConfig) -> None:
    """Reject unsafe or internally inconsistent configuration values.

    This is the main software safety gate before connecting to the robot. It
    limits vertical/lateral travel, checks velocity/acceleration ratios, and
    ensures timing/tolerance values are positive.
    """
    if config.arm not in ("A", "B"):
        raise ValueError("arm must be 'A' or 'B'")
    if config.control_hz <= 0.0:
        raise ValueError("control_hz must be positive")
    if config.hold_s <= 0.0:
        raise ValueError("hold_s must be positive")
    if config.cycles <= 0:
        raise ValueError("cycles must be positive")
    if abs(float(config.dz_mm)) > 80.0:
        raise ValueError("dz-mm magnitude must be <= 80")
    if abs(float(config.lateral_mm)) > 50.0:
        raise ValueError("lateral-mm magnitude must be <= 50")
    if not (0 <= int(config.vel_ratio) <= 100):
        raise ValueError("vel-ratio must be in [0, 100]")
    if not (0 <= int(config.acc_ratio) <= 100):
        raise ValueError("acc-ratio must be in [0, 100]")
    if config.init_joints is not None and len(config.init_joints) != 7:
        raise ValueError("init-joints must contain 7 values")
    if config.init_timeout_s <= 0.0:
        raise ValueError("init-timeout-s must be positive")
    if config.init_tolerance_deg <= 0.0:
        raise ValueError("init-tolerance-deg must be positive")
    if config.command_mode not in ("pln-cart", "position", "cart-impedance", "joint-impedance"):
        raise ValueError("command-mode must be 'pln-cart', 'position', 'cart-impedance', or 'joint-impedance'")
    if len(config.joint_k) != 7 or len(config.joint_d) != 7:
        raise ValueError("joint-k and joint-d must contain 7 values")
    if not all(math.isfinite(value) and value >= 0.0 for value in config.joint_k):
        raise ValueError("joint-k values must be finite and non-negative")
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in config.joint_d):
        raise ValueError("joint-d values must be finite and in [0, 1]")
    axis_index(config.chop_axis)
    axis_index(config.lateral_axis)
    if config.chop_axis == config.lateral_axis and config.lateral:
        raise ValueError("chop-axis and lateral-axis must differ when lateral motion is enabled")
    if config.lateral_phase not in ("separate", "retract"):
        raise ValueError("lateral-phase must be 'separate' or 'retract'")
    if int(config.trajectory_stride) <= 0:
        raise ValueError("trajectory-stride must be positive")
    if int(config.feedback_stride) <= 0:
        raise ValueError("feedback-stride must be positive")
    if int(config.force_feedback_stride) <= 0:
        raise ValueError("force-feedback-stride must be positive")


def axis_index(axis: str) -> int:
    """Return xyzabc index for a Cartesian translation axis."""
    axis_map = {"x": 0, "y": 1, "z": 2}
    try:
        return axis_map[str(axis).lower()]
    except KeyError as exc:
        raise ValueError("axis must be one of 'x', 'y', or 'z'") from exc


def chop_delta_mm(config: MotionConfig) -> float:
    """Return the signed Cartesian down-stroke displacement in millimeters.

    On the real robot frame used here, +Y is physically downward. Operators
    still tend to pass ``--dz-mm`` as a negative depth because the MuJoCo demo
    used Z as vertical, so for the default Y chopping axis we treat the CLI
    value as a magnitude and make the downward stroke positive. Other axes keep
    the explicit sign for diagnostic experiments.
    """
    dz = float(config.dz_mm)
    if str(config.chop_axis).lower() == "y":
        return abs(dz)
    return dz


def cut_cycle_progress(cycle_step: int, steps_per_cycle: int, lateral_phase: str = "separate") -> tuple[float, float, float]:
    """Return normalized down/retract/shift progress for one sampled cycle.

    ``separate`` splits a cycle into descend, retract, then horizontal shift.
    ``retract`` keeps the original MuJoCo-style behavior where lateral shift
    happens while retracting upward.
    """
    if lateral_phase == "retract":
        descent_steps = max(1, int(steps_per_cycle) // 2)
        retract_steps = max(1, int(steps_per_cycle) - descent_steps)
        if int(cycle_step) < descent_steps:
            denominator = max(descent_steps - 1, 1)
            return float(cycle_step) / float(denominator), 0.0, 0.0

        retract_step = min(int(cycle_step) - descent_steps, retract_steps - 1)
        denominator = max(retract_steps - 1, 1)
        retract_progress = float(retract_step) / float(denominator)
        return 1.0 - retract_progress, retract_progress, retract_progress

    vertical_steps = max(2, int(round(float(steps_per_cycle) * 2.0 / 3.0)))
    shift_steps = max(1, int(steps_per_cycle) - vertical_steps)
    descent_steps = max(1, vertical_steps // 2)
    retract_steps = max(1, vertical_steps - descent_steps)
    if int(cycle_step) < descent_steps:
        denominator = max(descent_steps - 1, 1)
        return float(cycle_step) / float(denominator), 0.0, 0.0
    if int(cycle_step) < vertical_steps:
        retract_step = min(int(cycle_step) - descent_steps, retract_steps - 1)
        denominator = max(retract_steps - 1, 1)
        retract_progress = float(retract_step) / float(denominator)
        return 1.0 - retract_progress, retract_progress, 0.0

    if shift_steps == 1:
        return 0.0, 1.0, 1.0
    shift_step = min(int(cycle_step) - vertical_steps, shift_steps - 1)
    shift_progress = float(shift_step) / float(shift_steps - 1)
    return 0.0, 1.0, shift_progress


def build_relative_targets(start_pose_mm: np.ndarray, config: MotionConfig) -> list[np.ndarray]:
    """Generate per-control-tick Cartesian targets for sampled command modes.

    Starting from ``start_pose_mm`` (xyzabc), the function creates a list of
    target poses over ``cycles``. By default Y moves down and back up each
    cycle because +Y is downward on the real robot frame; X shifts between
    cycles when ``config.lateral`` is enabled. Orientation values a/b/c are
    held constant.
    """
    start = np.asarray(start_pose_mm, dtype=float).reshape(6)
    steps_per_cycle = max(3, int(round(float(config.hold_s) * float(config.control_hz))))
    cycle_count = max(1, int(config.cycles))
    chop_delta = chop_delta_mm(config)
    lateral_step = abs(float(config.lateral_mm)) if config.lateral else 0.0
    chop_idx = axis_index(config.chop_axis)
    lateral_idx = axis_index(config.lateral_axis)

    targets: list[np.ndarray] = []
    for step_index in range(steps_per_cycle * cycle_count):
        cycle_index = min(step_index // steps_per_cycle, cycle_count - 1)
        cycle_step = step_index - cycle_index * steps_per_cycle
        z_progress, _retract_progress, shift_progress = cut_cycle_progress(
            cycle_step, steps_per_cycle, config.lateral_phase
        )
        target = start.copy()
        target[chop_idx] = start[chop_idx] + chop_delta * z_progress
        target[lateral_idx] = start[lateral_idx] + lateral_step * (float(cycle_index) + shift_progress)
        targets.append(target)
    return targets


def build_chop_segments(start_pose_mm: np.ndarray, config: MotionConfig) -> list[PlannedSegment]:
    """Build coarse MOVLA segments for the default planned Cartesian path.

    Each cycle contributes a ``descend`` segment and a ``retract`` segment.
    Between cycles, an optional ``shift`` segment advances along the lateral
    axis before the next descend. These segments are later handed to
    ``kine.movLA`` and then sent with ``robot.setPln_Cart`` when ``--execute``
    is set.
    """
    up_pose = np.asarray(start_pose_mm, dtype=float).reshape(6).copy()
    chop_delta = chop_delta_mm(config)
    lateral_step = abs(float(config.lateral_mm)) if config.lateral else 0.0
    chop_idx = axis_index(config.chop_axis)
    lateral_idx = axis_index(config.lateral_axis)
    segments: list[PlannedSegment] = []

    for cycle_index in range(max(1, int(config.cycles))):
        down_pose = up_pose.copy()
        down_pose[chop_idx] = up_pose[chop_idx] + chop_delta
        segments.append(PlannedSegment("descend", up_pose.copy(), down_pose.copy()))
        segments.append(PlannedSegment("retract", down_pose.copy(), up_pose.copy()))
        if lateral_step and cycle_index < int(config.cycles) - 1:
            next_up = up_pose.copy()
            next_up[lateral_idx] += lateral_step
            segments.append(PlannedSegment("shift", up_pose.copy(), next_up.copy()))
            up_pose = next_up

    return segments


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint that runs the debug routine and prints a short summary."""
    try:
        config = parse_args(argv)
        result = run_real_debug(config)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print("execute:", result["execute"])
    print("command_mode:", result["command_mode"])
    print("initialized:", result["initialized"])
    print("ik_success:", result["ik_success"])
    print("steps:", result["steps"])
    print("start_joints:", [round(v, 3) for v in result["start_joints"]])
    print("final_joints:", [round(v, 3) for v in result["final_joints"]])
    if result["trace_csv"]:
        print("trace_csv:", result["trace_csv"])
    return 0


def run_real_debug(config: MotionConfig) -> dict:
    """Connect to the robot, run initialization/planning/execution, and report.

    This is the central orchestration function. It selects SDK arm index 0 for
    arm A and 1 for arm B, initializes the kinematics model, optionally sends
    the initialization joint command, computes the starting FK pose, dispatches
    the selected command mode, writes trace CSV rows, and disables/releases the
    robot in ``finally`` unless ``keep_enabled`` is requested.
    """
    validate_config(config)
    _ensure_sdk_path(config.sdk_root)
    from SDK_PYTHON.fx_kine import FX_InvKineSolvePara, Marvin_Kine
    from SDK_PYTHON.fx_robot import DCSS, Marvin_Robot

    arm_index = 0 if config.arm == "A" else 1
    dcss = DCSS()
    robot = Marvin_Robot()
    kine = Marvin_Kine()
    trace_rows: list[dict] = []
    last_target_joints: list[float] | None = None
    connected = False
    force_feedback_configured = False

    try:
        connected = bool(robot.connect(config.robot_ip))
        if not connected:
            raise RuntimeError("failed to connect to the robot")
        robot.check_error_and_clear(dcss)
        _verify_frame_updates(robot, dcss, arm_index)
        robot.log_switch("1")
        robot.local_log_switch("1")
        force_feedback_configured = _configure_force_feedback(robot, config)

        kine.log_switch(0)
        kine_config = kine.load_config(arm_type=arm_index, config_path=str(config.kine_config))
        if not kine.initial_kine(
            robot_type=kine_config["TYPE"][arm_index],
            dh=kine_config["DH"][arm_index],
            pnva=kine_config["PNVA"][arm_index],
            j67=kine_config["BD"][arm_index],
        ):
            raise RuntimeError("initial_kine failed")

        feedback = robot.subscribe(dcss)
        current_joints = list(feedback["outputs"][arm_index]["fb_joint_pos"])
        initialized = False
        if config.init_joints is not None:
            init_target = [float(value) for value in config.init_joints]
            if config.execute:
                _set_position_mode(robot, config)
                robot.clear_set()
                robot.set_joint_cmd_pose(arm=config.arm, joints=init_target)
                _send_cmd_prefer_wait(robot)
                current_joints, initialized = _wait_until_reached(
                    robot, dcss, arm_index, init_target, config.init_timeout_s, config.init_tolerance_deg,
                )
                if not initialized:
                    feedback = robot.subscribe(dcss)
                    raise RuntimeError(_initialization_failure_message(init_target, current_joints, feedback, arm_index))
            else:
                current_joints = init_target
                initialized = True

        current_pose = np.asarray(kine.mat4x4_to_xyzabc(kine.fk(current_joints)), dtype=float)
        start_joints = list(current_joints)
        if config.command_mode == "pln-cart":
            _set_pln_cart_position_mode(robot, config, dcss, arm_index)
            trace_rows, last_target_joints, actual_joints = _run_pln_cart_chop(
                robot, kine, dcss, arm_index, current_pose, start_joints, config,
            )
        else:
            targets = build_relative_targets(current_pose, config)
            if config.command_mode == "cart-impedance":
                _configure_cartesian_impedance(robot, config, current_pose)
            elif config.command_mode == "joint-impedance":
                _configure_joint_impedance(robot, config)
            else:
                _set_position_mode(robot, config)

            ref_joints = list(current_joints)
            period_s = 1.0 / float(config.control_hz)
            trace_start_time_s = time.time()
            for step_index, target_xyzabc in enumerate(targets):
                target_matrix = kine.xyzabc_to_mat4x4(target_xyzabc.tolist())
                ik_para = FX_InvKineSolvePara()
                ik_para.set_input_ik_target_tcp(kine.mat4x4_to_mat1x16(target_matrix))
                ik_para.set_input_ik_ref_joint(ref_joints)
                ik_para.set_input_ik_zsp_type(0)
                ik = kine.ik(ik_para)
                ik_success = ik.get_output_result_num() >= 1
                target_joints = list(ik.get_output_ret_joint())
                if config.print_trajectory and step_index % int(config.trajectory_stride) == 0:
                    _print_sampled_trajectory_row(
                        _sampled_trajectory_row(step_index, target_xyzabc, target_joints, ik_success)
                    )
                if ik_success:
                    ref_joints = target_joints
                    last_target_joints = target_joints
                    if config.execute:
                        _send_sampled_joint_command(robot, config, target_joints)

                feedback = robot.subscribe(dcss)
                sample_time_s = time.time()
                actual_joints = list(feedback["outputs"][arm_index]["fb_joint_pos"])
                actual_joint_velocities = list(feedback["outputs"][arm_index].get("fb_joint_vel", []))
                actual_joint_torques = list(feedback["outputs"][arm_index].get("fb_joint_sToq", []))
                actual_pose = np.asarray(kine.mat4x4_to_xyzabc(kine.fk(actual_joints)), dtype=float)
                if config.print_feedback and step_index % int(config.feedback_stride) == 0:
                    _print_feedback_row(step_index, target_xyzabc, actual_pose)
                force_feedback = _force_feedback_from_feedback(feedback, arm_index) if config.force_feedback else None
                if (
                    config.print_force_feedback
                    and force_feedback is not None
                    and step_index % int(config.force_feedback_stride) == 0
                ):
                    _print_force_feedback_row(step_index, force_feedback)
                trace_rows.append(
                    _trace_row(
                        step_index,
                        target_xyzabc,
                        actual_pose,
                        target_joints,
                        actual_joints,
                        ik_success,
                        timestamp_s=sample_time_s,
                        elapsed_s=sample_time_s - trace_start_time_s,
                        actual_joint_velocities=actual_joint_velocities,
                        actual_joint_torques=actual_joint_torques,
                        force_feedback=force_feedback,
                    )
                )
                time.sleep(period_s)

        if config.trace_csv is not None:
            _write_trace_csv(config.trace_csv, trace_rows)
        return {
            "ik_success": all(row["ik_success"] for row in trace_rows),
            "steps": len(trace_rows),
            "last_target_joints": last_target_joints,
            "trace_csv": str(config.trace_csv) if config.trace_csv else "",
            "execute": config.execute,
            "command_mode": config.command_mode,
            "initialized": initialized,
            "force_feedback_configured": force_feedback_configured,
            "start_joints": start_joints,
            "final_joints": list(actual_joints) if trace_rows else start_joints,
        }
    finally:
        if connected and not config.keep_enabled:
            try:
                robot.clear_set()
                robot.set_state(arm=config.arm, state=0)
                robot.send_cmd()
            except Exception as exc:
                print(f"warning: failed to disable arm {config.arm}: {exc}", file=sys.stderr)
        try:
            robot.release_robot()
        except Exception:
            pass


def _ensure_sdk_path(sdk_root: Path) -> None:
    """Add the SDK/project root to ``sys.path`` before importing SDK modules."""
    root = Path(sdk_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _verify_frame_updates(robot, dcss, arm_index: int) -> None:
    """Confirm feedback frames are updating for the selected arm.

    A real robot run should not continue if feedback is stale, because later
    initialization checks and trajectory completion checks depend on fresh
    controller feedback.
    """
    observed = set()
    for _ in range(5):
        data = robot.subscribe(dcss)
        observed.add(data["outputs"][arm_index]["frame_serial"])
        time.sleep(0.01)
    if len(observed - {0}) == 0:
        raise RuntimeError("robot feedback frame did not update")


def _set_position_mode(robot, config: MotionConfig) -> None:
    """Put the selected arm into joint position mode with configured limits.

    This mode is used before sending the initialization joint pose and by the
    sampled ``position`` path.
    """
    robot.clear_set()
    robot.set_vel_acc(arm=config.arm, velRatio=int(config.vel_ratio), AccRatio=int(config.acc_ratio))
    _send_cmd_prefer_wait(robot)
    time.sleep(0.1)

    robot.clear_set()
    robot.set_state(arm=config.arm, state=1)
    _send_cmd_prefer_wait(robot)
    time.sleep(0.2)


def _feedback_joints(robot, dcss, arm_index: int) -> list[float]:
    """Read the current feedback joint angles for the selected SDK arm index."""
    data = robot.subscribe(dcss)
    return [float(value) for value in data["outputs"][arm_index]["fb_joint_pos"]]


def _wait_until_reached(
    robot,
    dcss,
    arm_index: int,
    target: list[float],
    timeout_s: float,
    tolerance_deg: float,
) -> tuple[list[float], bool]:
    """Poll feedback until the arm reaches a joint target or times out.

    Returns the final observed joints and whether the maximum joint error was
    within ``tolerance_deg`` before ``timeout_s`` elapsed.
    """
    deadline = time.monotonic() + float(timeout_s)
    final = _feedback_joints(robot, dcss, arm_index)
    while time.monotonic() < deadline:
        final = _feedback_joints(robot, dcss, arm_index)
        error = max(abs(float(a) - float(t)) for a, t in zip(final, target))
        if error <= float(tolerance_deg):
            return final, True
        time.sleep(0.05)
    return final, False


def _send_cmd_prefer_wait(robot, timeout_ms: int = 100) -> object:
    """Send a command, preferring the SDK's wait-for-response API when present."""
    if hasattr(robot, "send_cmd_wait_response"):
        return robot.send_cmd_wait_response(int(timeout_ms))
    return robot.send_cmd()


def _initialization_failure_message(target: Sequence[float], final: Sequence[float], feedback: dict, arm_index: int) -> str:
    """Build a diagnostic error string when initialization fails to reach target."""
    errors = [abs(float(actual) - float(expected)) for actual, expected in zip(final, target)]
    max_error = max(errors) if errors else float("nan")
    state = feedback.get("states", [{}])[arm_index] if feedback.get("states") else {}
    output = feedback.get("outputs", [{}])[arm_index] if feedback.get("outputs") else {}
    rounded_final = [round(float(value), 3) for value in final]
    rounded_target = [round(float(value), 3) for value in target]
    return (
        "initialization target was not reached; "
        f"max_error_deg={max_error:.3f}; "
        f"cur_state={state.get('cur_state')}; "
        f"err_code={state.get('err_code')}; "
        f"traj_state={output.get('traj_state')!r}; "
        f"target_joints={rounded_target}; "
        f"final_joints={rounded_final}"
    )


def _set_pln_cart_position_mode(robot, config: MotionConfig, dcss=None, arm_index: int | None = None) -> None:
    """Prepare the selected arm for planned Cartesian position commands.

    The function sets velocity/acceleration ratios, enables position planning
    state, and optionally verifies from controller feedback that the state
    transition completed before ``setPln_Cart`` is used.
    """
    robot.clear_set()
    robot.set_vel_acc(arm=config.arm, velRatio=int(config.vel_ratio), AccRatio=int(config.acc_ratio))
    if hasattr(robot, "send_cmd_wait_response"):
        robot.send_cmd_wait_response(100)
    else:
        robot.send_cmd()
    time.sleep(0.1)

    robot.clear_set()
    robot.set_state(arm=config.arm, state=1)
    if hasattr(robot, "send_cmd_wait_response"):
        robot.send_cmd_wait_response(100)
    else:
        robot.send_cmd()
    time.sleep(0.2)
    if dcss is not None and arm_index is not None:
        _wait_until_planning_state(robot, dcss, int(arm_index), config.arm)


def _wait_until_planning_state(robot, dcss, arm_index: int, arm: str, timeout_s: float = 1.0) -> None:
    """Wait until controller feedback reports position planning state."""
    deadline = time.monotonic() + float(timeout_s)
    feedback = robot.subscribe(dcss)
    while time.monotonic() < deadline:
        feedback = robot.subscribe(dcss)
        state = feedback.get("states", [{}])[arm_index]
        if state.get("cur_state") == 1:
            return
        time.sleep(0.02)
    state = feedback.get("states", [{}])[arm_index] if feedback.get("states") else {}
    output = feedback.get("outputs", [{}])[arm_index] if feedback.get("outputs") else {}
    raise RuntimeError(
        f"{arm} arm did not enter position planning mode before setPln_Cart; "
        f"cur_state={state.get('cur_state')}; "
        f"cmd_state={state.get('cmd_state')}; "
        f"err_code={state.get('err_code')}; "
        f"traj_state={output.get('traj_state')!r}"
    )


def _run_pln_cart_chop(
    robot,
    kine,
    dcss,
    arm_index: int,
    start_pose: np.ndarray,
    start_joints: list[float],
    config: MotionConfig,
) -> tuple[list[dict], list[float] | None, list[float]]:
    """Plan and optionally execute the coarse Cartesian chopping segments.

    For every segment from ``build_chop_segments``, MOVLA produces a trajectory
    package (``pset``). Dry-run mode records the planned final joints without
    commanding the robot. Execute mode sends each ``pset`` to the selected arm
    and waits for the trajectory state to return idle.
    """
    rows: list[dict] = []
    ref_joints = list(start_joints)
    actual_joints = list(start_joints)
    last_target_joints: list[float] | None = None

    for segment_index, segment in enumerate(build_chop_segments(start_pose, config)):
        points, pset = kine.movLA(
            start_xyzabc=segment.start_xyzabc.tolist(),
            end_xyzabc=segment.end_xyzabc.tolist(),
            ref_joints=ref_joints,
            vel=max(0.1, min(1000.0, abs(float(config.dz_mm)) / max(float(config.hold_s) * 0.5, 0.1))),
            acc=100.0,
            freq_hz=int(config.control_hz),
        )
        if pset is None:
            raise RuntimeError(f"MOVLA planning failed for segment {segment.label}")
        target_joints = list(points[-1])[:7] if points else list(ref_joints)
        last_target_joints = target_joints
        if config.print_trajectory:
            _print_planned_trajectory(kine, segment_index, segment, points, int(config.trajectory_stride))

        if config.execute:
            robot.setPln_Cart(arm=config.arm, pset=pset)
            _wait_until_traj_idle(robot, dcss, arm_index, timeout_s=max(float(config.hold_s) * 3.0, 3.0))
            actual_joints = _feedback_joints(robot, dcss, arm_index)
        else:
            actual_joints = target_joints
        actual_pose = np.asarray(kine.mat4x4_to_xyzabc(kine.fk(actual_joints)), dtype=float)
        if config.print_feedback:
            _print_feedback_row(segment_index, segment.end_xyzabc, actual_pose)
        rows.append(_trace_row(segment_index, segment.end_xyzabc, actual_pose, target_joints, actual_joints, True))
        ref_joints = list(actual_joints)

    return rows, last_target_joints, actual_joints


def _planned_trajectory_rows(
    kine,
    segment_index: int,
    segment: PlannedSegment,
    points: Sequence[Sequence[float]],
    stride: int,
) -> list[dict]:
    """Convert planned MOVLA joint points into printable Cartesian/joint rows."""
    rows: list[dict] = []
    step = max(1, int(stride))
    for point_index, point in enumerate(points):
        if point_index % step != 0 and point_index != len(points) - 1:
            continue
        joints = [float(value) for value in list(point)[:7]]
        xyzabc = np.asarray(kine.mat4x4_to_xyzabc(kine.fk(joints)), dtype=float).reshape(6)
        row = {
            "segment": int(segment_index),
            "label": segment.label,
            "point": int(point_index),
            "x": float(xyzabc[0]),
            "y": float(xyzabc[1]),
            "z": float(xyzabc[2]),
            "a": float(xyzabc[3]),
            "b": float(xyzabc[4]),
            "c": float(xyzabc[5]),
        }
        for joint_index in range(7):
            row[f"q{joint_index}"] = joints[joint_index] if joint_index < len(joints) else 0.0
        rows.append(row)
    return rows


def _print_planned_trajectory(
    kine,
    segment_index: int,
    segment: PlannedSegment,
    points: Sequence[Sequence[float]],
    stride: int,
) -> None:
    """Print one planned MOVLA segment as CSV-like rows to terminal."""
    print(
        "planned_trajectory_header: "
        "segment,label,point,x,y,z,a,b,c,q0,q1,q2,q3,q4,q5,q6"
    )
    for row in _planned_trajectory_rows(kine, segment_index, segment, points, stride):
        values = [
            str(row["segment"]),
            str(row["label"]),
            str(row["point"]),
            *[f"{float(row[name]):.6f}" for name in ("x", "y", "z", "a", "b", "c")],
            *[f"{float(row[f'q{joint_index}']):.6f}" for joint_index in range(7)],
        ]
        print("planned_trajectory: " + ",".join(values))


def _wait_until_traj_idle(robot, dcss, arm_index: int, timeout_s: float) -> None:
    """Block until the planned Cartesian trajectory finishes or times out."""
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        data = robot.subscribe(dcss)
        if data["outputs"][arm_index]["traj_state"] == b"\x00":
            return
        time.sleep(0.001)
    raise RuntimeError("planned Cartesian trajectory did not finish before timeout")


def _send_sampled_joint_command(robot, config: MotionConfig, target_joints: Sequence[float]) -> None:
    """Send one sampled IK joint target using the selected SDK command path."""
    joints = [float(value) for value in target_joints]
    robot.clear_set()
    if config.command_mode == "joint-impedance" and hasattr(robot, "set_joint_position_cmd"):
        robot.set_joint_position_cmd(config.arm, joints)
    else:
        robot.set_joint_cmd_pose(arm=config.arm, joints=joints)
    robot.send_cmd()


def _configure_joint_impedance(robot, config: MotionConfig) -> None:
    """Configure SDK joint impedance mode for sampled IK joint targets.

    Newer SDK wrappers expose ``set_imp_joint_state``. The real controller SDK
    in this workspace exposes the lower-level sequence instead: torque state,
    impedance type 1, velocity/acceleration limits, then joint K/D parameters.
    """
    joint_k = list(config.joint_k)
    joint_d = list(config.joint_d)
    if hasattr(robot, "set_imp_joint_state"):
        robot.clear_set()
        ok = robot.set_imp_joint_state(
            arm=config.arm,
            velRatio=int(config.vel_ratio),
            AccRatio=int(config.acc_ratio),
            K=joint_k,
            D=joint_d,
        )
        if ok is False:
            raise RuntimeError("failed to configure joint impedance mode")
        _send_cmd_prefer_wait(robot)
        time.sleep(0.2)
        return

    robot.clear_set()
    robot.set_state(arm=config.arm, state=3)
    robot.set_impedance_type(arm=config.arm, type=1)
    robot.set_vel_acc(arm=config.arm, velRatio=int(config.vel_ratio), AccRatio=int(config.acc_ratio))
    _send_cmd_prefer_wait(robot)
    time.sleep(0.2)

    robot.clear_set()
    ok = robot.set_joint_kd_params(arm=config.arm, K=joint_k, D=joint_d)
    if ok is False:
        raise RuntimeError("failed to configure joint impedance K/D parameters")
    _send_cmd_prefer_wait(robot)
    time.sleep(0.2)


def _configure_cartesian_impedance(robot, config: MotionConfig, current_pose: np.ndarray) -> None:
    """Configure the SDK Cartesian impedance mode for sampled impedance tests.

    This path is not used by the default wrapper script, but remains available
    with ``--command-mode cart-impedance``. It enables impedance state, sets
    Cartesian stiffness/damping, and configures the end-effector control frame
    from the current pose orientation.
    """
    cart_dir = [float(current_pose[3]), float(current_pose[4]), float(current_pose[5]), 0.0, 0.0, 0.0, 0.0]
    robot.clear_set()
    robot.set_state(arm=config.arm, state=3)
    robot.set_impedance_type(arm=config.arm, type=2)
    robot.set_vel_acc(arm=config.arm, velRatio=int(config.vel_ratio), AccRatio=int(config.acc_ratio))
    robot.send_cmd()
    time.sleep(0.2)

    robot.clear_set()
    robot.set_cart_kd_params(
        arm=config.arm,
        K=[10.0, 5000.0, 5000.0, 600.0, 600.0, 600.0, 20.0],
        D=[0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 1.0],
        type=2,
    )
    robot.send_cmd()
    time.sleep(0.2)

    robot.clear_set()
    robot.set_EefCart_control_params(arm=config.arm, fcType=1, CartCtrlPara=cart_dir)
    robot.send_cmd()
    time.sleep(0.2)


def _configure_force_feedback(robot, config: MotionConfig) -> bool:
    """Select the DCSS user-specified data channel for A-arm 6-axis force."""
    if not config.force_feedback:
        return False
    setter = getattr(robot, "set_user_specified_data", None)
    if setter is None:
        print(
            "warning: robot SDK has no set_user_specified_data; "
            "recording est_joint_firc_dot without switching DCSS user data channel",
            file=sys.stderr,
        )
        return False
    ok = setter(config.arm, DCSS_CMD_ARM0_GET_DATA_6FT)
    if ok is False:
        raise RuntimeError("failed to configure DCSS 6-axis force feedback channel")
    return True


def _force_feedback_from_feedback(feedback: dict, arm_index: int) -> tuple[float, float, float, float, float, float] | None:
    """Extract the six-axis force/torque values mapped by set_user_specified_data."""
    try:
        values = feedback["outputs"][arm_index]["est_joint_firc_dot"][:6]
    except (KeyError, IndexError, TypeError):
        return None
    if len(values) < 6:
        return None
    return tuple(float(value) for value in values[:6])  # type: ignore[return-value]


def _print_force_feedback_row(step_index: int, force_feedback: Sequence[float]) -> None:
    """Print one six-axis force/torque feedback sample."""
    fx, fy, fz, tx, ty, tz = [float(value) for value in list(force_feedback)[:6]]
    print(
        "wrench_6ft: "
        f"step={int(step_index)},"
        f"fx={fx:.6f},fy={fy:.6f},fz={fz:.6f},"
        f"tx={tx:.6f},ty={ty:.6f},tz={tz:.6f}"
    )


def _print_feedback_row(step_index: int, target_xyzabc: np.ndarray, actual_xyzabc: np.ndarray) -> None:
    """Print target/actual Cartesian feedback for live debugging."""
    target = np.asarray(target_xyzabc, dtype=float).reshape(6)
    actual = np.asarray(actual_xyzabc, dtype=float).reshape(6)
    print(
        "feedback: "
        f"step={int(step_index)},"
        f"target_y={target[1]:.6f},actual_y={actual[1]:.6f},"
        f"target_z={target[2]:.6f},actual_z={actual[2]:.6f},"
        f"target_x={target[0]:.6f},actual_x={actual[0]:.6f}"
    )


def _sampled_trajectory_row(
    step_index: int,
    target_xyzabc: np.ndarray,
    target_joints: Sequence[float],
    ik_success: bool,
) -> dict:
    """Create one printable sampled IK trajectory row."""
    target = np.asarray(target_xyzabc, dtype=float).reshape(6)
    row = {"step": int(step_index), "ik_success": bool(ik_success)}
    for i, name in enumerate(("x", "y", "z", "a", "b", "c")):
        row[name] = float(target[i])
    for i in range(7):
        row[f"q{i}"] = float(target_joints[i]) if i < len(target_joints) else 0.0
    return row


def _print_sampled_trajectory_row(row: dict) -> None:
    """Print one sampled IK target as a CSV-like terminal row."""
    print("sampled_trajectory_header: step,ik_success,x,y,z,a,b,c,q0,q1,q2,q3,q4,q5,q6")
    values = [
        str(row["step"]),
        str(row["ik_success"]),
        *[f"{float(row[name]):.6f}" for name in ("x", "y", "z", "a", "b", "c")],
        *[f"{float(row[f'q{joint_index}']):.6f}" for joint_index in range(7)],
    ]
    print("sampled_trajectory: " + ",".join(values))


def _trace_row(
    step_index: int,
    target_xyzabc: np.ndarray,
    actual_xyzabc: np.ndarray,
    target_joints: Sequence[float],
    actual_joints: Sequence[float],
    ik_success: bool,
    timestamp_s: float | None = None,
    elapsed_s: float | None = None,
    actual_joint_velocities: Sequence[float] | None = None,
    actual_joint_torques: Sequence[float] | None = None,
    force_feedback: Sequence[float] | None = None,
) -> dict:
    """Create one CSV row comparing target pose/joints with feedback pose/joints."""
    row = {"step": int(step_index), "ik_success": bool(ik_success)}
    if timestamp_s is not None:
        row["timestamp_s"] = float(timestamp_s)
    if elapsed_s is not None:
        row["elapsed_s"] = float(elapsed_s)
    for i, name in enumerate(("x", "y", "z", "a", "b", "c")):
        row[f"target_{name}"] = float(target_xyzabc[i])
        row[f"actual_{name}"] = float(actual_xyzabc[i])
    velocities = list(actual_joint_velocities or [])
    torques = list(actual_joint_torques or [])
    for i in range(7):
        row[f"target_q_{i}"] = float(target_joints[i]) if i < len(target_joints) else 0.0
        row[f"actual_q_{i}"] = float(actual_joints[i]) if i < len(actual_joints) else 0.0
        row[f"actual_qd_{i}"] = float(velocities[i]) if i < len(velocities) else 0.0
        row[f"actual_tau_{i}"] = float(torques[i]) if i < len(torques) else 0.0
    if force_feedback is not None:
        for name, value in zip(("force_fx", "force_fy", "force_fz", "torque_tx", "torque_ty", "torque_tz"), force_feedback):
            row[name] = float(value)
    return row


def _write_trace_csv(path: Path, rows: list[dict]) -> None:
    """Write collected trace rows to CSV, creating the parent directory if needed."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
