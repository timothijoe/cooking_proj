#!/usr/bin/env python3
"""Real left-arm IK + Cartesian impedance lateral down-up debug script."""

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


@dataclass(frozen=True)
class MotionConfig:
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
    vel_ratio: int = 10
    acc_ratio: int = 10
    init_joints: tuple[float, float, float, float, float, float, float] | None = LEFT_ARM_DEFAULT_INIT_JOINTS
    init_timeout_s: float = 10.0
    init_tolerance_deg: float = 0.5
    command_mode: str = "pln-cart"
    execute: bool = False
    keep_enabled: bool = False
    trace_csv: Path | None = None


@dataclass(frozen=True)
class PlannedSegment:
    label: str
    start_xyzabc: np.ndarray
    end_xyzabc: np.ndarray


def parse_joints(value: str) -> tuple[float, float, float, float, float, float, float] | None:
    if str(value).strip().lower() in ("", "none", "skip"):
        return None
    try:
        joints = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("--init-joints must be seven comma-separated floats") from exc
    if len(joints) != 7:
        raise ValueError("--init-joints must contain 7 values")
    if not all(math.isfinite(joint) for joint in joints):
        raise ValueError("--init-joints must contain 7 finite values")
    return joints  # type: ignore[return-value]


def parse_args(argv: list[str] | None = None) -> MotionConfig:
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
    parser.add_argument("--vel-ratio", type=int, default=10)
    parser.add_argument("--acc-ratio", type=int, default=10)
    parser.add_argument("--init-joints", type=parse_joints, default=LEFT_ARM_DEFAULT_INIT_JOINTS, help="Seven comma-separated left-arm initialization joints in degrees; use 'none' to skip")
    parser.add_argument("--init-timeout-s", type=float, default=10.0)
    parser.add_argument("--init-tolerance-deg", type=float, default=0.5)
    parser.add_argument("--command-mode", choices=("pln-cart", "position", "cart-impedance"), default="pln-cart")
    parser.add_argument("--execute", action="store_true", help="Send joint commands to the real robot")
    parser.add_argument("--keep-enabled", action="store_true", help="Do not disable the arm at the end")
    parser.add_argument("--trace-csv", type=Path, default=None)
    config = MotionConfig(**vars(parser.parse_args(argv)))
    validate_config(config)
    return config


def validate_config(config: MotionConfig) -> None:
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
    if config.command_mode not in ("pln-cart", "position", "cart-impedance"):
        raise ValueError("command-mode must be 'pln-cart', 'position', or 'cart-impedance'")


def cut_cycle_progress(cycle_step: int, steps_per_cycle: int) -> tuple[float, float]:
    descent_steps = max(1, int(steps_per_cycle) // 2)
    retract_steps = max(1, int(steps_per_cycle) - descent_steps)
    if int(cycle_step) < descent_steps:
        denominator = max(descent_steps - 1, 1)
        return float(cycle_step) / float(denominator), 0.0

    retract_step = min(int(cycle_step) - descent_steps, retract_steps - 1)
    denominator = max(retract_steps - 1, 1)
    retract_progress = float(retract_step) / float(denominator)
    return 1.0 - retract_progress, retract_progress


def build_relative_targets(start_pose_mm: np.ndarray, config: MotionConfig) -> list[np.ndarray]:
    start = np.asarray(start_pose_mm, dtype=float).reshape(6)
    steps_per_cycle = max(3, int(round(float(config.hold_s) * float(config.control_hz))))
    cycle_count = max(1, int(config.cycles))
    direction = -1.0 if float(config.dz_mm) <= 0.0 else 1.0
    dz = abs(float(config.dz_mm))
    lateral_step = abs(float(config.lateral_mm)) if config.lateral else 0.0

    targets: list[np.ndarray] = []
    for step_index in range(steps_per_cycle * cycle_count):
        cycle_index = min(step_index // steps_per_cycle, cycle_count - 1)
        cycle_step = step_index - cycle_index * steps_per_cycle
        z_progress, retract_progress = cut_cycle_progress(cycle_step, steps_per_cycle)
        target = start.copy()
        target[2] = start[2] + direction * dz * z_progress
        target[1] = start[1] + lateral_step * (float(cycle_index) + retract_progress)
        targets.append(target)
    return targets


def build_chop_segments(start_pose_mm: np.ndarray, config: MotionConfig) -> list[PlannedSegment]:
    up_pose = np.asarray(start_pose_mm, dtype=float).reshape(6).copy()
    direction = -1.0 if float(config.dz_mm) <= 0.0 else 1.0
    dz = abs(float(config.dz_mm))
    lateral_step = abs(float(config.lateral_mm)) if config.lateral else 0.0
    segments: list[PlannedSegment] = []

    for cycle_index in range(max(1, int(config.cycles))):
        down_pose = up_pose.copy()
        down_pose[2] = up_pose[2] + direction * dz
        segments.append(PlannedSegment("descend", up_pose.copy(), down_pose.copy()))
        segments.append(PlannedSegment("retract", down_pose.copy(), up_pose.copy()))
        if lateral_step and cycle_index < int(config.cycles) - 1:
            next_up = up_pose.copy()
            next_up[1] += lateral_step
            segments.append(PlannedSegment("shift", up_pose.copy(), next_up.copy()))
            up_pose = next_up

    return segments


def main(argv: list[str] | None = None) -> int:
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

    try:
        connected = bool(robot.connect(config.robot_ip))
        if not connected:
            raise RuntimeError("failed to connect to the robot")
        robot.check_error_and_clear(dcss)
        _verify_frame_updates(robot, dcss, arm_index)
        robot.log_switch("1")
        robot.local_log_switch("1")

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
            else:
                _set_position_mode(robot, config)

            ref_joints = list(current_joints)
            period_s = 1.0 / float(config.control_hz)
            for step_index, target_xyzabc in enumerate(targets):
                target_matrix = kine.xyzabc_to_mat4x4(target_xyzabc.tolist())
                ik_para = FX_InvKineSolvePara()
                ik_para.set_input_ik_target_tcp(kine.mat4x4_to_mat1x16(target_matrix))
                ik_para.set_input_ik_ref_joint(ref_joints)
                ik_para.set_input_ik_zsp_type(0)
                ik = kine.ik(ik_para)
                ik_success = ik.get_output_result_num() >= 1
                target_joints = list(ik.get_output_ret_joint())
                if ik_success:
                    ref_joints = target_joints
                    last_target_joints = target_joints
                    if config.execute:
                        robot.clear_set()
                        robot.set_joint_cmd_pose(arm=config.arm, joints=target_joints)
                        robot.send_cmd()

                feedback = robot.subscribe(dcss)
                actual_joints = list(feedback["outputs"][arm_index]["fb_joint_pos"])
                actual_pose = np.asarray(kine.mat4x4_to_xyzabc(kine.fk(actual_joints)), dtype=float)
                trace_rows.append(_trace_row(step_index, target_xyzabc, actual_pose, target_joints, actual_joints, ik_success))
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
    root = Path(sdk_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _verify_frame_updates(robot, dcss, arm_index: int) -> None:
    observed = set()
    for _ in range(5):
        data = robot.subscribe(dcss)
        observed.add(data["outputs"][arm_index]["frame_serial"])
        time.sleep(0.01)
    if len(observed - {0}) == 0:
        raise RuntimeError("robot feedback frame did not update")


def _set_position_mode(robot, config: MotionConfig) -> None:
    robot.clear_set()
    robot.set_vel_acc(arm=config.arm, velRatio=int(config.vel_ratio), AccRatio=int(config.acc_ratio))
    _send_cmd_prefer_wait(robot)
    time.sleep(0.1)

    robot.clear_set()
    robot.set_state(arm=config.arm, state=1)
    _send_cmd_prefer_wait(robot)
    time.sleep(0.2)


def _feedback_joints(robot, dcss, arm_index: int) -> list[float]:
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
    if hasattr(robot, "send_cmd_wait_response"):
        return robot.send_cmd_wait_response(int(timeout_ms))
    return robot.send_cmd()


def _initialization_failure_message(target: Sequence[float], final: Sequence[float], feedback: dict, arm_index: int) -> str:
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

        if config.execute:
            robot.setPln_Cart(arm=config.arm, pset=pset)
            _wait_until_traj_idle(robot, dcss, arm_index, timeout_s=max(float(config.hold_s) * 3.0, 3.0))
            actual_joints = _feedback_joints(robot, dcss, arm_index)
        else:
            actual_joints = target_joints
        actual_pose = np.asarray(kine.mat4x4_to_xyzabc(kine.fk(actual_joints)), dtype=float)
        rows.append(_trace_row(segment_index, segment.end_xyzabc, actual_pose, target_joints, actual_joints, True))
        ref_joints = list(actual_joints)

    return rows, last_target_joints, actual_joints


def _wait_until_traj_idle(robot, dcss, arm_index: int, timeout_s: float) -> None:
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        data = robot.subscribe(dcss)
        if data["outputs"][arm_index]["traj_state"] == b"\x00":
            return
        time.sleep(0.001)
    raise RuntimeError("planned Cartesian trajectory did not finish before timeout")


def _configure_cartesian_impedance(robot, config: MotionConfig, current_pose: np.ndarray) -> None:
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


def _trace_row(
    step_index: int,
    target_xyzabc: np.ndarray,
    actual_xyzabc: np.ndarray,
    target_joints: Sequence[float],
    actual_joints: Sequence[float],
    ik_success: bool,
) -> dict:
    row = {"step": int(step_index), "ik_success": bool(ik_success)}
    for i, name in enumerate(("x", "y", "z", "a", "b", "c")):
        row[f"target_{name}"] = float(target_xyzabc[i])
        row[f"actual_{name}"] = float(actual_xyzabc[i])
    for i in range(7):
        row[f"target_q_{i}"] = float(target_joints[i]) if i < len(target_joints) else 0.0
        row[f"actual_q_{i}"] = float(actual_joints[i]) if i < len(actual_joints) else 0.0
    return row


def _write_trace_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
