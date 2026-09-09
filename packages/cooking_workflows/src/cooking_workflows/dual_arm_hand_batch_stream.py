#!/usr/bin/env python3
"""Batch both arm targets + Wuji hand into one SDK frame, running at 200 Hz.

This is the hand-integrated successor to dual_arm_batch_stream.py.  It uses the
proven low-level Marvin_Robot API (clear_set → set_joint_cmd_pose → send_cmd)
for both arms, plus the wujihandpy SDK for the hand — all synchronised on the
same wall-clock timeline.

Usage:
    # Dry-run (no device connection)
    PYTHONPATH=. python3 real_robot_debug/dual_arm_hand_batch_stream.py

    # Execute on real hardware (arms + hand)
    PYTHONPATH=. python3 real_robot_debug/dual_arm_hand_batch_stream.py \
        --source-npz recordings/recorded_hand_guarded_chop_200hz_official_right_retarget_candidate_forward_x_plus_80mm_right_z_plus_130mm.npz \
        --execute \
        --entry-duration-s 15 \
        --speed-scale 0.05

    # Hand only
    PYTHONPATH=. python3 real_robot_debug/dual_arm_hand_batch_stream.py \
        --execute --hand-only

    # Arms only (original dual_arm_batch_stream.py behaviour)
    PYTHONPATH=. python3 real_robot_debug/dual_arm_hand_batch_stream.py \
        --execute --arm-only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


from robot_core.paths import local_root
ROOT = local_root()
SOURCE = ROOT / "recordings" / "recorded_hand_guarded_chop_200hz_official_right_retarget_candidate.npz"
ROBOT_IP = "192.168.1.190"
WUJI_SERIAL = "344D345D3533"  # 当前实际设备 SN (2026-09-06)
JOINT_K = [8.0, 8.0, 8.0, 4.0, 2.0, 1.5, 1.0]
JOINT_D = [0.8, 0.8, 0.8, 0.6, 0.4, 0.3, 0.2]

# ── Wuji hand hardware limits (rad) ──
HAND_LOWER = np.array([
    [-0.11925496, -0.25735966, -0.54967515, -0.58205475],
    [-0.26239912, -0.43876813, -0.52905068, -0.50485284],
    [-0.23443433, -0.44822569, -0.56918519, -0.57158002],
    [-0.24246004, -0.47819426, -0.57648532, -0.5434772 ],
    [-0.25753832, -0.47106056, -0.68819585, -0.57163156],
])
HAND_UPPER = np.array([
    [1.67901294, 0.94978802, 1.66274323, 1.64120234],
    [1.64607823, 0.34152377, 1.67151943, 1.63989344],
    [1.6439589 , 0.33032859, 1.64428097, 1.66215395],
    [1.64423041, 0.27520675, 1.67904946, 1.68428525],
    [1.63236697, 0.2921523 , 1.53570436, 1.64613478],
])
HAND_MARGIN = 0.02  # rad safety margin
FINGER_NAMES = ["拇指", "食指", "中指", "无名指", "小指"]

# ── Arm joint limits (degrees) ──
ARM_JOINT_LIMITS_DEG = [
    (-170, 170), (-100, 120), (-170, 170), (-130, 130),
    (-170, 170), (-90, 220), (-170, 170),
]

ARMS = [
    {"sdk_arm": "A", "index": 0, "npz_key": "left_arm_target_rad",  "name": "Left  (A)"},
    {"sdk_arm": "B", "index": 1, "npz_key": "right_arm_target_rad", "name": "Right (B)"},
]


# ═══════════════════════════════════════════════════════════════
# Low-level SDK helpers (proven 200 Hz path)
# ═══════════════════════════════════════════════════════════════

def _target(values: np.ndarray) -> list[float]:
    value = np.asarray(values, dtype=float)
    if value.shape != (7,) or not np.all(np.isfinite(value)):
        raise ValueError("each arm target must contain seven finite values")
    return value.tolist()


def send_dual_arm_frame(robot: Any, left_deg: np.ndarray, right_deg: np.ndarray) -> None:
    """Stage A and B targets, then atomically request one SDK buffer send."""
    left = _target(left_deg)
    right = _target(right_deg)
    robot.clear_set()
    robot.set_joint_cmd_pose(arm="A", joints=left)
    robot.set_joint_cmd_pose(arm="B", joints=right)
    if not robot.send_cmd():
        raise RuntimeError("controller rejected the batched dual-arm send")


def send_single_arm_frame(robot: Any, arm: str, joints_deg: np.ndarray) -> None:
    """Stage one arm target and send atomically (for --hand-only / --arm-only modes)."""
    joints = _target(joints_deg)
    robot.clear_set()
    robot.set_joint_cmd_pose(arm=arm, joints=joints)
    if not robot.send_cmd():
        raise RuntimeError(f"controller rejected {arm}-arm send")


def validate_dual_feedback(data: dict, left_target: np.ndarray, right_target: np.ndarray) -> float:
    """Verify both arms remain active and return their worst absolute tracking error."""
    actuals: list[np.ndarray] = []
    for index, arm in enumerate(("A", "B")):
        state = data["states"][index]
        if state.get("err_code", 0) != 0:
            raise RuntimeError(f"{arm}-arm controller error: {state.get('err_code')}")
        if state.get("cur_state") != 3:
            raise RuntimeError(f"{arm}-arm left joint-impedance state: {state.get('cur_state')}")
        actual = np.asarray(data["outputs"][index]["fb_joint_pos"], dtype=float)
        if actual.shape != (7,) or not np.all(np.isfinite(actual)):
            raise RuntimeError(f"{arm}-arm feedback is invalid")
        actuals.append(actual)
    return float(max(np.abs(actuals[0] - left_target).max(), np.abs(actuals[1] - right_target).max()))


def dual_arms_ready_issue(data: dict) -> str | None:
    """Return the first reason that prevents a safe dual-arm stream start."""
    for index, arm in enumerate(("A", "B")):
        state = data["states"][index]
        if state.get("err_code", 0) != 0:
            return f"{arm}-arm controller error: {state.get('err_code')}"
        if state.get("cur_state") != 3:
            return f"{arm}-arm left joint-impedance state: {state.get('cur_state')}"
        actual = np.asarray(data["outputs"][index]["fb_joint_pos"], dtype=float)
        if actual.shape != (7,) or not np.all(np.isfinite(actual)):
            return f"{arm}-arm feedback is invalid"
    return None


def validate_single_arm_feedback(data: dict, arm: str, arm_index: int, target: np.ndarray) -> float:
    """Verify a single arm's feedback."""
    state = data["states"][arm_index]
    if state.get("err_code", 0) != 0:
        raise RuntimeError(f"{arm}-arm controller error: {state.get('err_code')}")
    if state.get("cur_state") != 3:
        raise RuntimeError(f"{arm}-arm left joint-impedance state: {state.get('cur_state')}")
    actual = np.asarray(data["outputs"][arm_index]["fb_joint_pos"], dtype=float)
    if actual.shape != (7,) or not np.all(np.isfinite(actual)):
        raise RuntimeError(f"{arm}-arm feedback is invalid")
    return float(np.abs(actual - target).max())


# ═══════════════════════════════════════════════════════════════
# Trajectory loading & validation
# ═══════════════════════════════════════════════════════════════

def load_trajectory(source: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Load NPZ; return (time_s, left_deg, right_deg, hand_rad_or_None)."""
    with np.load(source, allow_pickle=False) as data:
        time_s = np.asarray(data["time_s"], dtype=float)
        left = np.rad2deg(np.asarray(data["left_arm_target_rad"], dtype=float))
        right = np.rad2deg(np.asarray(data["right_arm_target_rad"], dtype=float))
        if "right_hand_target_rad" in data:
            hand = np.asarray(data["right_hand_target_rad"], dtype=float)
        else:
            hand = None

    if time_s.ndim != 1 or time_s.size < 2 or not np.all(np.isfinite(time_s)):
        raise ValueError("time_s must be a finite one-dimensional sequence")
    if left.shape != (time_s.size, 7) or right.shape != (time_s.size, 7):
        raise ValueError("both arm arrays must be shaped (N, 7)")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("trajectory contains non-finite arm targets")
    if hand is not None:
        if hand.shape != (time_s.size, 20) or not np.all(np.isfinite(hand)):
            raise ValueError("right_hand_target_rad must be finite (N,20)")
    intervals = np.diff(time_s)
    if np.any(intervals <= 0) or not np.allclose(intervals, 0.005, atol=1e-9, rtol=0):
        raise ValueError("trajectory must have strictly increasing 5 ms timestamps")
    return time_s - time_s[0], left, right, hand


def check_arm_limits(left_deg: np.ndarray, right_deg: np.ndarray) -> None:
    """Check arm trajectories are within hardware limits (1 deg margin)."""
    for name, targets_deg in [("Left (A)", left_deg), ("Right (B)", right_deg)]:
        for j in range(7):
            lo, hi = ARM_JOINT_LIMITS_DEG[j]
            col = targets_deg[:, j]
            if col.min() < lo + 1.0 or col.max() > hi - 1.0:
                raise ValueError(
                    f"{name} Joint {j+1}: [{col.min():.1f}, {col.max():.1f}] deg "
                    f"超出限位 [{lo}, {hi}] (1 deg margin)"
                )


def check_and_clip_hand(hand_rad: np.ndarray) -> np.ndarray:
    """Clip hand trajectory to hardware limits. Returns clipped copy."""
    clipped = hand_rad.copy().reshape(-1, 5, 4)
    violations = []
    for i in range(5):
        for j in range(4):
            lo, hi = HAND_LOWER[i, j] + HAND_MARGIN, HAND_UPPER[i, j] - HAND_MARGIN
            col = clipped[:, i, j]
            if col.min() < lo:
                violations.append(f"{FINGER_NAMES[i]} J{j+1}: min {col.min():.4f} < {lo:.4f}")
            if col.max() > hi:
                violations.append(f"{FINGER_NAMES[i]} J{j+1}: max {col.max():.4f} > {hi:.4f}")
            clipped[:, i, j] = np.clip(col, lo, hi)
    if violations:
        print("⚠️ Wuji 手轨迹超出限位（已自动裁剪）:")
        for v in violations:
            print(f"  {v}")
    else:
        print("✅ Wuji 手轨迹在硬件限位内")
    return clipped.reshape(-1, 20)


# ═══════════════════════════════════════════════════════════════
# Trajectory generation helpers
# ═══════════════════════════════════════════════════════════════

def build_quintic_entry(start, target, *, duration_s: float, hz: float = 200.0):
    """Quintic rest-to-rest trajectory from start to target."""
    start = np.asarray(start, dtype=float)
    target = np.asarray(target, dtype=float)
    steps = int(round(duration_s * hz))
    if steps < 1:
        steps = 1
    phase = np.linspace(0.0, 1.0, steps + 1)
    smooth = 10 * phase**3 - 15 * phase**4 + 6 * phase**5
    result = start + smooth[:, None] * (target - start)
    result[0] = start
    result[-1] = target
    return result


def build_hold_frames(values: np.ndarray, *, frames: int):
    """Repeat the same values for a given number of frames."""
    return np.tile(np.asarray(values, dtype=float), (int(frames), 1))


# ═══════════════════════════════════════════════════════════════
# SDK connection helpers
# ═══════════════════════════════════════════════════════════════

def configure_dual_joint_impedance(robot: Any) -> None:
    """Commit state/type/vel, then K/D in two separate send_cmd calls."""
    robot.clear_set()
    for arm in ("A", "B"):
        robot.set_state(arm=arm, state=3)
        robot.set_impedance_type(arm=arm, type=1)
        robot.set_vel_acc(arm=arm, velRatio=100, AccRatio=100)
    if not robot.send_cmd():
        raise RuntimeError("controller rejected dual-arm impedance mode configuration")
    time.sleep(0.2)

    robot.clear_set()
    for arm in ("A", "B"):
        if not robot.set_joint_kd_params(arm=arm, K=JOINT_K, D=JOINT_D):
            raise RuntimeError(f"controller rejected {arm}-arm impedance parameters")
    if not robot.send_cmd():
        raise RuntimeError("controller rejected dual-arm impedance K/D configuration")
    time.sleep(0.2)


def configure_single_arm_impedance(robot: Any, arm: str) -> None:
    """Configure impedance for a single arm."""
    robot.clear_set()
    robot.set_state(arm=arm, state=3)
    robot.set_impedance_type(arm=arm, type=1)
    robot.set_vel_acc(arm=arm, velRatio=100, AccRatio=100)
    if not robot.send_cmd():
        raise RuntimeError(f"controller rejected {arm}-arm impedance mode configuration")
    time.sleep(0.2)

    robot.clear_set()
    if not robot.set_joint_kd_params(arm=arm, K=JOINT_K, D=JOINT_D):
        raise RuntimeError(f"controller rejected {arm}-arm impedance parameters")
    if not robot.send_cmd():
        raise RuntimeError(f"controller rejected {arm}-arm impedance K/D configuration")
    time.sleep(0.2)


def disable_both_arms(robot_ip: str, *, concise_factory: Any | None = None) -> None:
    """Use the SDK's proven direct disable call after releasing the batch client."""
    if concise_factory is None:
        from SDK_PYTHON.fx_robot import Concise_Marvin_Robot
        concise_factory = Concise_Marvin_Robot
    robot = concise_factory()
    try:
        if not robot.connect(robot_ip):
            raise RuntimeError("failed to reconnect for dual-arm disable")
        for arm in ("A", "B"):
            if robot.disable(arm) is not True:
                raise RuntimeError(f"failed to disable {arm} arm")
    finally:
        robot.release_robot()


# ═══════════════════════════════════════════════════════════════
# Streaming core
# ═══════════════════════════════════════════════════════════════

def _stream_dual_arm_frames(
    robot: Any, dcss: Any,
    left: np.ndarray, right: np.ndarray, *,
    period_s: float, label: str,
    hand: Any = None, hand_frames: np.ndarray | None = None,
) -> None:
    """Send aligned A/B targets (and optional hand) against a monotonic clock."""
    started = time.monotonic()
    previous_frames: tuple[int, int] | None = None
    n = len(left)
    for index in range(n):
        send_dual_arm_frame(robot, left[index], right[index])

        # Hand command (if provided)
        if hand is not None and hand_frames is not None:
            hand.write_joint_target_position(hand_frames[index].reshape(5, 4))

        deadline = started + (index + 1) * period_s
        lag_s = time.monotonic() - deadline

        if index % 200 == 0 or index == n - 1:
            data = robot.subscribe(dcss)
            tracking_error = validate_dual_feedback(data, left[index], right[index])
            frames = tuple(int(data["outputs"][arm_index].get("frame_serial", 0)) for arm_index in (0, 1))
            if 0 in frames or frames == previous_frames:
                raise RuntimeError(f"{label} feedback frames did not update: {frames}")
            previous_frames = frames
            if tracking_error > 5.0:
                raise RuntimeError(f"{label} tracking error exceeded 5 deg: {tracking_error:.3f}")
            print(
                f"{label}_frame={index} lag_s={lag_s:.4f} max_error_deg={tracking_error:.3f} frames={frames}",
                flush=True,
            )

        sleep_s = deadline - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)


def _stream_single_arm_frames(
    robot: Any, dcss: Any,
    arm: str, arm_index: int,
    targets: np.ndarray, *,
    period_s: float, label: str,
    hand: Any = None, hand_frames: np.ndarray | None = None,
) -> None:
    """Send one arm targets (and optional hand) against a monotonic clock."""
    started = time.monotonic()
    previous_frame: int | None = None
    n = len(targets)
    for index in range(n):
        send_single_arm_frame(robot, arm, targets[index])

        # Hand command (if provided)
        if hand is not None and hand_frames is not None:
            hand.write_joint_target_position(hand_frames[index].reshape(5, 4))

        deadline = started + (index + 1) * period_s
        lag_s = time.monotonic() - deadline

        if index % 200 == 0 or index == n - 1:
            data = robot.subscribe(dcss)
            tracking_error = validate_single_arm_feedback(data, arm, arm_index, targets[index])
            frame = int(data["outputs"][arm_index].get("frame_serial", 0))
            if frame == 0 or frame == previous_frame:
                raise RuntimeError(f"{label} feedback frame did not update: {frame}")
            previous_frame = frame
            if tracking_error > 5.0:
                raise RuntimeError(f"{label} tracking error exceeded 5 deg: {tracking_error:.3f}")
            print(
                f"{label}_frame={index} lag_s={lag_s:.4f} max_error_deg={tracking_error:.3f} frame={frame}",
                flush=True,
            )

        sleep_s = deadline - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)


def _stream_hand_only(
    hand: Any,
    hand_frames: np.ndarray, *,
    period_s: float, label: str,
) -> None:
    """Send hand-only frames against a monotonic clock."""
    started = time.monotonic()
    n = len(hand_frames)
    for index in range(n):
        hand.write_joint_target_position(hand_frames[index].reshape(5, 4))

        deadline = started + (index + 1) * period_s
        lag_s = time.monotonic() - deadline

        if index % 200 == 0 or index == n - 1:
            print(f"{label}_frame={index} lag_s={lag_s:.4f}", flush=True)

        sleep_s = deadline - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)


# ═══════════════════════════════════════════════════════════════
# Entry phase: stream quintic entry for arms + hand
# ═══════════════════════════════════════════════════════════════

def _stream_entry_dual(
    robot: Any, dcss: Any,
    left_entry: np.ndarray, right_entry: np.ndarray,
    hand: Any | None, hand_entry: np.ndarray | None,
    entry_hz: float,
) -> None:
    """Stream dual-arm entry with optional hand."""
    entry_dt = 1.0 / entry_hz
    started = time.monotonic()
    previous_frames: tuple[int, int] | None = None
    n = len(left_entry)
    for index in range(n):
        send_dual_arm_frame(robot, left_entry[index], right_entry[index])

        if hand is not None and hand_entry is not None:
            hand.write_joint_target_position(hand_entry[index].reshape(5, 4))

        deadline = started + (index + 1) * entry_dt
        if index % 200 == 0 or index == n - 1:
            data = robot.subscribe(dcss)
            tracking_error = validate_dual_feedback(data, left_entry[index], right_entry[index])
            frames = tuple(int(data["outputs"][arm_index].get("frame_serial", 0)) for arm_index in (0, 1))
            if 0 in frames or frames == previous_frames:
                raise RuntimeError(f"entry feedback frames did not update: {frames}")
            previous_frames = frames
            if tracking_error > 5.0:
                raise RuntimeError(f"entry tracking error exceeded 5 deg: {tracking_error:.3f}")
            print(
                f"entry_frame={index} max_error_deg={tracking_error:.3f} frames={frames}",
                flush=True,
            )
        sleep_s = deadline - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)


def _stream_entry_single_arm(
    robot: Any, dcss: Any,
    arm: str, arm_index: int,
    entry: np.ndarray,
    hand: Any | None, hand_entry: np.ndarray | None,
    entry_hz: float,
) -> None:
    """Stream single-arm entry with optional hand."""
    entry_dt = 1.0 / entry_hz
    started = time.monotonic()
    previous_frame: int | None = None
    n = len(entry)
    for index in range(n):
        send_single_arm_frame(robot, arm, entry[index])

        if hand is not None and hand_entry is not None:
            hand.write_joint_target_position(hand_entry[index].reshape(5, 4))

        deadline = started + (index + 1) * entry_dt
        if index % 200 == 0 or index == n - 1:
            data = robot.subscribe(dcss)
            tracking_error = validate_single_arm_feedback(data, arm, arm_index, entry[index])
            frame = int(data["outputs"][arm_index].get("frame_serial", 0))
            if frame == 0 or frame == previous_frame:
                raise RuntimeError(f"entry feedback frame did not update: {frame}")
            previous_frame = frame
            if tracking_error > 5.0:
                raise RuntimeError(f"entry tracking error exceeded 5 deg: {tracking_error:.3f}")
            print(
                f"entry_frame={index} max_error_deg={tracking_error:.3f} frame={frame}",
                flush=True,
            )
        sleep_s = deadline - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)


def _stream_entry_hand_only(
    hand: Any,
    hand_entry: np.ndarray,
    entry_hz: float,
) -> None:
    """Stream hand-only entry."""
    entry_dt = 1.0 / entry_hz
    started = time.monotonic()
    n = len(hand_entry)
    for index in range(n):
        hand.write_joint_target_position(hand_entry[index].reshape(5, 4))
        deadline = started + (index + 1) * entry_dt
        if index % 200 == 0 or index == n - 1:
            print(f"entry_frame={index}", flush=True)
        sleep_s = deadline - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("must be true or false")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-npz", type=Path, default=SOURCE)
    parser.add_argument("--robot-ip", default=ROBOT_IP)
    parser.add_argument("--wuji-serial", default=WUJI_SERIAL)
    parser.add_argument("--speed-scale", type=float, default=0.05)
    parser.add_argument("--entry-duration-s", type=float, default=15.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--hand-only", action="store_true")
    parser.add_argument("--arm-only", action="store_true")
    parser.add_argument(
        "--require-both-arms-ready",
        type=_parse_bool,
        default=True,
        metavar="true|false",
        help="Require healthy A and B feedback before streaming (default: true)",
    )
    parser.add_argument("--probe-current", action="store_true",
                        help="Send one batched hold command at both current feedback poses")
    parser.add_argument("--probe-frames", type=int, default=200,
                        help="Number of 200 Hz current-hold frames in probe mode")
    args = parser.parse_args(argv)

    if not args.source_npz.is_file():
        raise ValueError(f"source NPZ not found: {args.source_npz}")
    if not 0 < args.speed_scale <= 1:
        raise ValueError("speed-scale must be in (0, 1]")
    if args.entry_duration_s <= 0:
        raise ValueError("entry-duration-s must be positive")
    if args.probe_current and not args.execute:
        raise ValueError("--probe-current requires --execute")
    if args.probe_frames <= 0:
        raise ValueError("probe-frames must be positive")
    if args.hand_only and args.arm_only:
        raise ValueError("--hand-only and --arm-only cannot both be set")
    return args


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        time_s, left, right, hand_rad = load_trajectory(args.source_npz)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    period_s = 0.005 / args.speed_scale
    print(f"frames={len(time_s)} source_dt_s=0.005 send_period_s={period_s:.3f}")

    # ── Safety checks ──
    if not args.hand_only:
        check_arm_limits(left, right)
        print("✅ 双臂轨迹在硬件限位内")

    hand_clipped: np.ndarray | None = None
    if hand_rad is not None and not args.arm_only:
        hand_clipped = check_and_clip_hand(hand_rad)
        print(f"✅ Wuji 手轨迹: {len(hand_clipped)} 帧")
    elif hand_rad is None and not args.arm_only:
        print("⚠️ NPZ 不包含 right_hand_target_rad，跳过手部控制")

    # ── Print trajectory ranges ──
    if not args.hand_only:
        for name, targets_deg in [("Left  (A)", left), ("Right (B)", right)]:
            max_step = np.abs(np.diff(targets_deg, axis=0)).max()
            print(f"\n{name} (degrees):")
            for j in range(7):
                print(f"  J{j+1}: [{targets_deg[:,j].min():7.1f}, {targets_deg[:,j].max():7.1f}]")
            print(f"  最大帧间步长: {max_step:.2f} deg")

    if hand_clipped is not None:
        print(f"\nWuji 手 (rad):")
        for i in range(5):
            for j in range(4):
                col = hand_clipped[:, i * 4 + j]
                print(f"  {FINGER_NAMES[i]} J{j+1}: [{col.min():.4f}, {col.max():.4f}]")

    print()

    if not args.execute:
        mode = "双手臂+手" if not args.hand_only and not args.arm_only else \
               "仅手" if args.hand_only else "仅臂"
        print(f"DRY_RUN: 规划完成，未连接设备。模式: {mode}")
        print(f"  entry: {args.entry_duration_s}s → 首帧")
        print(f"  playback: {len(time_s)} 帧 at {args.speed_scale}x")
        return 0

    # ═════════════════════════════════════════════════════════════
    # Real execution
    # ═════════════════════════════════════════════════════════════
    from SDK_PYTHON.fx_robot import DCSS, Marvin_Robot

    dcss = DCSS()
    robot: Any = None
    hand: Any = None
    arms_configured = False
    hand_enabled = False

    try:
        # ── Connect Wuji hand ──
        if not args.arm_only and hand_clipped is not None:
            try:
                from wujihandpy import Hand as WujiHand
            except ModuleNotFoundError:
                print("ERROR: 需要 wujihandpy 模块，请使用 .venv-wujihand/bin/python 运行")
                print("  命令: PYTHONPATH=. .venv-wujihand/bin/python real_robot_debug/dual_arm_hand_batch_stream.py ...")
                return 3
            print(f"\n连接 Wuji 手 SN={args.wuji_serial} ...")
            hand = WujiHand(serial_number=args.wuji_serial)
            errors = np.asarray(hand.read_joint_error_code())
            if errors.any():
                print(f"⚠️ Wuji 手错误码非零 (sum={errors.sum()})，但继续执行")
            temps = np.asarray(hand.read_joint_temperature(), dtype=float)
            print(f"Wuji 手温度: max={temps.max():.1f}°C")
            print("✅ Wuji 手已连接")

        # ── Connect robot ──
        if not args.hand_only:
            print(f"\n连接机器人 {args.robot_ip} ...")
            robot = Marvin_Robot()
            if not robot.connect(args.robot_ip):
                raise RuntimeError("failed to connect to robot")

            data = robot.subscribe(dcss)
            for index, arm in enumerate(("A", "B")):
                state = data["states"][index]
                if state["err_code"] != 0:
                    raise RuntimeError(f"{arm}-arm controller error: {state['err_code']}")

            configure_dual_joint_impedance(robot)
            arms_configured = True

            feedback = robot.subscribe(dcss)
            readiness_issue = dual_arms_ready_issue(feedback)
            if args.require_both_arms_ready:
                if readiness_issue is not None:
                    print(f"READY_CHECK_BLOCKED: {readiness_issue}", file=sys.stderr, flush=True)
                    raise RuntimeError(readiness_issue)
                print("READY_CHECK_PASSED: both arms ready", flush=True)
            else:
                print(
                    "READY_CHECK_SKIPPED: --require-both-arms-ready=false"
                    + (f" ({readiness_issue})" if readiness_issue is not None else ""),
                    flush=True,
                )

            current_left = np.asarray(feedback["outputs"][0]["fb_joint_pos"], dtype=float)
            current_right = np.asarray(feedback["outputs"][1]["fb_joint_pos"], dtype=float)

            # ── Probe mode ──
            if args.probe_current:
                hold_left = build_hold_frames(current_left, frames=args.probe_frames)
                hold_right = build_hold_frames(current_right, frames=args.probe_frames)
                _stream_dual_arm_frames(robot, dcss, hold_left, hold_right,
                                        period_s=0.005, label="probe")
                observed = robot.subscribe(dcss)
                final_left = np.asarray(observed["outputs"][0]["fb_joint_pos"], dtype=float)
                final_right = np.asarray(observed["outputs"][1]["fb_joint_pos"], dtype=float)
                print(
                    "PROBE_ACCEPTED "
                    f"max_hold_error_deg={max(np.abs(final_left-current_left).max(), np.abs(final_right-current_right).max()):.3f}",
                    flush=True,
                )
                return 0

        # ── Enable hand ──
        if hand is not None:
            print(f"\n>>> Wuji 手使能 ...")
            hand.write_joint_enabled(True)
            hand_enabled = True
            time.sleep(0.1)
            print("✅ Wuji 手已使能")

        # ═══════════════════════════════════════════════════════════
        # Phase 1: Entry
        # ═══════════════════════════════════════════════════════════
        entry_hz = 200.0

        if not args.hand_only:
            left_entry = build_quintic_entry(current_left, left[0],
                                             duration_s=args.entry_duration_s, hz=entry_hz)
            right_entry = build_quintic_entry(current_right, right[0],
                                              duration_s=args.entry_duration_s, hz=entry_hz)
            entry_frames = len(left_entry)
            arm_entry_jump = max(
                np.abs(left[0] - current_left).max(),
                np.abs(right[0] - current_right).max(),
            )
            print(
                f"entry_frames={entry_frames} entry_max_jump_deg={arm_entry_jump:.3f}",
                flush=True,
            )
        else:
            entry_frames = int(args.entry_duration_s * entry_hz) + 1

        # Hand entry
        hand_entry: np.ndarray | None = None
        if hand is not None and hand_clipped is not None:
            hand_current = np.asarray(hand.read_joint_actual_position(), dtype=float).reshape(20)
            hand_first_target = hand_clipped[0]
            hand_entry = build_quintic_entry(hand_current, hand_first_target,
                                             duration_s=args.entry_duration_s, hz=entry_hz)

        print(f"\n[Phase 1/2] 缓入: {args.entry_duration_s}s (Ctrl+C 可中断)")

        if not args.hand_only:
            _stream_entry_dual(robot, dcss, left_entry, right_entry,
                               hand, hand_entry, entry_hz)
        elif hand is not None and hand_entry is not None:
            _stream_entry_hand_only(hand, hand_entry, entry_hz)
        print("✅ 缓入完成")

        # ═══════════════════════════════════════════════════════════
        # Phase 2: Playback
        # ═══════════════════════════════════════════════════════════
        print(f"\n[Phase 2/2] 回放: {len(time_s)} 帧 at {args.speed_scale}x")

        if not args.hand_only:
            _stream_dual_arm_frames(robot, dcss, left, right,
                                    period_s=period_s, label="playback",
                                    hand=hand, hand_frames=hand_clipped)
        elif hand is not None and hand_clipped is not None:
            _stream_hand_only(hand, hand_clipped, period_s=period_s, label="playback")
        print("✅ 回放完成")
        return 0

    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断", file=sys.stderr)
        return 130
    finally:
        # ── Cleanup ──
        print("\n>>> 清理中 ...")

        # Disable hand
        if hand_enabled and hand is not None:
            try:
                hand.write_joint_enabled(False)
                print("Wuji 手已去使能")
            except Exception as exc:
                print(f"WARNING: Wuji 手去使能失败: {exc}", file=sys.stderr)

        # Release robot & disable arms
        if arms_configured and robot is not None:
            try:
                robot.release_robot()
            except Exception:
                pass
            try:
                disable_both_arms(args.robot_ip)
                print("双臂已禁用")
            except Exception as exc:
                print(f"WARNING: 双臂禁用失败: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())