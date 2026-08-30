#!/usr/bin/env python3
"""Batch both arm targets into one low-level SDK send operation per frame.

The default mode inspects the offline trajectory only.  ``--execute`` is kept
separate from the legacy Concise-SDK player and is intentionally not used by
automated tests or diagnostics.
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
JOINT_K = [8.0, 8.0, 8.0, 4.0, 2.0, 1.5, 1.0]
JOINT_D = [0.8, 0.8, 0.8, 0.6, 0.4, 0.3, 0.2]


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


def build_dual_arm_entry(
    start_left_deg: np.ndarray,
    target_left_deg: np.ndarray,
    start_right_deg: np.ndarray,
    target_right_deg: np.ndarray,
    *,
    duration_s: float,
    control_hz: float = 200.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create two rest-to-rest joint-space entry trajectories with exact endpoints."""
    start_left = np.asarray(start_left_deg, dtype=float)
    target_left = np.asarray(target_left_deg, dtype=float)
    start_right = np.asarray(start_right_deg, dtype=float)
    target_right = np.asarray(target_right_deg, dtype=float)
    for values in (start_left, target_left, start_right, target_right):
        if values.shape != (7,) or not np.all(np.isfinite(values)):
            raise ValueError("entry endpoints must each contain seven finite values")
    if duration_s <= 0 or control_hz <= 0:
        raise ValueError("entry duration and control rate must be positive")
    phase = np.linspace(0.0, 1.0, int(round(duration_s * control_hz)) + 1)
    smooth = 10 * phase**3 - 15 * phase**4 + 6 * phase**5
    left = start_left + smooth[:, None] * (target_left - start_left)
    right = start_right + smooth[:, None] * (target_right - start_right)
    left[0], left[-1] = start_left, target_left
    right[0], right[-1] = start_right, target_right
    return left, right


def build_hold_frames(left_deg: np.ndarray, right_deg: np.ndarray, *, frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Repeat the two current joint targets for a bounded transport-only probe."""
    if int(frames) <= 0:
        raise ValueError("hold frame count must be positive")
    return (
        np.tile(np.asarray(_target(left_deg), dtype=float), (int(frames), 1)),
        np.tile(np.asarray(_target(right_deg), dtype=float), (int(frames), 1)),
    )


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


def _stream_frames(robot: Any, dcss: Any, left: np.ndarray, right: np.ndarray, *, period_s: float, label: str) -> None:
    """Send aligned A/B targets against a monotonic clock and expose deadline misses."""
    started = time.monotonic()
    previous_frames: tuple[int, int] | None = None
    for index in range(len(left)):
        send_dual_arm_frame(robot, left[index], right[index])
        deadline = started + (index + 1) * period_s
        lag_s = time.monotonic() - deadline
        if index % 200 == 0 or index == len(left) - 1:
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


def load_trajectory(source: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(source, allow_pickle=False) as data:
        time_s = np.asarray(data["time_s"], dtype=float)
        left = np.rad2deg(np.asarray(data["left_arm_target_rad"], dtype=float))
        right = np.rad2deg(np.asarray(data["right_arm_target_rad"], dtype=float))
    if time_s.ndim != 1 or time_s.size < 2 or not np.all(np.isfinite(time_s)):
        raise ValueError("time_s must be a finite one-dimensional sequence")
    if left.shape != (time_s.size, 7) or right.shape != (time_s.size, 7):
        raise ValueError("both arm arrays must be shaped (N, 7)")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("trajectory contains non-finite targets")
    intervals = np.diff(time_s)
    if np.any(intervals <= 0) or not np.allclose(intervals, 0.005, atol=1e-9, rtol=0):
        raise ValueError("trajectory must have strictly increasing 5 ms timestamps")
    return time_s - time_s[0], left, right


def _configure_dual_joint_impedance(robot: Any) -> None:
    """Commit the controller state transition before separately committing K/D."""
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
    parser.add_argument("--speed-scale", type=float, default=0.05)
    parser.add_argument("--entry-duration-s", type=float, default=15.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--require-both-arms-ready",
        type=_parse_bool,
        default=True,
        metavar="true|false",
        help="Require healthy A and B feedback before streaming (default: true)",
    )
    parser.add_argument("--probe-current", action="store_true", help="Send one batched hold command at both current feedback poses")
    parser.add_argument("--probe-frames", type=int, default=200, help="Number of 200 Hz current-hold frames in probe mode")
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
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        time_s, left, right = load_trajectory(args.source_npz)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2
    period_s = 0.005 / args.speed_scale
    print(f"frames={len(time_s)} source_dt_s=0.005 send_period_s={period_s:.3f}")
    if not args.execute:
        print("DRY_RUN: no SDK connection or robot command")
        return 0

    from SDK_PYTHON.fx_robot import DCSS, Marvin_Robot

    robot = Marvin_Robot()
    dcss = DCSS()
    configured = False
    try:
        if not robot.connect(args.robot_ip):
            raise RuntimeError("failed to connect to robot")
        data = robot.subscribe(dcss)
        for index, arm in enumerate(("A", "B")):
            state = data["states"][index]
            if state["err_code"] != 0:
                raise RuntimeError(f"{arm}-arm controller error: {state['err_code']}")
        _configure_dual_joint_impedance(robot)
        configured = True
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
        if args.probe_current:
            hold_left, hold_right = build_hold_frames(current_left, current_right, frames=args.probe_frames)
            _stream_frames(robot, dcss, hold_left, hold_right, period_s=0.005, label="probe")
            observed = robot.subscribe(dcss)
            final_left = np.asarray(observed["outputs"][0]["fb_joint_pos"], dtype=float)
            final_right = np.asarray(observed["outputs"][1]["fb_joint_pos"], dtype=float)
            print(
                "PROBE_ACCEPTED "
                f"max_hold_error_deg={max(np.abs(final_left-current_left).max(), np.abs(final_right-current_right).max()):.3f}",
                flush=True,
            )
            return 0
        entry_left, entry_right = build_dual_arm_entry(
            current_left, left[0], current_right, right[0], duration_s=args.entry_duration_s
        )
        print(
            f"entry_frames={len(entry_left)} entry_max_jump_deg="
            f"{max(np.abs(left[0] - current_left).max(), np.abs(right[0] - current_right).max()):.3f}",
            flush=True,
        )
        _stream_frames(robot, dcss, entry_left, entry_right, period_s=0.005, label="entry")
        _stream_frames(robot, dcss, left, right, period_s=period_s, label="playback")
        return 0
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        try:
            robot.release_robot()
        except Exception:
            pass
        if configured:
            try:
                disable_both_arms(args.robot_ip)
            except Exception as exc:
                print(f"WARNING: explicit dual-arm disable failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
