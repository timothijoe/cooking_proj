#!/usr/bin/env python3
"""Read controller feedback and diagnose one deliberately tiny A-arm command.

Default mode connects only to read and print live A-arm feedback.  It never
changes robot state or sends a joint target.  A physical command additionally
requires all of ``--execute``, ``--trial-joint``, and ``--trial-delta-deg``.
The trial is limited to one A-arm joint and an absolute displacement of 1 deg.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROBOT_IP = "192.168.1.190"
JOINT_K = [8.0, 8.0, 8.0, 4.0, 2.0, 1.5, 1.0]
JOINT_D = [0.8, 0.8, 0.8, 0.6, 0.4, 0.3, 0.2]
JOINT_LIMITS_DEG = [
    (-170.0, 170.0), (-100.0, 120.0), (-170.0, 170.0), (-130.0, 130.0),
    (-170.0, 170.0), (-90.0, 220.0), (-170.0, 170.0),
]


@dataclass(frozen=True)
class ArmDiagnosticConfig:
    arm: str = "A"
    robot_ip: str = ROBOT_IP
    execute: bool = False
    trial_joint: int | None = None
    trial_delta_deg: float | None = None
    entry_npz: Path | None = None
    execute_entry: bool = False
    entry_duration_s: float = 15.0
    vel_ratio: int = 10
    acc_ratio: int = 10
    monitor_s: float = 5.0
    poll_hz: float = 20.0


def validate_config(config: ArmDiagnosticConfig) -> None:
    """Reject unsafe or ambiguous settings before opening an SDK connection."""
    if config.arm != "A":
        raise ValueError("this diagnostic intentionally supports only arm A")
    if not math.isfinite(config.monitor_s) or config.monitor_s <= 0:
        raise ValueError("monitor-s must be positive and finite")
    if not math.isfinite(config.poll_hz) or not 1 <= config.poll_hz <= 50:
        raise ValueError("poll-hz must be in [1, 50]")
    if not math.isfinite(config.entry_duration_s) or config.entry_duration_s < 10:
        raise ValueError("entry-duration-s must be at least 10 seconds")
    if not 1 <= int(config.vel_ratio) <= 100:
        raise ValueError("vel-ratio must be in [1, 100]")
    if not 1 <= int(config.acc_ratio) <= 100:
        raise ValueError("acc-ratio must be in [1, 100]")
    if config.entry_npz is not None and not config.entry_npz.is_file():
        raise ValueError(f"entry-npz not found: {config.entry_npz}")
    if config.execute_entry:
        if not config.execute or config.entry_npz is None:
            raise ValueError("--execute-entry requires both --execute and --entry-npz")
        if config.trial_joint is not None or config.trial_delta_deg is not None:
            raise ValueError("entry execution cannot be combined with a single-joint trial")
    if config.execute and not config.execute_entry:
        if config.trial_joint is None or config.trial_delta_deg is None:
            raise ValueError("--execute requires both --trial-joint and --trial-delta-deg")
        if config.trial_joint not in range(1, 8):
            raise ValueError("trial-joint must be an integer in [1, 7]")
        if not math.isfinite(config.trial_delta_deg) or not 0 < abs(config.trial_delta_deg) <= 1:
            raise ValueError("trial-delta-deg must be in [-1, -0) or (0, 1]")
    elif not config.execute and (config.trial_joint is not None or config.trial_delta_deg is not None):
        raise ValueError("trial settings require --execute")


def load_entry_target_deg(path: Path) -> np.ndarray:
    """Load the first A-arm waypoint from a standard 200 Hz trajectory NPZ."""
    with np.load(path, allow_pickle=False) as data:
        time_s = np.asarray(data["time_s"], dtype=float)
        targets_rad = np.asarray(data["left_arm_target_rad"], dtype=float)
    if time_s.ndim != 1 or time_s.size < 2 or targets_rad.shape != (time_s.size, 7):
        raise ValueError("entry-npz must contain matching time_s and left_arm_target_rad (N,7)")
    if not np.all(np.isfinite(targets_rad)):
        raise ValueError("entry-npz contains non-finite A-arm targets")
    return np.rad2deg(targets_rad[0])


def build_quintic_entry(
    start_deg: np.ndarray, target_deg: np.ndarray, *, duration_s: float, control_hz: float = 200.0
) -> np.ndarray:
    """Return a rest-to-rest joint-space entry segment with exact endpoints."""
    start = np.asarray(start_deg, dtype=float)
    target = np.asarray(target_deg, dtype=float)
    if start.shape != (7,) or target.shape != (7,) or not np.all(np.isfinite(start)) or not np.all(np.isfinite(target)):
        raise ValueError("entry endpoints must each contain seven finite values")
    steps = int(round(duration_s * control_hz))
    phase = np.linspace(0.0, 1.0, steps + 1)
    smooth = 10 * phase**3 - 15 * phase**4 + 6 * phase**5
    result = start + smooth[:, None] * (target - start)
    result[0] = start
    result[-1] = target
    return result


def entry_progress_line(*, index: int, control_hz: float, elapsed_s: float) -> str:
    """Format the scheduling evidence needed to distinguish timing from tracking faults."""
    scheduled_s = float(index) / float(control_hz)
    lag_s = float(elapsed_s) - scheduled_s
    return (
        f"entry_frame={index} scheduled_s={scheduled_s:.3f} "
        f"elapsed_s={float(elapsed_s):.3f} lag_s={lag_s:.3f}"
    )


def command_single_joint_trial(
    robot: Any,
    *,
    current_deg: np.ndarray,
    joint_number: int,
    delta_deg: float,
) -> list[float]:
    """Issue one bounded direct A-arm target and fail if the SDK rejects it."""
    current = np.asarray(current_deg, dtype=float)
    if current.shape != (7,) or not np.all(np.isfinite(current)):
        raise ValueError("current_deg must be seven finite values")
    target = current.copy()
    target[joint_number - 1] += float(delta_deg)
    lower, upper = JOINT_LIMITS_DEG[joint_number - 1]
    if not lower <= target[joint_number - 1] <= upper:
        raise ValueError(
            f"trial target {target[joint_number - 1]:.3f} deg exceeds hardware limit "
            f"[{lower:.1f}, {upper:.1f}]"
        )
    target_list = target.tolist()
    accepted = robot.set_joint_position_cmd("A", target_list)
    if accepted is not True:
        raise RuntimeError("controller rejected the single-joint diagnostic command")
    return target_list


def _read_feedback(robot: Any, dcss: Any) -> tuple[dict, np.ndarray]:
    data = robot.subscribe(dcss)
    state = data["states"][0]
    joints = np.asarray(data["outputs"][0]["fb_joint_pos"], dtype=float)
    if joints.shape != (7,) or not np.all(np.isfinite(joints)):
        raise RuntimeError("A-arm feedback is not seven finite joint values")
    if state.get("err_code", 0) != 0:
        raise RuntimeError(f"A-arm controller error: {state.get('err_code')}")
    return data, joints


def _print_sample(data: dict, actual_deg: np.ndarray, target_deg: np.ndarray | None) -> None:
    output = data["outputs"][0]
    state = data["states"][0]
    message = (
        f"frame={output.get('frame_serial', '?')} state={state.get('cur_state', '?')} "
        f"actual_deg={np.round(actual_deg, 3).tolist()}"
    )
    if target_deg is not None:
        error = target_deg - actual_deg
        message += (
            f" target_deg={np.round(target_deg, 3).tolist()}"
            f" max_error_deg={float(np.max(np.abs(error))):.3f}"
        )
    print(message, flush=True)


def _verify_feedback_advances(robot: Any, dcss: Any) -> None:
    serials: set[int] = set()
    for _ in range(5):
        data, _ = _read_feedback(robot, dcss)
        serials.add(int(data["outputs"][0].get("frame_serial", 0)))
        time.sleep(0.02)
    if 0 in serials or len(serials) < 2:
        raise RuntimeError("A-arm feedback frames are not advancing")


def run_diagnostic(config: ArmDiagnosticConfig) -> int:
    """Run read-only monitoring or the explicitly opted-in one-joint trial."""
    validate_config(config)
    from SDK_PYTHON.fx_robot import Concise_Marvin_Robot, DCSS

    robot = Concise_Marvin_Robot()
    dcss = DCSS()
    executed = False
    try:
        print(f"Connecting read-only diagnostic to {config.robot_ip} ...", flush=True)
        if not robot.connect(config.robot_ip):
            raise RuntimeError("failed to connect to robot")
        _verify_feedback_advances(robot, dcss)
        data, initial_deg = _read_feedback(robot, dcss)
        _print_sample(data, initial_deg, None)

        entry_target_deg = load_entry_target_deg(config.entry_npz) if config.entry_npz is not None else None
        if entry_target_deg is not None:
            print(
                "ENTRY_INSPECTION: first_target_deg="
                f"{np.round(entry_target_deg, 3).tolist()} "
                f"max_jump_deg={float(np.max(np.abs(entry_target_deg - initial_deg))):.3f}",
                flush=True,
            )

        target_deg: np.ndarray | None = None
        if config.execute:
            if not robot.set_imp_joint_state(
                "A", velRatio=config.vel_ratio, AccRatio=config.acc_ratio, K=JOINT_K, D=JOINT_D
            ):
                raise RuntimeError("controller rejected A-arm joint-impedance setup")
            # The controller mode has changed, so every later failure must
            # still disable the arm in the finally block.
            executed = True
            time.sleep(0.3)
            data, current_deg = _read_feedback(robot, dcss)
            if data["states"][0].get("cur_state") != 3:
                raise RuntimeError("A arm did not enter joint-impedance state")
            if config.execute_entry:
                target_deg = entry_target_deg
                entry = build_quintic_entry(
                    current_deg, target_deg, duration_s=config.entry_duration_s
                )
                print(
                    f"ENTRY_EXECUTION: frames={len(entry)} duration_s={config.entry_duration_s:.1f} "
                    f"max_jump_deg={float(np.max(np.abs(target_deg - current_deg))):.3f} "
                    f"vel_ratio={config.vel_ratio} acc_ratio={config.acc_ratio}",
                    flush=True,
                )
                monitor_stride = max(1, int(round(200.0 / config.poll_hz)))
                entry_t0 = time.monotonic()
                for index, waypoint in enumerate(entry):
                    if robot.set_joint_position_cmd("A", waypoint.tolist()) is not True:
                        raise RuntimeError(f"controller rejected entry waypoint {index}")
                    if index % monitor_stride == 0 or index == len(entry) - 1:
                        data, actual_deg = _read_feedback(robot, dcss)
                        if data["states"][0].get("cur_state") != 3:
                            raise RuntimeError("A arm left joint-impedance state during entry")
                        print(
                            entry_progress_line(
                                index=index,
                                control_hz=200.0,
                                elapsed_s=time.monotonic() - entry_t0,
                            ),
                            flush=True,
                        )
                        _print_sample(data, actual_deg, target_deg=waypoint)
                        if float(np.max(np.abs(waypoint - actual_deg))) > 5.0:
                            raise RuntimeError("A-arm entry tracking error exceeded 5 deg")
                    time.sleep(1.0 / 200.0)
            else:
                target_deg = np.asarray(
                    command_single_joint_trial(
                        robot,
                        current_deg=current_deg,
                        joint_number=config.trial_joint,
                        delta_deg=config.trial_delta_deg,
                    ),
                    dtype=float,
                )
            print("COMMAND_ACCEPTED: monitoring actual feedback", flush=True)
        else:
            print("READ_ONLY: no controller state or joint target was sent", flush=True)

        deadline = time.monotonic() + config.monitor_s
        while time.monotonic() < deadline:
            data, actual_deg = _read_feedback(robot, dcss)
            _print_sample(data, actual_deg, target_deg)
            time.sleep(1.0 / config.poll_hz)
        return 0
    finally:
        if executed:
            try:
                robot.disable("A")
                print("A arm disabled after diagnostic trial", flush=True)
            except Exception as exc:
                print(f"WARNING: failed to disable A arm: {exc}", flush=True)
        try:
            robot.release_robot()
        except Exception:
            pass


def parse_args(argv: list[str] | None = None) -> ArmDiagnosticConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default=ROBOT_IP)
    parser.add_argument("--monitor-s", type=float, default=5.0)
    parser.add_argument("--poll-hz", type=float, default=20.0)
    parser.add_argument("--execute", action="store_true", help="Allow exactly one bounded A-arm test command")
    parser.add_argument("--trial-joint", type=int)
    parser.add_argument("--trial-delta-deg", type=float)
    parser.add_argument("--entry-npz", type=Path, help="Inspect or, with --execute-entry, ramp to this NPZ's first A-arm frame")
    parser.add_argument("--execute-entry", action="store_true", help="Execute a monitored A-arm entry ramp; requires --execute")
    parser.add_argument("--entry-duration-s", type=float, default=15.0)
    parser.add_argument("--vel-ratio", type=int, default=10)
    parser.add_argument("--acc-ratio", type=int, default=10)
    config = ArmDiagnosticConfig(**vars(parser.parse_args(argv)))
    validate_config(config)
    return config


def main(argv: list[str] | None = None) -> int:
    try:
        return run_diagnostic(parse_args(argv))
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
