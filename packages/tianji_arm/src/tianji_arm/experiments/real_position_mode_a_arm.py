#!/usr/bin/env python3
"""Real A-arm position-mode debug script."""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


from robot_core.paths import local_root
ROOT = local_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class PositionConfig:
    arm: str = "A"
    robot_ip: str = "192.168.1.190"
    sdk_root: Path = ROOT
    joints: list[float] | None = None
    vel_ratio: int = 10
    acc_ratio: int = 10
    timeout_s: float = 10.0
    tolerance_deg: float = 0.5
    joint_delta_limit_deg: float = 45.0
    execute: bool = False
    keep_enabled: bool = False


def parse_args(argv: list[str] | None = None) -> PositionConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("A", "B"), default="A")
    parser.add_argument("--robot-ip", default="192.168.1.190")
    parser.add_argument("--sdk-root", type=Path, default=ROOT)
    parser.add_argument("--joints", type=parse_joints, default=None, help="Seven comma-separated target joints in degrees")
    parser.add_argument("--vel-ratio", type=int, default=10)
    parser.add_argument("--acc-ratio", type=int, default=10)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--tolerance-deg", type=float, default=0.5)
    parser.add_argument("--joint-delta-limit-deg", type=float, default=45.0)
    parser.add_argument("--execute", action="store_true", help="Send the target to the real robot")
    parser.add_argument("--keep-enabled", action="store_true", help="Do not disable the arm at the end")
    config = PositionConfig(**vars(parser.parse_args(argv)))
    validate_config(config)
    return config


def parse_joints(value: str) -> list[float]:
    try:
        joints = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--joints must be seven comma-separated floats") from exc
    if len(joints) != 7:
        raise ValueError("--joints must contain 7 values")
    if not all(math.isfinite(joint) for joint in joints):
        raise ValueError("--joints must contain 7 finite values")
    return joints


def validate_config(config: PositionConfig) -> None:
    if config.arm not in ("A", "B"):
        raise ValueError("arm must be 'A' or 'B'")
    if not (0 <= int(config.vel_ratio) <= 100):
        raise ValueError("vel-ratio must be in [0, 100]")
    if not (0 <= int(config.acc_ratio) <= 100):
        raise ValueError("acc-ratio must be in [0, 100]")
    if config.timeout_s <= 0.0:
        raise ValueError("timeout-s must be positive")
    if config.tolerance_deg <= 0.0:
        raise ValueError("tolerance-deg must be positive")
    if config.joint_delta_limit_deg <= 0.0:
        raise ValueError("joint_delta_limit_deg must be positive")
    if config.joints is not None and len(config.joints) != 7:
        raise ValueError("joints must contain 7 values")


def main(argv: list[str] | None = None) -> int:
    try:
        config = parse_args(argv)
        result = run_position_debug(config)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print("execute:", result["execute"])
    print("initial_joints:", [round(v, 3) for v in result["initial_joints"]])
    if result["target_joints"] is not None:
        print("target_joints:", [round(v, 3) for v in result["target_joints"]])
        print("final_joints:", [round(v, 3) for v in result["final_joints"]])
        print("reached:", result["reached"])
        if result.get("diagnostic"):
            print("diagnostic:", result["diagnostic"])
    else:
        print("target_joints: <none>")
    return 0


def run_position_debug(config: PositionConfig) -> dict:
    validate_config(config)
    _ensure_sdk_path(config.sdk_root)
    from SDK_PYTHON.fx_robot import DCSS, Marvin_Robot

    arm_index = 0 if config.arm == "A" else 1
    dcss = DCSS()
    robot = Marvin_Robot()
    connected = False
    try:
        connected = bool(robot.connect(config.robot_ip))
        if not connected:
            raise RuntimeError("failed to connect to the robot")
        robot.check_error_and_clear(dcss)
        _verify_frame_updates(robot, dcss, arm_index)
        robot.log_switch("1")
        robot.local_log_switch("1")

        initial = _feedback_joints(robot, dcss, arm_index)
        if config.joints is None:
            return {
                "execute": False,
                "initial_joints": initial,
                "target_joints": None,
                "final_joints": initial,
                "reached": False,
            }

        _validate_target_delta(initial, config.joints, config.joint_delta_limit_deg)
        _set_position_mode(robot, config)
        if config.execute:
            robot.clear_set()
            robot.set_joint_cmd_pose(arm=config.arm, joints=config.joints)
            robot.send_cmd()
            final, reached = _wait_until_reached(robot, dcss, arm_index, config.joints, config.timeout_s, config.tolerance_deg)
        else:
            final = _feedback_joints(robot, dcss, arm_index)
            reached = False
        feedback = robot.subscribe(dcss) if config.execute and not reached else None
        return {
            "execute": bool(config.execute),
            "initial_joints": initial,
            "target_joints": list(config.joints),
            "final_joints": final,
            "reached": reached,
            "diagnostic": _target_failure_message(config.joints, final, feedback, arm_index) if feedback else "",
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


def _feedback_joints(robot, dcss, arm_index: int) -> list[float]:
    data = robot.subscribe(dcss)
    return [float(value) for value in data["outputs"][arm_index]["fb_joint_pos"]]


def _validate_target_delta(current: list[float], target: list[float], limit_deg: float) -> None:
    deltas = [abs(float(t) - float(c)) for c, t in zip(current, target)]
    if max(deltas) > float(limit_deg):
        raise ValueError(
            f"target joint delta exceeds {float(limit_deg):.1f} deg; "
            f"max delta is {max(deltas):.1f} deg"
        )


def _set_position_mode(robot, config: PositionConfig) -> None:
    robot.clear_set()
    robot.set_state(arm=config.arm, state=1)
    robot.set_vel_acc(arm=config.arm, velRatio=int(config.vel_ratio), AccRatio=int(config.acc_ratio))
    robot.send_cmd()
    time.sleep(0.5)


def _send_cmd_prefer_wait(robot, timeout_ms: int = 100) -> object:
    if hasattr(robot, "send_cmd_wait_response"):
        return robot.send_cmd_wait_response(int(timeout_ms))
    return robot.send_cmd()


def _target_failure_message(target: Sequence[float], final: Sequence[float], feedback: dict, arm_index: int) -> str:
    errors = [abs(float(actual) - float(expected)) for actual, expected in zip(final, target)]
    max_error = max(errors) if errors else float("nan")
    state = feedback.get("states", [{}])[arm_index] if feedback.get("states") else {}
    output = feedback.get("outputs", [{}])[arm_index] if feedback.get("outputs") else {}
    rounded_final = [round(float(value), 3) for value in final]
    rounded_target = [round(float(value), 3) for value in target]
    return (
        "target was not reached; "
        f"max_error_deg={max_error:.3f}; "
        f"cur_state={state.get('cur_state')}; "
        f"err_code={state.get('err_code')}; "
        f"traj_state={output.get('traj_state')!r}; "
        f"target_joints={rounded_target}; "
        f"final_joints={rounded_final}"
    )


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


if __name__ == "__main__":
    raise SystemExit(main())
