#!/usr/bin/env python3
"""Conservative single-arm Xbox Cartesian jog for the physical robot.

This tool is intentionally separate from the MuJoCo teleop entrypoint.  It
defaults to dry-run, controls exactly one arm, and sends no motion unless both
RB is held and ``--execute`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from tianji_arm.experiments.keyboard_cartesian_jog import (
    ROOT,
    _configure_planning_mode,
    _read_feedback_pose,
    _trajectory_is_idle,
    _verify_frame_updates,
    _wait_until_traj_idle,
    candidate_pose,
    inside_workspace,
    workspace_bounds_from_center,
)


_EVENT = struct.Struct("<IhBB")
_JS_EVENT_BUTTON = 0x01
_JS_EVENT_AXIS = 0x02
_JS_EVENT_INIT = 0x80
ARM_TO_SDK = {"left": "A", "right": "B"}


@dataclass(frozen=True)
class GamepadJogConfig:
    arm: Literal["right", "left"] = "right"
    device: Path = Path("/dev/input/by-id/usb-Microsoft_Xbox360_For_Windows-joystick")
    speed_mm_s: float = 10.0
    workspace_radius_mm: float = 50.0
    deadzone: float = 0.15
    control_period_s: float = 0.20
    execute: bool = False
    vel_ratio: int = 10
    acc_ratio: int = 10
    keep_enabled: bool = False
    robot_ip: str = "192.168.1.190"
    sdk_root: Path = ROOT
    kine_config: Path = ROOT / "test" / "ccs_m6_40.MvKDCfg"


def validate_config(config: GamepadJogConfig) -> None:
    if config.arm not in ARM_TO_SDK:
        raise ValueError("arm must be 'right' or 'left'")
    if not math.isfinite(config.speed_mm_s) or not 0.0 < config.speed_mm_s <= 10.0:
        raise ValueError("speed-mm-s must be in (0, 10] for physical teleoperation")
    if not math.isfinite(config.workspace_radius_mm) or not 0.0 < config.workspace_radius_mm <= 150.0:
        raise ValueError("workspace-around-current-mm must be in (0, 150]")
    if not 0.0 <= config.deadzone < 1.0:
        raise ValueError("deadzone must be in [0, 1)")
    if not math.isfinite(config.control_period_s) or not 0.05 <= config.control_period_s <= 1.0:
        raise ValueError("control-period-s must be in [0.05, 1]")
    if not 0 <= config.vel_ratio <= 100 or not 0 <= config.acc_ratio <= 100:
        raise ValueError("vel-ratio and acc-ratio must be in [0, 100]")


def shaped_axis(value: int, deadzone: float) -> float:
    normalized = float(value) / 32767.0
    if abs(normalized) <= deadzone:
        return 0.0
    return float(np.sign(normalized) * (abs(normalized) - deadzone) / (1.0 - deadzone))


def gamepad_delta_mm(axes: dict[int, int], config: GamepadJogConfig) -> np.ndarray:
    """Return one bounded base-frame XYZ increment from the documented mapping."""
    direction = np.array((
        shaped_axis(axes.get(0, 0), config.deadzone),
        -shaped_axis(axes.get(1, 0), config.deadzone),
        -shaped_axis(axes.get(4, 0), config.deadzone),
    ))
    return direction * config.speed_mm_s * config.control_period_s


class LinuxJoystick:
    """Small dependency-free reader for the Linux joystick API."""

    def __init__(self, device: Path):
        self._fd = os.open(device, os.O_RDONLY | os.O_NONBLOCK)
        self.axes: dict[int, int] = {}
        self.buttons: dict[int, int] = {}

    def poll(self) -> None:
        while True:
            try:
                payload = os.read(self._fd, _EVENT.size)
            except BlockingIOError:
                return
            if len(payload) != _EVENT.size:
                return
            _time_ms, value, event_type, number = _EVENT.unpack(payload)
            event_type &= ~_JS_EVENT_INIT
            if event_type == _JS_EVENT_AXIS:
                self.axes[number] = value
            elif event_type == _JS_EVENT_BUTTON:
                self.buttons[number] = value

    def close(self) -> None:
        os.close(self._fd)


def _arm_index(sdk_arm: str) -> int:
    return 0 if sdk_arm == "A" else 1


def _real_sdk_factory(config: GamepadJogConfig) -> tuple[object, object, object]:
    sdk_root = Path(config.sdk_root).resolve()
    if str(sdk_root) not in sys.path:
        sys.path.insert(0, str(sdk_root))
    from SDK_PYTHON.fx_kine import Marvin_Kine
    from SDK_PYTHON.fx_robot import DCSS, Marvin_Robot

    arm_index = _arm_index(ARM_TO_SDK[config.arm])
    kine = Marvin_Kine()
    kine.log_switch(0)
    kine_config = kine.load_config(arm_type=arm_index, config_path=str(config.kine_config))
    if not kine.initial_kine(
        robot_type=kine_config["TYPE"][arm_index], dh=kine_config["DH"][arm_index],
        pnva=kine_config["PNVA"][arm_index], j67=kine_config["BD"][arm_index],
    ):
        raise RuntimeError("initial_kine failed")
    return Marvin_Robot(), DCSS(), kine


def _shutdown(robot, sdk_arm: str, connected: bool, config: GamepadJogConfig) -> None:
    if connected and config.execute and not config.keep_enabled:
        try:
            robot.clear_set()
            robot.set_state(arm=sdk_arm, state=0)
            robot.send_cmd()
        except Exception:
            pass
    try:
        robot.release_robot()
    except Exception:
        pass


def run_gamepad_jog(
    config: GamepadJogConfig,
    *,
    sdk_factory=_real_sdk_factory,
    joystick_factory=LinuxJoystick,
) -> None:
    """Run one selected-arm session; this function is testable with fake adapters."""
    validate_config(config)
    sdk_arm = ARM_TO_SDK[config.arm]
    arm_index = _arm_index(sdk_arm)
    joystick = joystick_factory(config.device)
    robot, dcss, kine = sdk_factory(config)
    connected = False
    try:
        connected = bool(robot.connect(config.robot_ip))
        if not connected:
            raise RuntimeError("failed to connect to the robot")
        robot.check_error_and_clear(dcss)
        joints, current_pose = _read_feedback_pose(robot, dcss, kine, arm_index)
        lower, upper = workspace_bounds_from_center(current_pose[:3], config.workspace_radius_mm)
        print(f"Physical gamepad jog: {config.arm} arm (SDK {sdk_arm}); execute={config.execute}")
        print("Hold RB to move; left stick=X/Y; right-stick vertical=Z; Start exits.")
        print("startup_tcp_mm=", current_pose[:3].round(3).tolist(), "workspace_radius_mm=", config.workspace_radius_mm)
        if config.execute:
            _verify_frame_updates(robot, dcss, arm_index)
            setup = type("Setup", (), {"arm": sdk_arm, "vel_ratio": config.vel_ratio, "acc_ratio": config.acc_ratio})()
            _configure_planning_mode(robot, setup, dcss, arm_index)
            state = robot.subscribe(dcss)["states"][arm_index]
            if state.get("cur_state") != 1 or state.get("err_code") != 0:
                raise RuntimeError(f"{config.arm} arm is not ready: {state}")
        while True:
            joystick.poll()
            if joystick.buttons.get(7, 0):  # Xbox Start
                break
            if not joystick.buttons.get(5, 0):  # Xbox RB deadman
                time.sleep(0.01)
                continue
            delta = gamepad_delta_mm(joystick.axes, config)
            if not np.any(delta):
                time.sleep(0.01)
                continue
            target_pose = candidate_pose(current_pose, tuple(delta))
            if not inside_workspace(target_pose[:3], lower, upper):
                print("blocked: requested TCP lies outside the startup workspace")
                time.sleep(config.control_period_s)
                continue
            print("request_xyz_mm=", target_pose[:3].round(3).tolist(), "mode=execute" if config.execute else "mode=dry-run")
            if config.execute:
                if not _trajectory_is_idle(robot, dcss, arm_index):
                    print("blocked: selected arm trajectory is not idle")
                    time.sleep(config.control_period_s)
                    continue
                _points, pset = kine.movLA(
                    start_xyzabc=current_pose.tolist(), end_xyzabc=target_pose.tolist(),
                    ref_joints=list(joints), vel=config.speed_mm_s, acc=100.0, freq_hz=100,
                )
                if pset is None:
                    print("blocked: Cartesian planning failed")
                    time.sleep(config.control_period_s)
                    continue
                robot.setPln_Cart(arm=sdk_arm, pset=pset)
                _wait_until_traj_idle(robot, dcss, arm_index)
                joints, current_pose = _read_feedback_pose(robot, dcss, kine, arm_index)
            else:
                current_pose = target_pose
            time.sleep(config.control_period_s)
    finally:
        _shutdown(robot, sdk_arm, connected, config)
        joystick.close()


def parse_args(argv: list[str] | None = None) -> GamepadJogConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("right", "left"), default="right")
    parser.add_argument("--device", type=Path, default=GamepadJogConfig().device)
    parser.add_argument("--speed-mm-s", type=float, default=10.0)
    parser.add_argument("--workspace-around-current-mm", dest="workspace_radius_mm", type=float, default=50.0)
    parser.add_argument("--deadzone", type=float, default=0.15)
    parser.add_argument("--control-period-s", type=float, default=0.20)
    parser.add_argument("--vel-ratio", type=int, default=10)
    parser.add_argument("--acc-ratio", type=int, default=10)
    parser.add_argument("--robot-ip", default="192.168.1.190")
    parser.add_argument("--sdk-root", type=Path, default=ROOT)
    parser.add_argument("--kine-config", type=Path, default=ROOT / "test" / "ccs_m6_40.MvKDCfg")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--keep-enabled", action="store_true")
    config = GamepadJogConfig(**vars(parser.parse_args(argv)))
    validate_config(config)
    return config


def main(argv: list[str] | None = None) -> int:
    try:
        config = parse_args(argv)
        print("Physical E-stop must be reachable; RB is a software deadman only.")
        run_gamepad_jog(config)
        return 0
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("keyboard interrupt: selected-arm cleanup requested", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
