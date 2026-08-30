"""Simulation-only Xbox joystick Cartesian velocity teleoperation.

Axis mapping (base frame, controller-relative):
  left-stick up/down → base X   (up = +X)
  left-stick left/right → base Z (right = +Z, vertical)
  right-stick up/down → base Y  (up = +Y)

Wrist orientation (TCP / tool frame):
  right-stick left/right → yaw (rotate about tool Z)
  D-pad up/down → pitch (rotate about tool Y)
  D-pad left/right → roll (rotate about tool X)
"""

from __future__ import annotations

import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from twin_sim.robot import RightArmRobot


_EVENT = struct.Struct("<IhBB")
_JS_EVENT_BUTTON = 0x01
_JS_EVENT_AXIS = 0x02
_JS_EVENT_INIT = 0x80


@dataclass(frozen=True)
class TeleopConfig:
    device: Path = Path("/dev/input/by-id/usb-Microsoft_Xbox360_For_Windows-joystick")
    speed_mm_s: float = 100.0
    orientation_rate_dps: float = 30.0
    workspace_radius_mm: float = 350.0
    deadzone: float = 0.15
    control_dt_s: float = 0.02
    arm: Literal["right", "left"] = "right"


def shaped_axis(value: int, deadzone: float) -> float:
    normalized = float(value) / 32767.0
    if abs(normalized) <= deadzone:
        return 0.0
    return float(np.sign(normalized) * (abs(normalized) - deadzone) / (1.0 - deadzone))


class LinuxJoystick:
    """Non-blocking reader for the Linux joystick API; no third-party library."""

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


def _rot_x(angle: float) -> np.ndarray:
    """3x3 rotation matrix about the X axis (roll)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(angle: float) -> np.ndarray:
    """3x3 rotation matrix about the Y axis (pitch)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(angle: float) -> np.ndarray:
    """3x3 rotation matrix about the Z axis (yaw)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def run_gamepad_teleop(config: TeleopConfig = TeleopConfig()) -> None:
    if not 0.0 < config.speed_mm_s <= 300.0:
        raise ValueError("speed-mm-s must be in (0, 300]")
    if not 0.0 < config.workspace_radius_mm <= 350.0:
        raise ValueError("workspace-radius-mm must be in (0, 350]")
    if not 0.0 <= config.deadzone < 1.0:
        raise ValueError("deadzone must be in [0, 1)")
    if not 0.0 < config.orientation_rate_dps <= 90.0:
        raise ValueError("orientation-rate-dps must be in (0, 90]")
    if config.arm not in ("right", "left"):
        raise ValueError("arm must be 'right' or 'left'")
    joystick = LinuxJoystick(config.device)
    robot = RightArmRobot(viewer=True)
    try:
        kinematics = robot.right_kinematics if config.arm == "right" else robot.left_kinematics
        joints = robot.joint_positions if config.arm == "right" else robot.left_joint_positions
        command = robot.command if config.arm == "right" else robot.command_left
        target = kinematics.fk(joints)
        lower = target[:3, 3].copy() - config.workspace_radius_mm / 1000.0
        upper = target[:3, 3].copy() + config.workspace_radius_mm / 1000.0
        print(
            f"Gamepad teleop ({config.arm} arm): hold RB to move; "
            "left-stick up/down=X, left-stick left/right=Z, right-stick up/down=Y; Start exits."
        )
        print("Right-stick left/right=yaw, D-pad up/down=pitch, D-pad left/right=roll (TCP frame, deg/s).")
        print("startup_tcp_m=", target[:3, 3].round(4).tolist(), "workspace_radius_mm=", config.workspace_radius_mm)
        while robot._viewer is not None and robot._viewer.is_running():
            joystick.poll()
            if joystick.buttons.get(7, 0):  # Xbox Start
                break
            if joystick.buttons.get(5, 0):  # Xbox RB deadman
                # Translation velocity (m/s) in base frame
                velocity = np.array((
                    -shaped_axis(joystick.axes.get(1, 0), config.deadzone),  # left stick Y → base X
                    -shaped_axis(joystick.axes.get(4, 0), config.deadzone),  # right stick Y → base Y
                    shaped_axis(joystick.axes.get(0, 0), config.deadzone),   # left stick X → base Z
                )) * (config.speed_mm_s / 1000.0)

                # Orientation rates (rad/s) in base frame
                yaw_rate = shaped_axis(joystick.axes.get(3, 0), config.deadzone) * np.deg2rad(config.orientation_rate_dps)
                pitch_rate = -shaped_axis(joystick.axes.get(7, 0), config.deadzone) * np.deg2rad(config.orientation_rate_dps)
                roll_rate = -shaped_axis(joystick.axes.get(6, 0), config.deadzone) * np.deg2rad(config.orientation_rate_dps)

                candidate = target.copy()
                # Translation: integrate in base frame
                candidate[:3, 3] = np.clip(
                    candidate[:3, 3] + velocity * config.control_dt_s, lower, upper
                )
                # Orientation: TCP-frame small-angle rotation (right-multiply)
                if abs(yaw_rate) > 1e-6 or abs(pitch_rate) > 1e-6 or abs(roll_rate) > 1e-6:
                    R = candidate[:3, :3]
                    dR = (
                        _rot_z(yaw_rate * config.control_dt_s)
                        @ _rot_y(pitch_rate * config.control_dt_s)
                        @ _rot_x(roll_rate * config.control_dt_s)
                    )
                    candidate[:3, :3] = R @ dR  # right-multiply = TCP frame

                solved = kinematics.ik(candidate, joints)
                if solved.success:
                    target = candidate
                    command(solved.joints_rad)
                    joints = solved.joints_rad
            robot.step(config.control_dt_s)
    finally:
        robot.close()
        joystick.close()
