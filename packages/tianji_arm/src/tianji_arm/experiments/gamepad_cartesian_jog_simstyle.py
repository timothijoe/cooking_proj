#!/usr/bin/env python3
"""Sim-style single-arm Xbox Cartesian jog for the physical robot.

Uses 50 Hz continuous FK/IK velocity control, matching the MuJoCo simulation
architecture (``twin_sim/gamepad_teleop.py``).  Each control cycle:

  1. Integrates joystick velocity → target TCP pose (mm, deg)
  2. SDK IK → target joint angles (deg)
  3. Sends joint command via ``set_joint_cmd_pose``

Axis mapping (base frame, controller-relative):
  left-stick up/down → base X   (up = +X)
  left-stick left/right → base Z (right = +Z, vertical)
  right-stick up/down → base Y  (up = +Y)

Wrist orientation (TCP / tool frame):
  right-stick left/right → yaw (rotate about tool Z)
  D-pad up/down → pitch (rotate about tool Y)
  D-pad left/right → roll (rotate about tool X)

This avoids the inflexibility of MOVLA single-segment planning: the caller
can freely sequence or interleave motions without waiting for a trajectory
to complete.  The control loop, velocity scale, and workspace defaults are
all aligned with the simulation.

Defaults to dry-run; ``--execute`` is required to send motion commands.
RB is a deadman switch; Start exits.
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
from typing import Callable, Literal

import numpy as np

from tianji_arm.experiments.keyboard_cartesian_jog import (
    ROOT,
    _verify_frame_updates,
    inside_workspace,
    workspace_bounds_from_center,
)

_EVENT = struct.Struct("<IhBB")
_JS_EVENT_BUTTON = 0x01
_JS_EVENT_AXIS = 0x02
_JS_EVENT_INIT = 0x80
ARM_TO_SDK: dict[str, str] = {"left": "A", "right": "B"}


@dataclass(frozen=True)
class SimStyleJogConfig:
    """Configuration matching the simulation teleop scale and safety limits.

    Parameters
    ----------
    arm : "right" | "left"
        Which arm to control.
    device : Path
        Linux joystick device path.
    speed_mm_s : float
        Requested Cartesian speed in mm/s (default 100, matches simulation).
    workspace_radius_mm : float
        Half-side-length of the axis-aligned cubic workspace centred on the
        startup TCP (default 350, matches simulation).
    deadzone : float
        Joystick deadzone fraction [0, 1).
    control_period_s : float
        Control period in seconds (default 0.02 = 50 Hz, matches simulation).
    execute : bool
        If True, send joint commands to the physical robot.
    vel_ratio : int
        SDK velocity ratio [0, 100].
    acc_ratio : int
        SDK acceleration ratio [0, 100].
    keep_enabled : bool
        If True, do not disable the arm on normal exit.
    robot_ip : str
        IP address of the robot controller.
    sdk_root : Path
        Path to the SDK root directory.
    kine_config : Path
        Path to the robot kinematics configuration file.
    max_speed_mm_s : float
        Absolute upper bound for speed-mm-s (default 300).
    orientation_rate_dps : float
        Wrist orientation speed in degrees/second (default 30).
    """

    arm: Literal["right", "left"] = "right"
    device: Path = Path("/dev/input/by-id/usb-Microsoft_Xbox360_For_Windows-joystick")
    speed_mm_s: float = 100.0
    workspace_radius_mm: float = 350.0
    deadzone: float = 0.15
    control_period_s: float = 0.02
    execute: bool = False
    vel_ratio: int = 50
    acc_ratio: int = 50
    keep_enabled: bool = False
    robot_ip: str = "192.168.1.190"
    sdk_root: Path = ROOT
    kine_config: Path = ROOT / "test" / "ccs_m6_40.MvKDCfg"
    max_speed_mm_s: float = 300.0
    orientation_rate_dps: float = 30.0


def validate_config(config: SimStyleJogConfig) -> None:
    """Reject unsafe motion limits before any SDK connection is opened.

    Raises ValueError on any violation.
    """
    if config.arm not in ARM_TO_SDK:
        raise ValueError("arm must be 'right' or 'left'")
    if not math.isfinite(config.speed_mm_s) or not 0.0 < config.speed_mm_s <= config.max_speed_mm_s:
        raise ValueError(f"speed-mm-s must be in (0, {config.max_speed_mm_s}]")
    if not math.isfinite(config.workspace_radius_mm) or not 0.0 < config.workspace_radius_mm <= 500.0:
        raise ValueError("workspace-around-current-mm must be in (0, 500]")
    if not 0.0 <= config.deadzone < 1.0:
        raise ValueError("deadzone must be in [0, 1)")
    if not math.isfinite(config.control_period_s) or not 0.01 <= config.control_period_s <= 0.1:
        raise ValueError("control-period-s must be in [0.01, 0.1]")
    if not 0 <= config.vel_ratio <= 100 or not 0 <= config.acc_ratio <= 100:
        raise ValueError("vel-ratio and acc-ratio must be in [0, 100]")
    if not math.isfinite(config.orientation_rate_dps) or not 0.0 < config.orientation_rate_dps <= 90.0:
        raise ValueError("orientation-rate-dps must be in (0, 90]")


# ---------------------------------------------------------------------------
# Joystick helpers
# ---------------------------------------------------------------------------

def shaped_axis(value: int, deadzone: float) -> float:
    """Normalise a raw axis value to [-1, 1] with deadzone, matching simulation.

    Parameters
    ----------
    value : int
        Raw axis reading from the joystick (range approximately +/- 32767).
    deadzone : float
        Fractional deadzone in [0, 1).

    Returns
    -------
    float
        Normalised value in [-1, 1]; 0 when ``abs(value / 32767) <= deadzone``.
    """
    normalized = float(value) / 32767.0
    if abs(normalized) <= deadzone:
        return 0.0
    return float(np.sign(normalized) * (abs(normalized) - deadzone) / (1.0 - deadzone))


def _arm_index(sdk_arm: str) -> int:
    """Map SDK arm label to index (0 = A, 1 = B)."""
    return 0 if sdk_arm == "A" else 1


def _rot_x(angle_rad: float) -> np.ndarray:
    """3x3 rotation matrix about the X axis (roll)."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(angle_rad: float) -> np.ndarray:
    """3x3 rotation matrix about the Y axis (pitch)."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(angle_rad: float) -> np.ndarray:
    """3x3 rotation matrix about the Z axis (yaw)."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _mat4x4_to_xyzabc_euler(mat: np.ndarray) -> list[float]:
    """Convert a 4x4 homogeneous matrix to [x, y, z, a, b, c] (mm, deg).

    Uses ZYX (intrinsic) Euler angle convention: a=yaw(Z), b=pitch(Y), c=roll(X).
    This matches the SDK's own ``mat4x4_to_xyzabc`` convention.
    """
    x, y, z = mat[0, 3], mat[1, 3], mat[2, 3]
    R = mat[:3, :3]
    # ZYX intrinsic: a = atan2(R21, R11), b = -asin(R31), c = atan2(R32, R33)
    a = float(np.rad2deg(np.arctan2(R[1, 0], R[0, 0])))
    sin_b = -R[2, 0]
    sin_b_clipped = float(np.clip(sin_b, -1.0, 1.0))
    b = float(np.rad2deg(np.arcsin(sin_b_clipped)))
    c = float(np.rad2deg(np.arctan2(R[2, 1], R[2, 2])))
    return [x, y, z, a, b, c]


class LinuxJoystick:
    """Non-blocking reader for the Linux joystick API; no third-party library."""

    def __init__(self, device: Path):
        self._fd = os.open(device, os.O_RDONLY | os.O_NONBLOCK)
        self.axes: dict[int, int] = {}
        self.buttons: dict[int, int] = {}

    def poll(self) -> None:
        """Read all pending joystick events, updating internal axis/button state."""
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


# ---------------------------------------------------------------------------
# SDK wrappers
# ---------------------------------------------------------------------------

def _create_sdk(config: SimStyleJogConfig) -> tuple:
    """Create and configure vendor SDK objects for FK/IK.

    Returns
    -------
    tuple
        ``(Marvin_Robot, DCSS, Marvin_Kine)`` — all configured and ready.
    """
    sdk_root = Path(config.sdk_root).resolve()
    if str(sdk_root) not in sys.path:
        sys.path.insert(0, str(sdk_root))
    from SDK_PYTHON.fx_kine import Marvin_Kine
    from SDK_PYTHON.fx_robot import DCSS, Marvin_Robot

    arm_index = _arm_index(ARM_TO_SDK[config.arm])
    kine = Marvin_Kine()
    kine.log_switch(0)
    cfg = kine.load_config(arm_type=arm_index, config_path=str(config.kine_config))
    if cfg is None:
        raise RuntimeError("load_config failed")
    if not kine.initial_kine(
        robot_type=cfg["TYPE"][arm_index],
        dh=cfg["DH"][arm_index],
        pnva=cfg["PNVA"][arm_index],
        j67=cfg["BD"][arm_index],
    ):
        raise RuntimeError("initial_kine failed")
    return Marvin_Robot(), DCSS(), kine


def _read_feedback_pose(
    robot, dcss, kine, arm_index: int
) -> tuple[list[float], list[float]]:
    """Read feedback joints and return ``(joints_deg, xyzabc)`` via SDK FK.

    Parameters
    ----------
    robot : Marvin_Robot
        Connected robot instance.
    dcss : DCSS
        Data subscription structure.
    kine : Marvin_Kine
        Initialised kinematics instance.
    arm_index : int
        0 for arm A, 1 for arm B.

    Returns
    -------
    tuple
        ``(joints_deg, xyzabc)`` where *joints_deg* is a 7-element list of
        joint angles in degrees, and *xyzabc* is a 6-element list ``[x, y, z,
        a, b, c]`` (mm and degrees).
    """
    fb = robot.subscribe(dcss)
    joints = [float(v) for v in fb["outputs"][arm_index]["fb_joint_pos"]]
    fk_mat = kine.fk(joints)
    if not fk_mat:
        raise RuntimeError("FK returned False for feedback joints")
    xyzabc = kine.mat4x4_to_xyzabc(fk_mat)
    if not xyzabc:
        raise RuntimeError("mat4x4_to_xyzabc returned False")
    return joints, [float(v) for v in xyzabc]


def _ik_target_pose(
    kine, target_xyzabc: list[float], ref_joints: list[float]
) -> list[float] | None:
    """Solve IK for a target XYZABC pose, return joint angles (deg) or None.

    Parameters
    ----------
    kine : Marvin_Kine
        Initialised kinematics instance.
    target_xyzabc : list[float]
        6-element target pose ``[x, y, z, a, b, c]`` (mm, deg).
    ref_joints : list[float]
        7-element reference joint angles (deg) — the solver prefers a solution
        close to this configuration.

    Returns
    -------
    list[float] | None
        7-element joint angles in degrees, or None if IK failed.
    """
    from SDK_PYTHON.fx_kine import FX_InvKineSolvePara

    # XYZABC → 4x4 matrix → 16-value flat list (row-major)
    mat4x4 = kine.xyzabc_to_mat4x4(target_xyzabc)
    if not mat4x4:
        return None
    flat = kine.mat4x4_to_mat1x16(mat4x4)

    params = FX_InvKineSolvePara()
    params.set_input_ik_target_tcp(flat)

    # Sanity guard: the SDK IK solver requires joint 4 to be non-zero
    # (its doc: "参考角第四关节不能为零"); passing zero makes IK fail.
    # Nudge it to a small non-zero value in that case.
    safe_ref = list(ref_joints)
    if abs(safe_ref[3]) < 1e-9:
        safe_ref[3] = 0.001

    params.set_input_ik_ref_joint(safe_ref)
    params.set_input_ik_zsp_type(0)  # 0 = closest to reference joints (Euclidean)
    if not kine.ik(params):
        return None
    return params.get_output_ret_joint()


def _configure_position_mode(
    robot, config: SimStyleJogConfig, dcss, arm_index: int
) -> None:
    """Set the robot to position mode (state=1) for joint position control.

    Raises RuntimeError if the arm does not enter position mode within 1 s.
    """
    robot.clear_set()
    robot.set_vel_acc(
        arm=ARM_TO_SDK[config.arm],
        velRatio=config.vel_ratio,
        AccRatio=config.acc_ratio,
    )
    robot.send_cmd()
    time.sleep(0.1)
    robot.clear_set()
    robot.set_state(arm=ARM_TO_SDK[config.arm], state=1)
    robot.send_cmd()
    time.sleep(0.2)

    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        data = robot.subscribe(dcss)
        state = data["states"][arm_index]
        if state.get("cur_state") == 1:
            print(f"Position mode confirmed (state=1, err={state.get('err_code')})")
            return
        time.sleep(0.02)
    raise RuntimeError(
        f"arm did not enter position mode; "
        f"cur_state={state.get('cur_state')}, err_code={state.get('err_code')}"
    )


def _is_state_valid(robot, dcss, arm_index: int) -> bool:
    """Check that the arm is in a valid state (err_code==0).

    The arm may start in either state=1 (position mode) or state=3
    (torque/impedance mode).  ``_configure_position_mode`` handles the
    transition to state=1.
    """
    data = robot.subscribe(dcss)
    state = data["states"][arm_index]
    return state.get("err_code") == 0


def _shutdown(robot, config: SimStyleJogConfig, connected: bool) -> None:
    """Disable the arm and release the robot SDK connection."""
    if connected and config.execute and not config.keep_enabled:
        try:
            robot.clear_set()
            robot.set_state(arm=ARM_TO_SDK[config.arm], state=0)
            robot.send_cmd()
        except Exception:
            pass
    try:
        robot.release_robot()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main jog session
# ---------------------------------------------------------------------------

def run_simstyle_jog(
    config: SimStyleJogConfig,
    *,
    sdk_factory: Callable[[SimStyleJogConfig], tuple] = _create_sdk,
    joystick_factory: Callable[[Path], LinuxJoystick] = LinuxJoystick,
) -> None:
    """Run a 50 Hz FK/IK continuous-velocity jog session.

    This is the testable entry point: inject ``sdk_factory`` and
    ``joystick_factory`` to substitute fakes in unit tests.

    Parameters
    ----------
    config : SimStyleJogConfig
        Session configuration.
    sdk_factory : callable
        Callable ``config → (robot, dcss, kine)``.
    joystick_factory : callable
        Callable ``device_path → LinuxJoystick``.
    """
    validate_config(config)
    sdk_arm = ARM_TO_SDK[config.arm]
    arm_index = _arm_index(sdk_arm)
    joystick = joystick_factory(config.device)
    robot, dcss, kine = sdk_factory(config)
    connected = False

    try:
        # --- Connect ---
        connected = bool(robot.connect(config.robot_ip))
        if not connected:
            raise RuntimeError("failed to connect to the robot")
        robot.check_error_and_clear(dcss)
        print(f"Connected to {config.robot_ip}")

        # --- Verify frame updates (execute mode only) ---
        if config.execute:
            _verify_frame_updates(robot, dcss, arm_index)

        # --- Read startup pose ---
        fb_joints, fb_xyzabc = _read_feedback_pose(robot, dcss, kine, arm_index)
        lower, upper = workspace_bounds_from_center(
            np.array(fb_xyzabc[:3], dtype=float), config.workspace_radius_mm
        )

        # Track the last-sent target (matches simulation pattern where joint
        # state is the commanded target, not the feedback).
        current_joints = list(fb_joints)
        current_xyzabc = list(fb_xyzabc)

        print(f"Sim-style gamepad jog: {config.arm} arm (SDK {sdk_arm}); execute={config.execute}")
        print("Hold RB to move; left-stick up/down = X, left-stick left/right = Z, right-stick up/down = Y; Start exits.")
        print("Right-stick left/right = yaw, D-pad up/down = pitch, D-pad left/right = roll (deg/s).")
        print(
            f"startup_tcp_mm=({fb_xyzabc[0]:.1f}, {fb_xyzabc[1]:.1f}, {fb_xyzabc[2]:.1f})  "
            f"workspace_radius_mm={config.workspace_radius_mm}"
        )

        # --- Configure position mode ---
        if config.execute:
            if not _is_state_valid(robot, dcss, arm_index):
                raise RuntimeError(f"{config.arm} arm is not ready at startup")
            _configure_position_mode(robot, config, dcss, arm_index)

        # --- Main 50 Hz control loop ---
        while True:
            joystick.poll()

            # Start button → exit
            if joystick.buttons.get(7, 0):
                print("Start pressed: exiting")
                break

            # RB deadman: must be held for motion
            if not joystick.buttons.get(5, 0):
                time.sleep(0.001)
                continue

            # Compute Cartesian velocity (mm/s) from joystick axes.
            # The user-specified mapping overrides the simulation default:
            #   left stick  Y (axis 1, push-up +) → base X
            #   left stick  X (axis 0, push-right +) → base Z (vertical)
            #   right stick Y (axis 4, push-up +) → base Y
            vx = -shaped_axis(joystick.axes.get(1, 0), config.deadzone) * config.speed_mm_s
            vy = -shaped_axis(joystick.axes.get(4, 0), config.deadzone) * config.speed_mm_s
            vz = shaped_axis(joystick.axes.get(0, 0), config.deadzone) * config.speed_mm_s

            # Orientation rates (deg/s): right stick X → yaw (A),
            # D-pad up/down → pitch (B), D-pad left/right → roll (C).
            yaw_rate = shaped_axis(joystick.axes.get(3, 0), config.deadzone) * config.orientation_rate_dps
            pitch_rate = -shaped_axis(joystick.axes.get(7, 0), config.deadzone) * config.orientation_rate_dps
            roll_rate = -shaped_axis(joystick.axes.get(6, 0), config.deadzone) * config.orientation_rate_dps

            # Skip if no significant motion in either translation or orientation
            dt = config.control_period_s
            if (abs(vx) < 0.1 and abs(vy) < 0.1 and abs(vz) < 0.1
                    and abs(yaw_rate * dt) < 0.01 and abs(pitch_rate * dt) < 0.01
                    and abs(roll_rate * dt) < 0.01):
                time.sleep(0.001)
                continue

            # Integrate velocity → target position (mm), clip to workspace
            tx = float(np.clip(current_xyzabc[0] + vx * dt, lower[0], upper[0]))
            ty = float(np.clip(current_xyzabc[1] + vy * dt, lower[1], upper[1]))
            tz = float(np.clip(current_xyzabc[2] + vz * dt, lower[2], upper[2]))

            # Integrate orientation about the TCP (tool) frame: apply small-angle
            # rotations in the current tool axes (right-multiply the rotation):
            #   target_R = current_R @ Rz(yaw) @ Ry(pitch) @ Rx(roll)
            # This is the TCP-frame convention (tool-relative yaw/pitch/roll).
            current_mat = kine.xyzabc_to_mat4x4(current_xyzabc)
            if not current_mat:
                current_mat = np.eye(4)
            current_mat = np.asarray(current_mat, dtype=float)
            dR = (
                _rot_z(np.deg2rad(yaw_rate * dt))
                @ _rot_y(np.deg2rad(pitch_rate * dt))
                @ _rot_x(np.deg2rad(roll_rate * dt))
            )
            target_R = np.asarray(current_mat[:3, :3], dtype=float) @ dR
            target_mat = np.asarray(current_mat, dtype=float).copy()
            target_mat[:3, :3] = target_R
            target_mat[:3, 3] = [tx, ty, tz]
            # Use the SDK's own mat4x4_to_xyzabc to ensure the Euler angle
            # convention matches what xyzabc_to_mat4x4 uses downstream.
            xyzabc_result = kine.mat4x4_to_xyzabc(target_mat.tolist())
            if not xyzabc_result:
                print("mat4x4_to_xyzabc failed, keeping previous pose")
                time.sleep(config.control_period_s)
                continue
            target_xyzabc = [float(v) for v in xyzabc_result]

            # Skip sub-millimetre / sub-degree requests
            if (abs(target_xyzabc[0] - current_xyzabc[0]) < 0.01
                    and abs(target_xyzabc[1] - current_xyzabc[1]) < 0.01
                    and abs(target_xyzabc[2] - current_xyzabc[2]) < 0.01
                    and abs(target_xyzabc[3] - current_xyzabc[3]) < 0.01
                    and abs(target_xyzabc[4] - current_xyzabc[4]) < 0.01
                    and abs(target_xyzabc[5] - current_xyzabc[5]) < 0.01):
                time.sleep(0.001)
                continue

            print(
                f"request_xyz_mm=({target_xyzabc[0]:.1f}, {target_xyzabc[1]:.1f}, "
                f"{target_xyzabc[2]:.1f})  "
                f"mode={'execute' if config.execute else 'dry-run'}"
            )

            if config.execute:
                # IK: target pose → joint angles, using last-sent joints as
                # reference for smoothness (matching simulation pattern).
                target_joints = _ik_target_pose(kine, target_xyzabc, current_joints)
                if target_joints is None:
                    print("IK failed, keeping previous pose")
                    time.sleep(config.control_period_s)
                    continue

                robot.clear_set()
                robot.set_joint_cmd_pose(arm=sdk_arm, joints=target_joints)
                robot.send_cmd()

                current_joints = list(target_joints)
                current_xyzabc = list(target_xyzabc)
            else:
                # Dry-run: just advance the virtual pose
                current_xyzabc = list(target_xyzabc)

            time.sleep(config.control_period_s)

    finally:
        _shutdown(robot, config, connected)
        joystick.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> SimStyleJogConfig:
    """Parse CLI arguments and validate the resulting configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("right", "left"), default="right")
    parser.add_argument("--device", type=Path, default=SimStyleJogConfig().device)
    parser.add_argument("--speed-mm-s", type=float, default=100.0)
    parser.add_argument(
        "--workspace-around-current-mm",
        dest="workspace_radius_mm",
        type=float,
        default=350.0,
    )
    parser.add_argument("--deadzone", type=float, default=0.15)
    parser.add_argument("--control-period-s", type=float, default=0.02)
    parser.add_argument("--vel-ratio", type=int, default=50)
    parser.add_argument("--acc-ratio", type=int, default=50)
    parser.add_argument("--robot-ip", default="192.168.1.190")
    parser.add_argument("--sdk-root", type=Path, default=ROOT)
    parser.add_argument("--kine-config", type=Path, default=ROOT / "test" / "ccs_m6_40.MvKDCfg")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--keep-enabled", action="store_true")
    parser.add_argument("--max-speed-mm-s", type=float, default=300.0)
    parser.add_argument("--orientation-rate-dps", type=float, default=30.0)
    config = SimStyleJogConfig(**vars(parser.parse_args(argv)))
    validate_config(config)
    return config


def main(argv: list[str] | None = None) -> int:
    try:
        config = parse_args(argv)
        print("Physical E-stop must be reachable; RB is a software deadman only.")
        print(f"Control period: {config.control_period_s}s ({1.0 / config.control_period_s:.0f} Hz)")
        run_simstyle_jog(config)
        return 0
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("keyboard interrupt: selected-arm cleanup requested", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())