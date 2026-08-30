#!/usr/bin/env python3
"""Conservative base-frame keyboard Cartesian jog for one physical robot arm."""

from __future__ import annotations

import argparse
import math
import sys
import termios
import time
import tty
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


KEY_DIRECTIONS: dict[str, tuple[float, float, float]] = {
    "w": (1.0, 0.0, 0.0),
    "s": (-1.0, 0.0, 0.0),
    "a": (0.0, 1.0, 0.0),
    "d": (0.0, -1.0, 0.0),
    "r": (0.0, 0.0, 1.0),
    "f": (0.0, 0.0, -1.0),
}
from robot_core.paths import local_root
ROOT = local_root()


@dataclass(frozen=True)
class JogConfig:
    """Safety-critical configuration for a keyboard jog session."""

    arm: str = "A"
    step_mm: float = 5.0
    execute: bool = False
    workspace_min: tuple[float, float, float] | None = None
    workspace_max: tuple[float, float, float] | None = None
    workspace_radius_mm: float | None = None
    vel_ratio: int = 10
    acc_ratio: int = 10
    keep_enabled: bool = False
    robot_ip: str = "192.168.1.190"
    sdk_root: Path = ROOT
    kine_config: Path = ROOT / "test" / "ccs_m6_40.MvKDCfg"


def validate_config(config: JogConfig) -> None:
    """Reject invalid motion limits before any SDK connection is opened."""
    if config.arm not in ("A", "B"):
        raise ValueError("arm must be 'A' or 'B'")
    if not math.isfinite(float(config.step_mm)) or not 0.0 < float(config.step_mm) <= 10.0:
        raise ValueError("step-mm must be in (0, 10]")
    if not (0 <= int(config.vel_ratio) <= 100):
        raise ValueError("vel-ratio must be in [0, 100]")
    if not (0 <= int(config.acc_ratio) <= 100):
        raise ValueError("acc-ratio must be in [0, 100]")
    has_explicit_workspace = config.workspace_min is not None and config.workspace_max is not None
    if config.workspace_radius_mm is not None:
        if not math.isfinite(float(config.workspace_radius_mm)) or not 0.0 < float(config.workspace_radius_mm) <= 150.0:
            raise ValueError("workspace-around-current-mm must be in (0, 150]")
        if has_explicit_workspace:
            raise ValueError("workspace bounds and workspace-around-current-mm are mutually exclusive")
    if config.execute and not has_explicit_workspace and config.workspace_radius_mm is None:
        raise ValueError("--execute requires --workspace-min and --workspace-max, or --workspace-around-current-mm")
    if (config.workspace_min is None) != (config.workspace_max is None):
        raise ValueError("workspace-min and workspace-max must be supplied together")
    if config.workspace_min is not None and config.workspace_max is not None:
        lower = np.asarray(config.workspace_min, dtype=float)
        upper = np.asarray(config.workspace_max, dtype=float)
        if lower.shape != (3,) or upper.shape != (3,) or not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("workspace bounds must contain 3 finite values")
        if not np.all(lower < upper):
            raise ValueError("workspace-min must be strictly below workspace-max")


def parse_xyz(value: str, flag_name: str) -> tuple[float, float, float]:
    """Parse exactly three finite comma-separated Cartesian coordinates."""
    try:
        values = tuple(float(item.strip()) for item in str(value).split(","))
    except ValueError as exc:
        raise ValueError(f"{flag_name} must be three comma-separated numbers") from exc
    if len(values) != 3 or not all(math.isfinite(item) for item in values):
        raise ValueError(f"{flag_name} must contain three finite values")
    return values


def parse_args(argv: list[str] | None = None) -> JogConfig:
    """Parse CLI options and reject unsafe execution before opening the SDK."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("A", "B"), default="A")
    parser.add_argument("--robot-ip", default="192.168.1.190")
    parser.add_argument("--sdk-root", type=Path, default=ROOT)
    parser.add_argument("--kine-config", type=Path, default=ROOT / "test" / "ccs_m6_40.MvKDCfg")
    parser.add_argument("--step-mm", type=float, default=5.0)
    parser.add_argument("--vel-ratio", type=int, default=10)
    parser.add_argument("--acc-ratio", type=int, default=10)
    parser.add_argument("--workspace-min", type=lambda text: parse_xyz(text, "--workspace-min"))
    parser.add_argument("--workspace-max", type=lambda text: parse_xyz(text, "--workspace-max"))
    parser.add_argument(
        "--workspace-around-current-mm",
        dest="workspace_radius_mm",
        type=float,
        default=None,
        help="Create a fixed +/- millimetre workspace around the startup feedback TCP (maximum: 150)",
    )
    parser.add_argument("--execute", action="store_true", help="Send approved planned steps to the physical robot")
    parser.add_argument("--keep-enabled", action="store_true", help="Do not disable the arm on normal exit")
    config = JogConfig(**vars(parser.parse_args(argv)))
    validate_config(config)
    return config


def inside_workspace(
    point_xyz: np.ndarray,
    lower_xyz: tuple[float, float, float],
    upper_xyz: tuple[float, float, float],
) -> bool:
    """Return whether a point lies within an inclusive Cartesian workspace box."""
    point = np.asarray(point_xyz, dtype=float)
    lower = np.asarray(lower_xyz, dtype=float)
    upper = np.asarray(upper_xyz, dtype=float)
    return bool(point.shape == (3,) and np.all(point >= lower) and np.all(point <= upper))


def workspace_bounds_from_center(center_xyz: np.ndarray, radius_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Return an axis-aligned fixed workspace centred on one feedback TCP sample."""
    center = np.asarray(center_xyz, dtype=float)
    radius = float(radius_mm)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("workspace center must contain three finite values")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("workspace radius must be positive and finite")
    return center - radius, center + radius


def key_to_delta(key: str, step_mm: float) -> tuple[float, float, float] | None:
    """Map a movement key to one base-frame Cartesian step in millimetres."""
    direction = KEY_DIRECTIONS.get(str(key).lower())
    if direction is None:
        return None
    return tuple(float(step_mm) * axis for axis in direction)


def candidate_pose(current_xyzabc: np.ndarray, delta_xyz: tuple[float, float, float]) -> np.ndarray:
    """Translate XYZ while retaining the supplied TCP ABC orientation."""
    target = np.asarray(current_xyzabc, dtype=float).copy()
    if target.shape != (6,):
        raise ValueError("current_xyzabc must contain exactly 6 values")
    target[:3] += np.asarray(delta_xyz, dtype=float)
    return target


def _arm_index(arm: str) -> int:
    return 0 if arm == "A" else 1


def _feedback(robot, dcss, arm_index: int) -> dict:
    return robot.subscribe(dcss)


def _read_feedback_pose(robot, dcss, kine, arm_index: int) -> tuple[list[float], np.ndarray]:
    """Read feedback joints and convert their FK result to XYZABC."""
    feedback = _feedback(robot, dcss, arm_index)
    joints = [float(value) for value in feedback["outputs"][arm_index]["fb_joint_pos"]]
    pose = np.asarray(kine.mat4x4_to_xyzabc(kine.fk(joints)), dtype=float)
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise RuntimeError("kinematics returned an invalid feedback pose")
    return joints, pose


def _verify_frame_updates(robot, dcss, arm_index: int) -> None:
    """Refuse execute mode when feedback frame serials are not advancing."""
    observed = set()
    for _ in range(5):
        observed.add(_feedback(robot, dcss, arm_index)["outputs"][arm_index].get("frame_serial", 0))
        # Controller feedback advances at a finite rate; allow a full 10 ms
        # sample interval rather than falsely rejecting rapid SDK reads.
        time.sleep(0.01)
    if not any(int(serial) != 0 for serial in observed) or len(observed) < 2:
        raise RuntimeError("robot feedback frame did not update")


def _trajectory_is_idle(robot, dcss, arm_index: int) -> bool:
    output = _feedback(robot, dcss, arm_index)["outputs"][arm_index]
    return output.get("traj_state") in (None, 0, b"\x00", "idle", "IDLE")


def _wait_until_planning_state(robot, dcss, arm_index: int, arm: str, timeout_s: float = 1.0) -> None:
    """Confirm the controller entered planned Cartesian position mode."""
    deadline = time.monotonic() + float(timeout_s)
    last_state: dict = {}
    while time.monotonic() < deadline:
        data = _feedback(robot, dcss, arm_index)
        last_state = data.get("states", [{}])[arm_index]
        if last_state.get("cur_state") == 1:
            return
        time.sleep(0.02)
    raise RuntimeError(
        f"{arm} arm did not enter position planning mode; "
        f"cur_state={last_state.get('cur_state')}; err_code={last_state.get('err_code')}"
    )


def _configure_planning_mode(robot, config: JogConfig, dcss, arm_index: int) -> None:
    """Set the SDK position-planning state once before executing jog steps."""
    robot.clear_set()
    robot.set_vel_acc(arm=config.arm, velRatio=int(config.vel_ratio), AccRatio=int(config.acc_ratio))
    robot.send_cmd()
    time.sleep(0.1)
    robot.clear_set()
    robot.set_state(arm=config.arm, state=1)
    robot.send_cmd()
    time.sleep(0.2)
    _wait_until_planning_state(robot, dcss, arm_index, config.arm)


def _wait_until_traj_idle(robot, dcss, arm_index: int, timeout_s: float = 3.0) -> None:
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        if _trajectory_is_idle(robot, dcss, arm_index):
            return
        time.sleep(0.02)
    raise RuntimeError("selected arm trajectory did not return idle")


def plan_or_execute_step(
    robot,
    dcss,
    kine,
    arm_index: int,
    joints: list[float],
    current_pose: np.ndarray,
    target_pose: np.ndarray,
    config: JogConfig,
) -> np.ndarray:
    """Plan one Cartesian step and send it only when execute is enabled."""
    _points, pset = kine.movLA(
        start_xyzabc=current_pose.tolist(),
        end_xyzabc=target_pose.tolist(),
        ref_joints=list(joints),
        vel=10.0,
        acc=100.0,
        freq_hz=100,
    )
    if pset is None:
        raise RuntimeError("MOVLA planning failed")
    if not config.execute:
        return target_pose
    robot.setPln_Cart(arm=config.arm, pset=pset)
    _wait_until_traj_idle(robot, dcss, arm_index)
    _joints, actual_pose = _read_feedback_pose(robot, dcss, kine, arm_index)
    return actual_pose


def _shutdown(robot, config: JogConfig, connected: bool) -> None:
    if connected and not config.keep_enabled:
        try:
            robot.clear_set()
            robot.set_state(arm=config.arm, state=0)
            robot.send_cmd()
        except Exception:
            pass
    try:
        robot.release_robot()
    except Exception:
        pass


def run_jog_session(
    config: JogConfig,
    read_key: Callable[[], str],
    sdk_factory: Callable[[], tuple[object, object, object]],
) -> list[np.ndarray]:
    """Run a key-driven jog session using an injected SDK adapter.

    The factory injection makes this control flow testable without importing or
    connecting to the vendor SDK.  Production supplies the real factory.
    """
    validate_config(config)
    robot, dcss, kine = sdk_factory()
    arm_index = _arm_index(config.arm)
    connected = False
    poses: list[np.ndarray] = []
    try:
        connected = bool(robot.connect(config.robot_ip))
        if not connected:
            raise RuntimeError("failed to connect to the robot")
        robot.check_error_and_clear(dcss)
        joints, current_pose = _read_feedback_pose(robot, dcss, kine, arm_index)
        workspace_min = config.workspace_min
        workspace_max = config.workspace_max
        if config.execute:
            _verify_frame_updates(robot, dcss, arm_index)
            if config.workspace_radius_mm is not None:
                workspace_min_array, workspace_max_array = workspace_bounds_from_center(
                    current_pose[:3], config.workspace_radius_mm
                )
                workspace_min = tuple(float(value) for value in workspace_min_array)
                workspace_max = tuple(float(value) for value in workspace_max_array)
                print(
                    "workspace_from_startup_tcp_mm=",
                    [round(float(value), 3) for value in current_pose[:3]],
                    "workspace_min_mm=",
                    [round(float(value), 3) for value in workspace_min],
                    "workspace_max_mm=",
                    [round(float(value), 3) for value in workspace_max],
                )
            _configure_planning_mode(robot, config, dcss, arm_index)
        poses.append(current_pose)
        while True:
            try:
                key = read_key()
            except StopIteration:
                break
            if not key or key.lower() == "q" or key in (" ", "\x03"):
                break
            delta = key_to_delta(key, config.step_mm)
            if delta is None:
                continue
            target_pose = candidate_pose(current_pose, delta)
            print(
                "request_xyz_mm=",
                [round(float(value), 3) for value in target_pose[:3]],
                "mode=execute" if config.execute else "mode=dry-run",
            )
            if config.execute:
                if workspace_min is None or workspace_max is None:
                    raise RuntimeError("execute workspace validation was bypassed")
                if not inside_workspace(target_pose[:3], workspace_min, workspace_max):
                    raise ValueError(f"requested pose is outside workspace: {target_pose[:3].tolist()}")
                if not _trajectory_is_idle(robot, dcss, arm_index):
                    raise RuntimeError("selected arm trajectory is not idle")
            current_pose = plan_or_execute_step(
                robot, dcss, kine, arm_index, joints, current_pose, target_pose, config,
            )
            if config.execute:
                joints, current_pose = _read_feedback_pose(robot, dcss, kine, arm_index)
            poses.append(current_pose)
            print("current_xyz_mm=", [round(float(value), 3) for value in current_pose[:3]])
        return poses
    finally:
        _shutdown(robot, config, connected)


def _real_sdk_factory(config: JogConfig) -> tuple[object, object, object]:
    """Create configured vendor SDK objects only for the interactive CLI."""
    sdk_root = Path(config.sdk_root).resolve()
    if str(sdk_root) not in sys.path:
        sys.path.insert(0, str(sdk_root))
    from SDK_PYTHON.fx_kine import Marvin_Kine
    from SDK_PYTHON.fx_robot import DCSS, Marvin_Robot

    arm_index = _arm_index(config.arm)
    kine = Marvin_Kine()
    kine.log_switch(0)
    kine_config = kine.load_config(arm_type=arm_index, config_path=str(config.kine_config))
    if not kine.initial_kine(
        robot_type=kine_config["TYPE"][arm_index],
        dh=kine_config["DH"][arm_index],
        pnva=kine_config["PNVA"][arm_index],
        j67=kine_config["BD"][arm_index],
    ):
        raise RuntimeError("initial_kine failed")
    return Marvin_Robot(), DCSS(), kine


@contextmanager
def raw_terminal_keys(stream=sys.stdin):
    """Yield a one-character reader while preserving OS Ctrl+C handling."""
    previous = termios.tcgetattr(stream.fileno())
    try:
        tty.setcbreak(stream.fileno())
        yield lambda: stream.read(1)
    finally:
        termios.tcsetattr(stream.fileno(), termios.TCSADRAIN, previous)


def main(argv: list[str] | None = None) -> int:
    """Run the interactive keyboard jog; this is the only production entrypoint."""
    try:
        config = parse_args(argv)
        print("Keyboard Cartesian jog (base frame)")
        print(f"arm={config.arm} step_mm={config.step_mm:g} execute={config.execute}")
        print("W/S: +/-X, A/D: +/-Y, R/F: +/-Z, Space/Q: stop and disable")
        print("Physical E-stop must be reachable; Space is only a software disable.")
        with raw_terminal_keys() as read_key:
            run_jog_session(config, read_key, lambda: _real_sdk_factory(config))
        return 0
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("keyboard interrupt: selected arm cleanup requested", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
