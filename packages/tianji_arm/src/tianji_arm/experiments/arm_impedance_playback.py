#!/usr/bin/env python3
"""Real robot joint-impedance playback from an offline NPZ trajectory.

Loads arm targets from a 200 Hz NPZ, connects to the real robot, and plays
back the trajectory in joint impedance mode at a configurable speed scale.

USAGE:
    PYTHONPATH=. python3 real_robot_debug/arm_impedance_playback.py             # dry-run (A arm)
    PYTHONPATH=. python3 real_robot_debug/arm_impedance_playback.py --arm A --execute
    PYTHONPATH=. python3 real_robot_debug/arm_impedance_playback.py --arm B --execute
    PYTHONPATH=. python3 real_robot_debug/arm_impedance_playback.py --arm AB --execute

Safety:
    - Default is dry-run (no robot commands).
    - Use low --speed-scale (e.g. 0.05) for first real test.
    - Ctrl+C stops the arm(s) and disables them.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np


from robot_core.paths import local_root
ROOT = local_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SOURCE = ROOT / "recordings" / "recorded_hand_guarded_chop_200hz_official_right_retarget_candidate.npz"
DEFAULT_KINE_CONFIG = ROOT / "test" / "ccs_m6_40.MvKDCfg"
ROBOT_IP = "192.168.1.190"

# Default joint impedance parameters (conservative)
DEFAULT_JOINT_K = (8.0, 8.0, 8.0, 4.0, 2.0, 1.5, 1.0)
DEFAULT_JOINT_D = (0.8, 0.8, 0.8, 0.6, 0.4, 0.3, 0.2)

# Joint limits (degrees)
JOINT_LIMITS_DEG = [
    (-170, 170), (-100, 120), (-170, 170), (-130, 130),
    (-170, 170), (-90, 220), (-170, 170),
]

# Arm registry: singles + both
ARM_REGISTRY = {
    "A":  {"sdk_arm": "A", "index": 0, "npz_key": "left_arm_target_rad",  "name": "Left  (A)"},
    "B":  {"sdk_arm": "B", "index": 1, "npz_key": "right_arm_target_rad", "name": "Right (B)"},
    "AB": None,  # resolved below
}
ARM_REGISTRY["AB"] = {"sdk_arm": "AB", "index": 0, "npz_key": None, "name": "Both (A+B)?"}

# ── Signal handling ──
_interrupted = False


def _on_sigint(signum, frame):
    global _interrupted
    _interrupted = True
    print("\n⚠️  Ctrl+C — stopping arm(s) ...")


signal.signal(signal.SIGINT, _on_sigint)


# ── Helpers ──


def resolve_arms(arm_choice: str) -> list[dict]:
    """Return the list of arm dicts to control based on --arm choice."""
    choice = arm_choice.strip().upper()
    if choice == "AB":
        return [ARM_REGISTRY["A"], ARM_REGISTRY["B"]]
    if choice in ("A", "B"):
        return [ARM_REGISTRY[choice]]
    raise ValueError("--arm must be one of 'A', 'B', 'AB'")


def load_arm_trajectory(path: Path, arm: dict) -> tuple[np.ndarray, np.ndarray]:
    """Load one arm's trajectory from the NPZ. Returns (time_s, targets_deg)."""
    with np.load(path, allow_pickle=False) as data:
        time_s = np.asarray(data["time_s"], dtype=float)
        targets_rad = np.asarray(data[arm["npz_key"]], dtype=float)
    if time_s.ndim != 1 or time_s.size < 2 or not np.all(np.isfinite(time_s)):
        raise ValueError("time_s must be finite 1-D array with >=2 samples")
    intervals = np.diff(time_s)
    if np.any(intervals <= 0) or not np.allclose(intervals, intervals[0], rtol=0, atol=1e-9):
        raise ValueError("time_s must be strictly increasing and uniform")
    if targets_rad.shape != (time_s.size, 7) or not np.all(np.isfinite(targets_rad)):
        raise ValueError(f"{arm['npz_key']} must be finite (N,7) matching time_s")
    return time_s - time_s[0], np.rad2deg(targets_rad)


def check_joint_limits(targets_deg: np.ndarray, arm_name: str):
    """Verify trajectory joints are within hardware limits (1 deg margin)."""
    for j in range(7):
        lo, hi = JOINT_LIMITS_DEG[j]
        col = targets_deg[:, j]
        if col.min() < lo + 1.0 or col.max() > hi - 1.0:
            raise ValueError(
                f"{arm_name} Joint {j+1}: [{col.min():.1f}, {col.max():.1f}] deg "
                f"violates limit [{lo}, {hi}] (1 deg margin)"
            )


def build_entry(start_deg: np.ndarray, first_target_deg: np.ndarray,
                duration_s: float = 10.0, control_hz: float = 200.0) -> np.ndarray:
    """Build a quintic (smooth) entry trajectory from current pose to first target."""
    steps = max(int(round(duration_s * control_hz)), 1)
    phase = np.linspace(0.0, 1.0, steps + 1)
    smooth = 10 * phase**3 - 15 * phase**4 + 6 * phase**5
    result = start_deg + smooth[:, None] * (first_target_deg - start_deg)
    result[0] = start_deg
    result[-1] = first_target_deg
    return result


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-npz", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--robot-ip", default=ROBOT_IP)
    parser.add_argument("--kine-config", type=Path, default=DEFAULT_KINE_CONFIG)
    parser.add_argument("--arm", choices=("A", "B", "AB"), default="A",
                        help="Which arm(s) to control: A (left), B (right), AB (both). Default: A")
    parser.add_argument("--speed-scale", type=float, default=0.05,
                        help="Playback speed (0.01-1.0).")
    parser.add_argument("--entry-duration-s", type=float, default=15.0,
                        help="Smooth ramp time from current to first frame.")
    parser.add_argument("--execute", action="store_true",
                        help="Send commands to the real robot.")
    parser.add_argument("--no-keep-enabled", action="store_true",
                        help="Disable arm(s) after playback.")
    args = parser.parse_args(argv)
    if not 0 < args.speed_scale <= 1.0:
        raise ValueError("--speed-scale must be in (0, 1.0]")
    if args.entry_duration_s <= 0:
        raise ValueError("--entry-duration-s must be positive")
    if not args.source_npz.is_file():
        raise ValueError(f"source NPZ not found: {args.source_npz}")
    return args


def main(argv: list[str] | None = None) -> int:
    global _interrupted

    try:
        args = parse_args(argv)
        arms = resolve_arms(args.arm)
    except ValueError as e:
        print(f"error: {e}")
        return 2

    arm_names = " + ".join(a["name"] for a in arms)
    print(f"Selected arms: {arm_names}")

    # ── Load trajectory for each arm ──
    arm_trajs = {}
    total_frames = None
    total_duration = None
    for arm in arms:
        time_s, targets_deg = load_arm_trajectory(args.source_npz, arm)
        check_joint_limits(targets_deg, arm["name"])
        arm_trajs[arm["sdk_arm"]] = {"time_s": time_s, "targets_deg": targets_deg}
        if total_frames is None:
            total_frames = time_s.size
            total_duration = float(time_s[-1])
        print(f"\n{arm['name']} (degrees):")
        for j in range(7):
            print(f"  Joint {j+1}: [{targets_deg[:,j].min():7.1f}, {targets_deg[:,j].max():7.1f}]")
        print(f"  Max frame step: {np.abs(np.diff(targets_deg, axis=0)).max():.2f} deg")
        print("  ✅ Joint limits OK (1 deg margin)")

    source_dt = 0.005  # 200 Hz
    play_dt = source_dt / args.speed_scale
    play_duration = total_duration / args.speed_scale

    print(f"\nTotal frames:      {total_frames}")
    print(f"Source DT:         5 ms (200 Hz)")
    print(f"Play DT:           {play_dt*1000:.1f} ms (speed={args.speed_scale}x)")
    print(f"Source duration:   {total_duration:.1f}s")
    print(f"Playback duration: {play_duration:.1f}s")

    if not args.execute:
        print(f"\n[Dry-run] Entry: {args.entry_duration_s}s quintic ramp → first frame")
        print(f"[Dry-run] Playback: {total_frames} frames at {args.speed_scale}x")
        print(f"[Dry-run] Arm(s): {arm_names}")
        print("[Dry-run] No robot commands sent.")
        print("\nTo execute on the real robot, add --execute")
        return 0

    # ═════════════════════════════════════════════════════════
    # Real robot execution
    # ═════════════════════════════════════════════════════════

    from SDK_PYTHON.fx_kine import Marvin_Kine
    from SDK_PYTHON.fx_robot import DCSS, Marvin_Robot, Concise_Marvin_Robot

    dcss = DCSS()
    base_robot = Marvin_Robot()
    robot = Concise_Marvin_Robot()
    kine = Marvin_Kine()
    connected = False

    try:
        # ── Connect + clear errors using base SDK ──
        print(f"\nConnecting to robot at {args.robot_ip} ...")
        connected = bool(base_robot.connect(args.robot_ip))
        if not connected:
            raise RuntimeError("failed to connect to robot (base SDK)")
        print("✅ Connected")

        base_robot.check_error_and_clear(dcss)
        base_robot.log_switch("1")
        base_robot.local_log_switch("1")
        time.sleep(0.3)

        # Verify + clear errors on selected arms
        sub = base_robot.subscribe(dcss)
        for arm in arms:
            idx = arm["index"]
            cur_state = sub["states"][idx]["cur_state"]
            err_code = sub["states"][idx]["err_code"]
            raw = [float(v) for v in sub["outputs"][idx]["fb_joint_pos"]]
            print(f"{arm['name']}: state={cur_state}, err_code={err_code}, "
                  f"pos={[round(v,1) for v in raw]}")
            if cur_state == 100:
                print(f"  ⚠️ {arm['name']} in error state (100), attempting clear ...")
                base_robot.clear_set()
                base_robot.clear_error(arm["sdk_arm"])
                base_robot.send_cmd()
                time.sleep(0.5)
                sub = base_robot.subscribe(dcss)
                cur_state = sub["states"][idx]["cur_state"]
                print(f"  After clear: state={cur_state}")
                if cur_state == 100:
                    raise RuntimeError(
                        f"{arm['name']} still in error state after clear. "
                        f"Check hardware (power/servo/e-stop)."
                    )

        # Re-connect with concise SDK (fresh instance required after release)
        base_robot.release_robot()
        time.sleep(0.3)
        robot = Concise_Marvin_Robot()
        ok = robot.connect(args.robot_ip)
        if not ok:
            raise RuntimeError("failed to connect concise robot after error clear")
        print("✅ Concise robot connected")
        time.sleep(0.3)

        # Initialize kinematics for selected arms
        kine.log_switch(0)
        kin_config = kine.load_config(arm_type=0, config_path=str(args.kine_config))
        for arm in arms:
            at = arm["index"]
            if not kine.initial_kine(
                robot_type=kin_config["TYPE"][at],
                dh=kin_config["DH"][at],
                pnva=kin_config["PNVA"][at],
                j67=kin_config["BD"][at],
            ):
                raise RuntimeError(f"initial_kine failed for {arm['name']}")
        print("✅ Kinematics initialized")

        # ── Read current joints for selected arms ──
        sub = robot.subscribe(dcss)
        current_joints = {}
        first_targets = {}
        for arm in arms:
            sdk_arm = arm["sdk_arm"]
            idx = arm["index"]
            raw = [float(v) for v in sub["outputs"][idx]["fb_joint_pos"]]
            current_joints[sdk_arm] = np.array(raw)
            first_targets[sdk_arm] = arm_trajs[sdk_arm]["targets_deg"][0]
            cur_state = sub["states"][idx]["cur_state"]
            max_jump = np.abs(current_joints[sdk_arm] - first_targets[sdk_arm]).max()
            print(f"{arm['name']}: state={cur_state}, "
                  f"pos={[round(v,1) for v in raw]}, jump={max_jump:.1f}deg")
            if max_jump > 90:
                print(f"  ⚠️  {arm['name']} feedback may be uninitialized (jump={max_jump:.1f}deg). "
                      f"Arm may not be powered on.")

        # ── Phase 0: position mode (read real feedback) ──
        print("\nStep 1: position mode for selected arm(s) ...")
        for arm in arms:
            sdk_arm = arm["sdk_arm"]
            ok = robot.set_position_state(arm=sdk_arm, velRatio=50, AccRatio=50)
            if ok is False:
                print(f"  ⚠️ {sdk_arm} set_position_state failed")
            else:
                print(f"  {sdk_arm} position mode OK")
            time.sleep(0.2)

        time.sleep(0.5)
        sub = robot.subscribe(dcss)
        for arm in arms:
            idx = arm["index"]
            raw = [float(v) for v in sub["outputs"][idx]["fb_joint_pos"]]
            current_joints[arm["sdk_arm"]] = np.array(raw)
            cur_state = sub["states"][idx]["cur_state"]
            print(f"{arm['name']} position mode: state={cur_state}, "
                  f"pos={[round(v,1) for v in raw]}")

        # ── Phase 0b: joint impedance ──
        print("\nStep 2: switching to joint impedance ...")
        joint_k = list(DEFAULT_JOINT_K)
        joint_d = list(DEFAULT_JOINT_D)
        for arm in arms:
            sdk_arm = arm["sdk_arm"]
            ok = robot.set_imp_joint_state(
                arm=sdk_arm, velRatio=100, AccRatio=100,
                K=joint_k, D=joint_d,
            )
            if ok is False:
                print(f"  ⚠️ {sdk_arm} set_imp_joint_state failed; falling back to position mode")
                robot.set_position_state(arm=sdk_arm, velRatio=50, AccRatio=50)
            else:
                print(f"  {sdk_arm} joint impedance OK")
            time.sleep(0.3)

        time.sleep(0.5)
        sub = robot.subscribe(dcss)
        for arm in arms:
            idx = arm["index"]
            raw = [float(v) for v in sub["outputs"][idx]["fb_joint_pos"]]
            current_joints[arm["sdk_arm"]] = np.array(raw)
            cur_state = sub["states"][idx]["cur_state"]
            max_jump = np.abs(current_joints[arm["sdk_arm"]] - first_targets[arm["sdk_arm"]]).max()
            print(f"{arm['name']} impedance: state={cur_state}, "
                  f"pos={[round(v,1) for v in raw]}, jump={max_jump:.1f}deg")

        print("✅ Arm(s) configured")

        # ── Phase 1: entry ramp ──
        entry_hz = 200.0
        entry_dt = 1.0 / entry_hz
        entry_trajs = {}
        for sdk_arm in current_joints:
            entry_trajs[sdk_arm] = build_entry(
                current_joints[sdk_arm], first_targets[sdk_arm],
                duration_s=args.entry_duration_s, control_hz=entry_hz,
            )
        entry_frames = len(list(entry_trajs.values())[0])

        print(f"\n[Phase 1/2] Entry ramp: {args.entry_duration_s}s → {entry_frames} frames")
        print("            (Ctrl+C to stop at any time)")

        for i in range(entry_frames):
            if _interrupted:
                break
            for sdk_arm in entry_trajs:
                robot.set_joint_position_cmd(sdk_arm, entry_trajs[sdk_arm][i].tolist())
            time.sleep(entry_dt)

        if _interrupted:
            print("Stopped during entry phase.")
        else:
            print("✅ Entry complete")

        # ── Phase 2: playback ──
        if not _interrupted:
            print(f"\n[Phase 2/2] Playback: {total_frames} frames "
                  f"at {args.speed_scale}x ({play_duration:.1f}s)")
            print("            (Ctrl+C to stop at any time)")

            playback_t0 = time.perf_counter()
            for fi in range(total_frames):
                if _interrupted:
                    break
                for arm in arms:
                    sdk_arm = arm["sdk_arm"]
                    target = arm_trajs[sdk_arm]["targets_deg"][fi]
                    robot.set_joint_position_cmd(sdk_arm, target.tolist())

                if fi < total_frames - 1:
                    target_time = playback_t0 + (fi + 1) * play_dt
                    sleep_for = target_time - time.perf_counter()
                    if sleep_for > 0:
                        time.sleep(sleep_for)

            if _interrupted:
                print("Stopped during playback phase.")
            else:
                print("✅ Playback complete")

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    finally:
        if _interrupted:
            print("\n⚠️  Playback was interrupted (Ctrl+C).")
        if not args.no_keep_enabled and connected and not _interrupted:
            print("Keeping arm(s) enabled (use --no-keep-enabled to disable).")
        else:
            if connected:
                for arm in arms:
                    try:
                        robot.disable(arm["sdk_arm"])
                    except Exception:
                        pass
                print("Selected arm(s) disabled.")
        try:
            robot.release_robot()
        except Exception:
            pass
        print("Robot released.")

    return 0 if not _interrupted else 130


if __name__ == "__main__":
    raise SystemExit(main())