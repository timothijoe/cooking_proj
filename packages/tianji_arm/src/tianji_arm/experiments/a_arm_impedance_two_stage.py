#!/usr/bin/env python3
"""Real dual-arm joint-impedance playback from an offline NPZ trajectory.

Loads left_arm_target_rad AND right_arm_target_rad from a 200 Hz NPZ,
connects to the real robot's A arm (left) and B arm (right), initializes
joint impedance mode on both arms using Concise_Marvin_Robot SDK, and
plays back the trajectory at a configurable speed scale with both arms
moving simultaneously.

USAGE:
    PYTHONPATH=. python3 real_robot_debug/a_arm_impedance_two_stage.py         # dry-run
    PYTHONPATH=. python3 real_robot_debug/a_arm_impedance_two_stage.py --execute   # real robot

Safety:
    - Default is dry-run (no robot commands).
    - Use low --speed-scale (e.g. 0.05) for first real test.
    - Ctrl+C stops both arms and disables them.
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

# Default joint impedance parameters (conservative, same for both arms)
DEFAULT_JOINT_K = (8.0, 8.0, 8.0, 4.0, 2.0, 1.5, 1.0)
DEFAULT_JOINT_D = (0.8, 0.8, 0.8, 0.6, 0.4, 0.3, 0.2)

# Joint limits for both arms (degrees)
JOINT_LIMITS_DEG = [
    (-170, 170), (-100, 120), (-170, 170), (-130, 130),
    (-170, 170), (-90, 220), (-170, 170),
]

ARMS = [
    {"sdk_arm": "A", "index": 0, "npz_key": "left_arm_target_rad",  "name": "Left  (A)"},
    {"sdk_arm": "B", "index": 1, "npz_key": "right_arm_target_rad", "name": "Right (B)"},
]

# ── Signal handling ──
_interrupted = False


def _on_sigint(signum, frame):
    global _interrupted
    _interrupted = True
    print("\n⚠️  Ctrl+C — stopping both arms ...")


signal.signal(signal.SIGINT, _on_sigint)


# ── NPZ loading ──


def load_dual_arm_trajectory(path: Path) -> dict:
    """Load the NPZ and return both arms' data in degrees at 200 Hz."""
    with np.load(path, allow_pickle=False) as data:
        time_s = np.asarray(data["time_s"], dtype=float)
        result = {"time_s": time_s - time_s[0], "arms": {}}
        for arm in ARMS:
            targets_rad = np.asarray(data[arm["npz_key"]], dtype=float)
            if targets_rad.shape != (time_s.size, 7) or not np.all(np.isfinite(targets_rad)):
                raise ValueError(f"{arm['npz_key']} must be finite (N,7) matching time_s")
            result["arms"][arm["sdk_arm"]] = {
                "targets_deg": np.rad2deg(targets_rad),
            }
    if time_s.ndim != 1 or time_s.size < 2 or not np.all(np.isfinite(time_s)):
        raise ValueError("time_s must be finite 1-D array with >=2 samples")
    intervals = np.diff(time_s)
    if np.any(intervals <= 0):
        raise ValueError("time_s must be strictly increasing")
    if not np.allclose(intervals, intervals[0], rtol=0, atol=1e-9):
        raise ValueError("time_s must have uniform 5 ms spacing (200 Hz)")
    result["total_frames"] = time_s.size
    result["total_duration_s"] = float(time_s[-1] - time_s[0])
    return result


def check_all_joint_limits(traj: dict):
    """Verify all trajectory joints for both arms are within hardware limits."""
    for arm in ARMS:
        targets_deg = traj["arms"][arm["sdk_arm"]]["targets_deg"]
        for j in range(7):
            lo, hi = JOINT_LIMITS_DEG[j]
            col = targets_deg[:, j]
            if col.min() < lo + 1.0 or col.max() > hi - 1.0:
                raise ValueError(
                    f"{arm['name']} Joint {j+1}: traj [{col.min():.1f}, {col.max():.1f}] deg "
                    f"violates limit [{lo}, {hi}] (1 deg margin)"
                )


def build_entry(start_deg, first_target_deg, duration_s=10.0, control_hz=200.0):
    """Build a quintic (smooth) entry trajectory."""
    steps = int(round(duration_s * control_hz))
    phase = np.linspace(0, 1, steps + 1)
    smooth = 10 * phase**3 - 15 * phase**4 + 6 * phase**5
    result = start_deg + smooth[:, None] * (first_target_deg - start_deg)
    result[0] = start_deg
    result[-1] = first_target_deg
    return result


# ── Main ──


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-npz", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--robot-ip", default=ROBOT_IP)
    parser.add_argument("--kine-config", type=Path, default=DEFAULT_KINE_CONFIG)
    parser.add_argument("--speed-scale", type=float, default=0.05,
                        help="Playback speed (0.01-1.0).")
    parser.add_argument("--entry-duration-s", type=float, default=15.0,
                        help="Smooth ramp time from current to first frame.")
    parser.add_argument("--execute", action="store_true",
                        help="Send commands to the real robot.")
    parser.add_argument("--no-keep-enabled", action="store_true",
                        help="Disable both arms after playback.")
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
    except ValueError as e:
        print(f"error: {e}")
        return 2

    # ── Load trajectory ──
    print(f"Loading: {args.source_npz}")
    traj = load_dual_arm_trajectory(args.source_npz)
    total_frames = traj["total_frames"]
    source_dt = 0.005
    play_dt = source_dt / args.speed_scale
    play_duration = traj["total_duration_s"] / args.speed_scale

    print(f"Total frames:      {total_frames}")
    print(f"Source DT:         5 ms (200 Hz)")
    print(f"Play DT:           {play_dt*1000:.1f} ms (speed={args.speed_scale}x)")
    print(f"Source duration:   {traj['total_duration_s']:.1f}s")
    print(f"Playback duration: {play_duration:.1f}s")

    check_all_joint_limits(traj)
    print("✅ Both arms — joint limits OK (with 1 deg margin)")

    print()
    for arm in ARMS:
        targets_deg = traj["arms"][arm["sdk_arm"]]["targets_deg"]
        max_step = np.abs(np.diff(targets_deg, axis=0)).max()
        print(f"{arm['name']}:")
        for j in range(7):
            print(f"  Joint {j+1}: [{targets_deg[:,j].min():7.1f}, {targets_deg[:,j].max():7.1f}]")
        print(f"  Max frame step: {max_step:.2f} deg")
        print()

    if not args.execute:
        print(f"[Dry-run] Entry: {args.entry_duration_s}s quintic ramp → first frame")
        print(f"[Dry-run] Playback: {total_frames} frames at {args.speed_scale}x")
        print("[Dry-run] Both arms planned. No robot commands sent.")
        print("\nTo execute on the real robot, add --execute")
        return 0

    # ═════════════════════════════════════════════════════════
    # ── Real robot execution ──
    # ═════════════════════════════════════════════════════════

    from SDK_PYTHON.fx_kine import Marvin_Kine
    from SDK_PYTHON.fx_robot import DCSS, Marvin_Robot, Concise_Marvin_Robot

    dcss = DCSS()
    # Use Marvin_Robot for connect + error clearing (Concise version rejects on error),
    # then use concise API for impedance config + playback.
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

        # Clear errors on both arms
        base_robot.check_error_and_clear(dcss)
        base_robot.log_switch("1")
        base_robot.local_log_switch("1")
        time.sleep(0.3)

        # Verify both arms are error-free and in a valid state
        sub = base_robot.subscribe(dcss)
        for arm in ARMS:
            idx = arm["index"]
            cur_state = sub["states"][idx]["cur_state"]
            err_code = sub["states"][idx]["err_code"]
            raw = [float(v) for v in sub["outputs"][idx]["fb_joint_pos"]]
            print(f"{arm['name']}: state={cur_state}, err_code={err_code}, "
                  f"pos={[round(v,1) for v in raw]}")

            if cur_state == 100:
                # Arm in error — try clear
                print(f"  ⚠️  {arm['name']} in error state (100), attempting clear ...")
                base_robot.clear_set()
                base_robot.clear_error(arm["sdk_arm"])
                base_robot.send_cmd()
                time.sleep(0.5)
                sub = base_robot.subscribe(dcss)
                cur_state = sub["states"][idx]["cur_state"]
                err_code = sub["states"][idx]["err_code"]
                print(f"  After clear: state={cur_state}, err_code={err_code}")
                if cur_state == 100:
                    raise RuntimeError(
                        f"{arm['name']} still in error state after clear. "
                        f"Please check hardware (power, servo, emergency stop)."
                    )

        # Now that errors are cleared, we can use the concise robot too
        # by re-connecting with concise (releases and reconnects):
        base_robot.release_robot()
        time.sleep(0.3)
        robot = Concise_Marvin_Robot()
        ok = robot.connect(args.robot_ip)
        if not ok:
            raise RuntimeError("failed to connect concise robot after error clear")
        print("✅ Concise robot connected")
        time.sleep(0.3)

        # ── Init kinematics for both arms (only for feedback FK, not IK) ──
        kine.log_switch(0)
        kin_config = kine.load_config(arm_type=0, config_path=str(args.kine_config))
        for arm_type in [0, 1]:
            if not kine.initial_kine(
                robot_type=kin_config["TYPE"][arm_type],
                dh=kin_config["DH"][arm_type],
                pnva=kin_config["PNVA"][arm_type],
                j67=kin_config["BD"][arm_type],
            ):
                raise RuntimeError(f"initial_kine failed for arm type {arm_type}")
        print("✅ Kinematics initialized (both arms)")

        # ── Verify feedback ──
        sub = robot.subscribe(dcss)
        for _ in range(5):
            sub = robot.subscribe(dcss)
            time.sleep(0.01)
        print("✅ Feedback updating")

        # ── Read current joint positions ──
        current_joints = {}
        first_targets = {}
        for arm in ARMS:
            sdk_arm = arm["sdk_arm"]
            idx = arm["index"]
            raw = [float(v) for v in sub["outputs"][idx]["fb_joint_pos"]]
            current_joints[sdk_arm] = np.array(raw)
            first_targets[sdk_arm] = traj["arms"][sdk_arm]["targets_deg"][0]

            state_info = sub.get("states", [{}])[idx]
            cur_state = state_info.get("cur_state", "?")
            err_code = state_info.get("err_code", "?")

            max_jump = np.abs(current_joints[sdk_arm] - first_targets[sdk_arm]).max()
            print(f"{arm['name']}: current={[round(v,1) for v in raw]}")
            print(f"  State: cur_state={cur_state}, err_code={err_code}")
            print(f"  Max jump current→first frame: {max_jump:.1f} deg")

            if max_jump > 90:
                print(f"  ⚠️  WARNING: feedback looks uninitialized. "
                      f"Arm may not be powered on or enabled.")

        # ── Configure BOTH arms ──
        # Strategy: first put both into position mode (state=1) to read real
        # feedback, then switch to joint impedance (state=3).
        print("\nStep 1: position mode for both arms (read real feedback) ...")
        for arm in ARMS:
            sdk_arm = arm["sdk_arm"]
            ok = robot.set_position_state(arm=sdk_arm, velRatio=50, AccRatio=50)
            if ok is False:
                raise RuntimeError(f"set_position_state failed for arm {sdk_arm}")
            time.sleep(0.2)
        print("✅ Position mode configured")

        # Re-read feedback in position mode
        time.sleep(0.5)
        sub = robot.subscribe(dcss)
        for arm in ARMS:
            sdk_arm = arm["sdk_arm"]
            idx = arm["index"]
            raw = [float(v) for v in sub["outputs"][idx]["fb_joint_pos"]]
            current_joints[sdk_arm] = np.array(raw)
            first_targets[sdk_arm] = traj["arms"][sdk_arm]["targets_deg"][0]
            cur_state = sub["states"][idx]["cur_state"]
            max_jump = np.abs(current_joints[sdk_arm] - first_targets[sdk_arm]).max()
            print(f"{arm['name']} position mode: "
                  f"state={cur_state}, pos={[round(v,1) for v in raw]}, "
                  f"jump={max_jump:.1f}deg")

        # Now switch to joint impedance
        print("\nStep 2: switching to joint impedance mode (both arms) ...")
        joint_k = list(DEFAULT_JOINT_K)
        joint_d = list(DEFAULT_JOINT_D)

        for arm in ARMS:
            sdk_arm = arm["sdk_arm"]
            print(f"  Arm {sdk_arm} ...")
            ok = robot.set_imp_joint_state(
                arm=sdk_arm, velRatio=100, AccRatio=100, K=joint_k, D=joint_d,
            )
            if ok is False:
                raise RuntimeError(f"set_imp_joint_state failed for arm {sdk_arm}")
            time.sleep(0.3)

        print(f"✅ Joint impedance configured (both arms)")
        print(f"   K: {joint_k}")
        print(f"   D: {joint_d}")

        # Final feedback read in impedance mode
        time.sleep(0.5)
        sub = robot.subscribe(dcss)
        for arm in ARMS:
            sdk_arm = arm["sdk_arm"]
            idx = arm["index"]
            raw = [float(v) for v in sub["outputs"][idx]["fb_joint_pos"]]
            current_joints[sdk_arm] = np.array(raw)
            cur_state = sub["states"][idx]["cur_state"]
            max_jump = np.abs(current_joints[sdk_arm] - first_targets[sdk_arm]).max()
            print(f"{arm['name']} impedance mode: "
                  f"state={cur_state}, pos={[round(v,1) for v in raw]}, "
                  f"jump={max_jump:.1f}deg")
            if cur_state == 100:
                print(f"  ⚠️  {arm['name']} still in error. "
                      f"Falling back to position mode for this arm.")
                robot.set_position_state(arm=sdk_arm, velRatio=50, AccRatio=50)
                time.sleep(0.3)

        # ═══ Phase 1: Entry ramp ═══
        entry_duration = args.entry_duration_s
        entry_hz = 200.0
        entry_dt = 1.0 / entry_hz

        entry_trajs = {}
        for sdk_arm in current_joints:
            entry_trajs[sdk_arm] = build_entry(
                current_joints[sdk_arm],
                first_targets[sdk_arm],
                duration_s=entry_duration,
                control_hz=entry_hz,
            )
        entry_frames = len(list(entry_trajs.values())[0])

        print(f"\n[Phase 1/2] Entry ramp: {entry_duration}s → {entry_frames} frames (both arms)")
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
            print("✅ Entry complete (both arms)")

        # ═══ Phase 2: Playback ═══
        if not _interrupted:
            print(f"\n[Phase 2/2] Playback: {total_frames} frames "
                  f"at {args.speed_scale}x ({play_duration:.1f}s) — both arms")
            print("            (Ctrl+C to stop at any time)")

            playback_t0 = time.perf_counter()
            for fi in range(total_frames):
                if _interrupted:
                    break
                for arm in ARMS:
                    sdk_arm = arm["sdk_arm"]
                    target = traj["arms"][sdk_arm]["targets_deg"][fi]
                    robot.set_joint_position_cmd(sdk_arm, target.tolist())

                if fi < total_frames - 1:
                    target_time = playback_t0 + (fi + 1) * play_dt
                    sleep_for = target_time - time.perf_counter()
                    if sleep_for > 0:
                        time.sleep(sleep_for)

            if _interrupted:
                print("Stopped during playback phase.")
            else:
                print("✅ Playback complete (both arms)")

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    finally:
        if _interrupted:
            print("\n⚠️  Playback was interrupted (Ctrl+C).")
        if not args.no_keep_enabled and connected and not _interrupted:
            print("Keeping both arms enabled (use --no-keep-enabled to disable).")
        else:
            if connected:
                for arm in ARMS:
                    try:
                        robot.disable(arm["sdk_arm"])
                    except Exception:
                        pass
                print("Both arms disabled.")
        try:
            robot.release_robot()
        except Exception:
            pass
        print("Robot released.")

    return 0 if not _interrupted else 130


if __name__ == "__main__":
    raise SystemExit(main())
