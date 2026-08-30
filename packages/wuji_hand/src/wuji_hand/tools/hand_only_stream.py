#!/usr/bin/env python3
"""Wuji 手单独回放程序 — 只控制灵巧手，不连接机械臂。

从 200 Hz NPZ 读取 right_hand_target_rad，在真机上慢速回放，
用于验证手部轨迹效果。全程不触碰 Tianji 双臂。

Usage:
    # 干跑（只看规划）
    PYTHONPATH=. python3 real_robot_debug/hand_only_stream.py \
        --source-npz recordings/recorded_hand_guarded_chop_200hz_official_right_retarget_candidate_forward_x_plus_80mm_right_z_plus_130mm.npz

    # 真机回放（默认 0.2x 慢速）
    PYTHONPATH=. python3 real_robot_debug/hand_only_stream.py \
        --source-npz recordings/recorded_hand_guarded_chop_200hz_official_right_retarget_candidate_forward_x_plus_80mm_right_z_plus_130mm.npz \
        --execute

    # 更慢（0.1x）方便观察
    PYTHONPATH=. python3 real_robot_debug/hand_only_stream.py \
        --source-npz recordings/...npz --execute --speed-scale 0.1

Safety:
    - 默认干跑，--execute 才连接
    - 手部目标超出硬件限位自动裁剪
    - Ctrl+C 中断时手停在当前姿态，可用 reset_wuji_hand_fast.py 恢复张开
    - 播放完成后保持使能，方便观察末帧姿态，再用 reset 恢复
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


from robot_core.paths import local_root
ROOT = local_root()
SOURCE = ROOT / "recordings" / "recorded_hand_guarded_chop_200hz_official_right_retarget_candidate.npz"
WUJI_SERIAL = "365939643134"

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
HAND_MARGIN = 0.02  # rad
FINGER_NAMES = ["拇指", "食指", "中指", "无名指", "小指"]


def load_hand_trajectory(source: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load NPZ; return (time_s, hand_rad)."""
    with np.load(source, allow_pickle=False) as data:
        time_s = np.asarray(data["time_s"], dtype=float)
        if "right_hand_target_rad" not in data:
            raise ValueError(f"NPZ 不含 right_hand_target_rad: {source}")
        hand = np.asarray(data["right_hand_target_rad"], dtype=float)
    if time_s.ndim != 1 or time_s.size < 2 or not np.all(np.isfinite(time_s)):
        raise ValueError("time_s must be a finite one-dimensional sequence")
    if hand.shape != (time_s.size, 20) or not np.all(np.isfinite(hand)):
        raise ValueError("right_hand_target_rad must be finite (N,20)")
    intervals = np.diff(time_s)
    if np.any(intervals <= 0) or not np.allclose(intervals, 0.005, atol=1e-9, rtol=0):
        raise ValueError("trajectory must have strictly increasing 5 ms timestamps")
    return time_s - time_s[0], hand


def check_and_clip_hand(hand_rad: np.ndarray) -> np.ndarray:
    """Clip hand trajectory to hardware limits. Returns clipped copy (N,20)."""
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


def build_quintic_entry(start, target, *, duration_s: float, hz: float = 200.0):
    """Quintic rest-to-rest trajectory from start to target (N,20)."""
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-npz", type=Path, default=SOURCE)
    parser.add_argument("--wuji-serial", default=WUJI_SERIAL)
    parser.add_argument("--speed-scale", type=float, default=0.2,
                        help="回放倍速 (0.01-1.0)。默认 0.2x 慢速便于观察。")
    parser.add_argument("--entry-duration-s", type=float, default=5.0,
                        help="缓入到首帧的时长（秒）")
    parser.add_argument("--execute", action="store_true",
                        help="连接 Wuji 手并执行。不指定则干跑。")
    args = parser.parse_args(argv)

    if not args.source_npz.is_file():
        raise ValueError(f"source NPZ not found: {args.source_npz}")
    if not 0 < args.speed_scale <= 1.0:
        raise ValueError("--speed-scale must be in (0, 1.0]")
    if args.entry_duration_s <= 0:
        raise ValueError("--entry-duration-s must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        time_s, hand_rad = load_hand_trajectory(args.source_npz)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    hand_clipped = check_and_clip_hand(hand_rad)
    total_frames = len(time_s)
    play_dt = 0.005 / args.speed_scale
    play_duration = float(time_s[-1]) / args.speed_scale

    print(f"\n总帧数:     {total_frames}")
    print(f"原始 DT:    5 ms (200 Hz)")
    print(f"播放 DT:    {play_dt*1000:.1f} ms (speed={args.speed_scale}x)")
    print(f"原始时长:   {time_s[-1]:.2f}s")
    print(f"回放时长:   {play_duration:.1f}s")
    print("\nWuji 手轨迹范围 (rad，裁剪后):")
    for i in range(5):
        for j in range(4):
            col = hand_clipped[:, i * 4 + j]
            print(f"  {FINGER_NAMES[i]} J{j+1}: [{col.min():.4f}, {col.max():.4f}]")
    print()

    if not args.execute:
        print("DRY_RUN: 规划完成，未连接设备。")
        print(f"  entry: {args.entry_duration_s}s → 手首帧")
        print(f"  playback: {total_frames} 帧 at {args.speed_scale}x")
        print(f"  恢复张开: .venv-wujihand/bin/python scripts/reset_wuji_hand_fast.py")
        return 0

    # ── 真机执行 ──
    from wujihandpy import Hand

    hand = None
    enabled = False
    try:
        print(f"连接 Wuji 手 SN={args.wuji_serial} ...")
        hand = Hand(serial_number=args.wuji_serial)
        errors = np.asarray(hand.read_joint_error_code())
        if errors.any():
            print(f"⚠️ Wuji 手错误码非零 (sum={errors.sum()})。Ctrl+C 中止或继续？")
            # 继续执行，有裁剪保护
        temps = np.asarray(hand.read_joint_temperature(), dtype=float)
        print(f"Wuji 手温度: max={temps.max():.1f}°C")
        print("✅ Wuji 手已连接")

        # 使能
        print("\n>>> Wuji 手使能 ...")
        hand.write_joint_enabled(True)
        enabled = True
        time.sleep(0.1)
        print("✅ 已使能")

        # Phase 1: 缓入到首帧
        hand_current = np.asarray(hand.read_joint_actual_position(), dtype=float).reshape(20)
        first_target = hand_clipped[0]
        max_jump = np.abs(hand_current - first_target).max()
        print(f"\n[Phase 1/2] 缓入 {args.entry_duration_s}s → 手首帧 (最大关节差 {np.rad2deg(max_jump):.1f}°)")

        entry = build_quintic_entry(hand_current, first_target,
                                    duration_s=args.entry_duration_s, hz=200.0)
        for i in range(len(entry)):
            hand.write_joint_target_position(entry[i].reshape(5, 4))
            time.sleep(0.005)
        print("✅ 缓入完成")

        # Phase 2: 回放
        print(f"\n[Phase 2/2] 回放 {total_frames} 帧 at {args.speed_scale}x ({play_duration:.1f}s)")
        print("            Ctrl+C 立即停止 (手停在当前姿态)")

        t0 = time.perf_counter()
        for fi in range(total_frames):
            hand.write_joint_target_position(hand_clipped[fi].reshape(5, 4))
            if fi < total_frames - 1:
                target_time = t0 + (fi + 1) * play_dt
                sleep_for = target_time - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)

        print("\n✅ 回放完成")

        # 保持末帧姿态，方便观察
        final = np.asarray(hand.read_joint_actual_position(), dtype=float).reshape(5, 4)
        print("\n最终关节位置:")
        for i, name in enumerate(FINGER_NAMES):
            print(f"  {name}: {np.round(final[i], 3).tolist()}")
        print("\n" + "=" * 55)
        print("  手停在末帧姿态，保持使能中。")
        print("  观察完毕后，运行以下命令恢复张开：")
        print("    .venv-wujihand/bin/python scripts/reset_wuji_hand_fast.py")
        print("=" * 55)
        return 0

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断。手停在当前姿态，保持使能。")
        print("  恢复张开: .venv-wujihand/bin/python scripts/reset_wuji_hand_fast.py")
        return 130
    except Exception:
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # 保持使能，让用户用 reset 脚本恢复张开，避免从中间姿态断电导致手指卡住
        pass


if __name__ == "__main__":
    raise SystemExit(main())