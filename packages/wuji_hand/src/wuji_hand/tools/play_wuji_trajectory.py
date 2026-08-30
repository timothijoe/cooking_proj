#!/usr/bin/env python3
"""播放 Wuji 右手重定向轨迹到实体左手 (0.2x 默认速度)。

使用方法:
    .venv-wujihand/bin/python scripts/play_wuji_trajectory.py \\
        /tmp/wuji_right_retarget_WuzTwX.npz

    .venv-wujihand/bin/python scripts/play_wuji_trajectory.py \\
        /tmp/wuji_right_retarget_WuzTwX.npz --speed 0.1 --ramp 5.0

安全:
    - Ctrl+C 会立即去使能并退出
    - 轨迹超出硬件限位会自动裁剪并警告
    - 完成后自动去使能
"""

import argparse
import signal
import sys
import time
import numpy as np
from wujihandpy import Hand

DEFAULT_SERIAL = "365939643134"
MARGIN = 0.02  # rad

FINGER_NAMES = ["拇指", "食指", "中指", "无名指", "小指"]


def _write_all_joints(hand, target_5x4):
    """一次性批量发送 20 个关节目标 — 20 个关节同时运动。"""
    hand.write_joint_target_position(target_5x4)


def _read_current(hand):
    """读取当前 20 个关节位置 (flat array)。"""
    return np.asarray(hand.read_joint_actual_position(), dtype=float).reshape(20)


def _show_positions(label, pos):
    """打印五指关节位置。"""
    print(f"  [{label}] ", end="")
    parts = []
    for i, name in enumerate(FINGER_NAMES):
        parts.append(f"{name}: {pos[i]}")
    print(" | ".join(parts))


def _show_positions_compact(label, pos):
    """紧凑打印五指 MCP 位置（最明显的动作）。"""
    mcps = [pos[i, 0] for i in range(5)]  # joint 0 of each finger
    print(f"  [{label}] MCP: "
          + " | ".join(f"{FINGER_NAMES[i]}: {mcps[i]:.3f}" for i in range(5)))


def main():
    parser = argparse.ArgumentParser(description="播放 Wuji 重定向轨迹")
    parser.add_argument("npz_path", help="轨迹 NPZ 文件路径")
    parser.add_argument("--serial-number", default=DEFAULT_SERIAL,
                        help=f"设备序列号 (默认: {DEFAULT_SERIAL})")
    parser.add_argument("--speed", type=float, default=0.2,
                        help="回放倍速 (默认: 0.2, 范围: 0.01-1.0)")
    parser.add_argument("--ramp", type=float, default=3.0,
                        help="缓入时间，秒 (默认: 3.0)")
    parser.add_argument("--hold", type=float, default=2.0,
                        help="结束后保持时间，秒 (默认: 2.0)")
    parser.add_argument("--verbose", action="store_true",
                        help="打印每根手指当前目标位置")
    args = parser.parse_args()

    if not 0 < args.speed <= 1.0:
        print("错误: --speed 必须在 (0, 1.0] 范围内")
        return 1

    # 加载轨迹
    data = np.load(args.npz_path)
    if "joint_positions_rad" in data:
        hand_traj_flat = np.asarray(data["joint_positions_rad"], dtype=float)
    elif "right_joint_positions_rad" in data:
        hand_traj_flat = np.asarray(data["right_joint_positions_rad"], dtype=float)
    elif "left_joint_positions_rad" in data:
        hand_traj_flat = np.asarray(data["left_joint_positions_rad"], dtype=float)
    else:
        print(f"错误: NPZ 中缺少 'joint_positions_rad' 字段，可用 keys: {list(data.keys())}")
        return 1

    if "side" in data:
        side = str(np.asarray(data["side"]).item())
    elif any(k.startswith("left_") for k in data.keys()):
        side = "left"
    else:
        side = "right"
    print(f"轨迹手别: {side}")

    hand_traj = hand_traj_flat.reshape(-1, 5, 4)        # (N, 5, 4)
    total_frames = len(hand_traj)

    # 时间戳
    if "timestamps_ns" in data:
        ts = data["timestamps_ns"].astype(np.int64)
        if total_frames > 1:
            traj_durations = np.diff(ts.astype(np.float64)) / 1e9
            original_dt = traj_durations.mean()
        else:
            original_dt = 0.01
        total_duration = (ts[-1] - ts[0]) / 1e9
    else:
        original_dt = 0.01
        traj_durations = np.full(max(total_frames - 1, 0), original_dt)
        total_duration = (total_frames - 1) * original_dt

    play_dt = original_dt / args.speed
    play_durations = traj_durations / args.speed

    print("=" * 55)
    print("Wuji 轨迹回放")
    print("=" * 55)
    print(f"轨迹文件: {args.npz_path}")
    print(f"总帧数:   {total_frames}")
    print(f"原始帧间隔: {original_dt*1000:.1f} ms (~{1/original_dt:.0f} Hz)")
    print(f"播放帧间隔: {play_dt*1000:.1f} ms ({args.speed}x)")
    print(f"原始时长: {total_duration:.1f}s")
    print(f"预期回放: {total_duration/args.speed:.1f}s")
    print()

    # 连接设备
    print(f"连接 SN={args.serial_number} ...")
    hand = Hand(serial_number=args.serial_number)

    # 读取硬件限位
    lower = np.asarray(hand.read_joint_lower_limit(), dtype=float).reshape(5, 4)
    upper = np.asarray(hand.read_joint_upper_limit(), dtype=float).reshape(5, 4)
    errors = np.asarray(hand.read_joint_error_code())
    temps = np.asarray(hand.read_joint_temperature(), dtype=float)

    if errors.any():
        print(f"❌ 关节错误码非零 (sum={errors.sum()})，请先运行 reset 脚本排查")
        return 1

    # 检查轨迹 vs 硬件限位
    violations = []
    clipped_traj = hand_traj.copy()
    for i in range(5):
        for j in range(4):
            lo, hi = lower[i, j] + MARGIN, upper[i, j] - MARGIN
            col = hand_traj[:, i, j]
            if col.min() < lo:
                violations.append(f"F{i+1}_J{j+1}: min {col.min():.4f} < {lo:.4f} — 已裁剪")
            if col.max() > hi:
                violations.append(f"F{i+1}_J{j+1}: max {col.max():.4f} > {hi:.4f} — 已裁剪")
            clipped_traj[:, i, j] = np.clip(col, lo, hi)

    if violations:
        print("⚠️  轨迹超出硬件限位（已自动裁剪）:")
        for v in violations:
            print(f"   {v}")
    else:
        print("✅ 轨迹在硬件限位内")

    print(f"温度: max={temps.max():.1f}°C")

    current_pos = _read_current(hand).reshape(5, 4)
    first_target = clipped_traj[0]
    max_jump = np.abs(first_target - current_pos).max()
    print(f"当前→首帧最大跳跃: {max_jump:.4f} rad")
    print()

    if args.verbose:
        print("当前手部姿态:")
        _show_positions("当前", current_pos)

    # ── Ctrl+C 安全处理 ──
    interrupted = False

    def on_sigint(signum, frame):
        nonlocal interrupted
        interrupted = True
        print("\n⚠️  收到 Ctrl+C，正在停止...")

    signal.signal(signal.SIGINT, on_sigint)

    # ── 执行 ──
    # Phase 1: 缓入
    print(f"[阶段 1/2] 使能 + {args.ramp}s 缓入到首帧 ...")
    print("          (按 Ctrl+C 可随时停止)")
    hand.write_joint_enabled(True)
    time.sleep(0.1)

    ramp_start = time.perf_counter()
    try:
        while True:
            if interrupted:
                break
            elapsed = time.perf_counter() - ramp_start
            if elapsed >= args.ramp:
                break
            alpha = elapsed / args.ramp
            interp = current_pos + (first_target - current_pos) * alpha
            _write_all_joints(hand, interp)
            time.sleep(0.005)

        if not interrupted:
            _write_all_joints(hand, first_target)
            time.sleep(0.05)

        # Phase 2: 回放
        if not interrupted:
            print(f"[阶段 2/2] 回放 {total_frames} 帧 "
                  f"(预期 {total_duration/args.speed:.1f}s, speed={args.speed}x) ...")

            playback_t0 = time.perf_counter()
            cum_times = np.cumsum(play_durations)  # (N-1,)

            first_traj = clipped_traj[0]
            last_traj = clipped_traj[-1]
            if args.verbose:
                print("  首帧姿态:")
                _show_positions("首帧", first_traj)
                print("  末帧姿态:")
                _show_positions("末帧", last_traj)

            for fi in range(total_frames):
                if interrupted:
                    break
                _write_all_joints(hand, clipped_traj[fi])

                if fi < total_frames - 1:
                    target_time = playback_t0 + cum_times[fi]
                    sleep_for = target_time - time.perf_counter()
                    if sleep_for > 0:
                        time.sleep(sleep_for)

            if not interrupted:
                print(f"[阶段 2/2] 回放完成 ✓")

    finally:
        hand.write_joint_enabled(False)
        print("[安全] 已去使能")

    elapsed_total = time.perf_counter() - ramp_start
    print(f"总耗时: {elapsed_total:.1f}s")

    if interrupted:
        print()
        print("▶ 要复位到张开手，请运行:")
        print("  .venv-wujihand/bin/python scripts/reset_wuji_hand.py")
        return 130

    print(f">>> 保持最终姿态 {args.hold}s ...")
    time.sleep(args.hold)

    final_pos = _read_current(hand).reshape(5, 4)
    max_err = np.abs(final_pos - clipped_traj[-1]).max()
    print(f"最终姿势最大跟踪误差: {max_err*1000:.1f} mrad")
    print("✅ 回放完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
