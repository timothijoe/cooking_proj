#!/usr/bin/env python3
"""将单帧 Wuji 手姿态从 NPZ 文件发送到真实机械手。

使用方法:
    .venv-wujihand/bin/python scripts/send_pose_to_wuji_hand.py \\
        /tmp/wuji_pose_left_20260823_123456.npz

    .venv-wujihand/bin/python scripts/send_pose_to_wuji_hand.py \\
        /tmp/wuji_pose_left.npz --serial-number 365939643134 --ramp 2.0 --hold 3.0

安全:
    - 错误码非零时拒绝执行（先运行 reset 脚本排查）
    - 目标超出硬件限位自动裁剪并警告，不越界发送
    - 使能后缓入（ramp 秒）到目标姿态，保持后自动去使能
    - Ctrl+C 立即去使能并退出
"""

import argparse
import signal
import sys
import time

import numpy as np
from wujihandpy import Hand

DEFAULT_SERIAL = "365939643134"
MARGIN = 0.02  # rad, 限位安全余量

FINGER_NAMES = ["拇指", "食指", "中指", "无名指", "小指"]


def _resolve_position(data: np.lib.npyio.NpzFile) -> np.ndarray:
    """从 NPZ 中提取 (20,) 或 (1, 20) 的单帧姿态。"""
    if "joint_positions_rad" in data:
        pos = np.asarray(data["joint_positions_rad"], dtype=float).reshape(-1, 20)
    elif "right_joint_positions_rad" in data:
        pos = np.asarray(data["right_joint_positions_rad"], dtype=float).reshape(-1, 20)
    elif "left_joint_positions_rad" in data:
        pos = np.asarray(data["left_joint_positions_rad"], dtype=float).reshape(-1, 20)
    else:
        raise ValueError(
            "NPZ 中缺少 'joint_positions_rad' 字段，可用 keys: "
            f"{sorted(data.keys())}"
        )
    if pos.shape[0] != 1:
        raise ValueError(
            f"send_pose 只接受单帧姿态，当前有 {pos.shape[0]} 帧；"
            "多帧请用 scripts/play_wuji_trajectory.py"
        )
    return pos[0]  # (20,)


def _resolve_side(data: np.lib.npyio.NpzFile) -> str | None:
    if "side" in data:
        side = str(np.asarray(data["side"]).item())
        return side if side in ("left", "right") else None
    for key in data.keys():
        if key.startswith("left_"):
            return "left"
        if key.startswith("right_"):
            return "right"
    return None


def _show_positions(label, pos):
    """打印五指关节位置。"""
    print(f"  [{label}] ", end="")
    parts = []
    for i, name in enumerate(FINGER_NAMES):
        parts.append(f"{name}: {pos[i]}")
    print(" | ".join(parts))


def main():
    parser = argparse.ArgumentParser(description="将单帧 Wuji 手姿态发送到真机")
    parser.add_argument("npz_path", help="NPZ 文件路径（含 joint_positions_rad 字段）")
    parser.add_argument("--serial-number", default=DEFAULT_SERIAL,
                        help=f"设备序列号 (默认: {DEFAULT_SERIAL})")
    parser.add_argument("--ramp", type=float, default=2.0,
                        help="缓入时间，秒 (默认: 2.0)")
    parser.add_argument("--hold", type=float, default=3.0,
                        help="到达后保持时间，秒 (默认: 3.0)，之后自动去使能")
    parser.add_argument("--verbose", action="store_true",
                        help="打印每根手指当前/目标位置")
    args = parser.parse_args()

    # ── 加载轨迹 ──
    data = np.load(args.npz_path)
    target_flat = _resolve_position(data)
    side = _resolve_side(data)
    print("=" * 55)
    print("Wuji 单帧姿态 → 真机")
    print("=" * 55)
    print(f"NPZ:   {args.npz_path}")
    print(f"Side:  {side or '未知'}")
    print()

    # ── 连接设备 ──
    print(f"连接 SN={args.serial_number} ...")
    hand = Hand(serial_number=args.serial_number)

    # 读取硬件限位
    lower = np.asarray(hand.read_joint_lower_limit(), dtype=float).reshape(5, 4)
    upper = np.asarray(hand.read_joint_upper_limit(), dtype=float).reshape(5, 4)
    errors = np.asarray(hand.read_joint_error_code())
    temps = np.asarray(hand.read_joint_temperature(), dtype=float)

    # ── 安全检查 ──
    if errors.any():
        print(f"❌ 关节错误码非零 (sum={errors.sum()})，请先运行 reset 脚本排查")
        return 1

    # 裁剪目标到硬件限位内（留安全余量）
    target_5x4 = target_flat.reshape(5, 4)
    lo = lower + MARGIN
    hi = upper - MARGIN
    clipped = np.clip(target_5x4, lo, hi)
    violations = (target_5x4 < lo) | (target_5x4 > hi)
    if violations.any():
        for i in range(5):
            for j in range(4):
                if violations[i, j]:
                    print(f"⚠️  F{i+1}_J{j+1}: {target_5x4[i,j]:.4f} → 裁剪为 "
                          f"{clipped[i,j]:.4f} (限位 [{lower[i,j]:.4f}, {upper[i,j]:.4f}])")
        print("⚠️  目标超出硬件限位（已自动裁剪）")
    else:
        print("✅ 目标在硬件限位内")

    print(f"温度: max={temps.max():.1f}°C")
    print(f"固件: {hand.get_firmware_version()}")

    current_pos = np.asarray(hand.read_joint_actual_position(), dtype=float).reshape(5, 4)
    max_jump = np.abs(clipped - current_pos).max()
    print(f"当前→目标最大跳跃: {max_jump:.4f} rad")
    print()

    if args.verbose:
        print("当前手部姿态:")
        _show_positions("当前", current_pos)
        print("目标姿态:")
        _show_positions("目标", clipped)

    # ── Ctrl+C 安全处理 ──
    interrupted = False

    def on_sigint(signum, frame):
        nonlocal interrupted
        interrupted = True
        print("\n⚠️  收到 Ctrl+C，正在停止...")

    signal.signal(signal.SIGINT, on_sigint)

    # ── 执行 ──
    print(f">>> 使能 + {args.ramp}s 缓入到目标姿态 ...")
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
            interp = current_pos + (clipped - current_pos) * alpha
            hand.write_joint_target_position(interp)
            time.sleep(0.005)

        if not interrupted:
            hand.write_joint_target_position(clipped)
            time.sleep(0.05)

            print(f">>> 保持最终姿态 {args.hold}s ...")
            hold_deadline = time.perf_counter() + args.hold
            while time.perf_counter() < hold_deadline:
                if interrupted:
                    break
                time.sleep(0.05)

    finally:
        hand.write_joint_enabled(False)
        print("[安全] 已去使能")

    elapsed_total = time.perf_counter() - ramp_start
    print(f"总耗时: {elapsed_total:.1f}s")

    if interrupted:
        print("▶ 要复位到张开手，请运行:")
        print("  .venv-wujihand/bin/python scripts/reset_wuji_hand.py")
        return 130

    final_pos = np.asarray(hand.read_joint_actual_position(), dtype=float).reshape(5, 4)
    max_err = np.abs(final_pos - clipped).max()
    print(f"最终姿势最大跟踪误差: {max_err*1000:.1f} mrad")
    print("✅ 姿态已发送到真机")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())