#!/usr/bin/env python3
"""Wuji 左手半握拳脚本 — 从当前姿态缓入到半握拳，保持使能后退出。

与 half_fist_wuji_hand.py 的区别：
    - 保持结束后不自动恢复张开，手停在半握拳状态；
    - 保持使能，程序退出后运行 reset_wuji_hand_fast.py 恢复张开；
    - 适合需要长时间保持半握拳姿势观察或测试的场景。

安全：
    - 目标 clamped 在硬件限位内（0.02 rad 余量）；
    - 错误码非零时拒绝执行；
    - 保持阶段按 Ctrl+C → 强制回零再去使能；
    - 程序退出后手保持使能，USB 连接释放，可被 reset 脚本接管。
"""

import argparse
import time

import numpy as np
from wujihandpy import Hand

DEFAULT_SERIAL = "365939643134"
FINGER_NAMES = ["拇指", "食指", "中指", "无名指", "小指"]

# 半握拳目标 (rad)：五指自然弯曲，约全握拳的 50%
HALF_FIST_TARGET = np.array([
    [0.45, 0.30, 0.50, 0.45],   # 拇指
    [0.60, 0.13, 0.60, 0.55],   # 食指
    [0.60, 0.12, 0.60, 0.55],   # 中指
    [0.60, 0.11, 0.60, 0.50],   # 无名指
    [0.55, 0.11, 0.55, 0.50],   # 小指
], dtype=float)


def main():
    parser = argparse.ArgumentParser(description="Wuji 左手半握拳（保持使能，不自动恢复）")
    parser.add_argument("--serial-number", default=DEFAULT_SERIAL,
                        help=f"设备序列号 (默认: {DEFAULT_SERIAL})")
    parser.add_argument("--ramp", type=float, default=3.0,
                        help="缓入到半握拳的时间，秒 (默认: 3.0)")
    parser.add_argument("--hold", type=float, default=0,
                        help="半握拳保持时间后再退出，秒 (默认: 0，即到达后立即退出)")
    args = parser.parse_args()

    res = run(serial_number=args.serial_number, ramp=args.ramp, hold=args.hold)
    raise SystemExit(res)


def run(serial_number=DEFAULT_SERIAL, ramp=3.0, hold=0):
    """执行半握拳并保持使能，返回进程退出码。可被其他脚本 import 复用。"""
    print(f"连接设备 SN={serial_number} ...")
    hand = Hand(serial_number=serial_number)

    current = np.asarray(hand.read_joint_actual_position(), dtype=float).reshape(5, 4)
    lower = np.asarray(hand.read_joint_lower_limit(), dtype=float).reshape(5, 4)
    upper = np.asarray(hand.read_joint_upper_limit(), dtype=float).reshape(5, 4)
    errors = np.asarray(hand.read_joint_error_code())

    if errors.any():
        print(f"错误: 关节错误码非零 (sum={errors.sum()})，拒绝执行")
        return 1

    target = np.clip(HALF_FIST_TARGET, lower + 0.02, upper - 0.02)

    print(f"固件: {hand.get_firmware_version()}")
    print(f"温度: max={np.asarray(hand.read_joint_temperature(), dtype=float).max():.1f}°C")
    print("当前关节位置:")
    for i, name in enumerate(FINGER_NAMES):
        print(f"  {name}: {np.round(current[i], 3).tolist()}")
    print("半握拳目标:")
    for i, name in enumerate(FINGER_NAMES):
        print(f"  {name}: {np.round(target[i], 3).tolist()}")
    print(f"缓入: {ramp}s, 保持: {hold}s")

    steps = max(int(ramp * 50), 20)
    dt = ramp / steps
    print(f"\n>>> 使能，{ramp}s 缓入半握拳 ...")
    hand.write_joint_enabled(True)
    time.sleep(0.1)

    try:
        for step in range(1, steps + 1):
            alpha = step / steps
            interp = current + (target - current) * alpha
            hand.write_joint_target_position(interp)
            time.sleep(dt)

        if hold > 0:
            print(f"\n>>> 半握拳保持中 ({hold}s)，按 Ctrl+C 可中断 ...")
            time.sleep(hold)

        final = np.asarray(hand.read_joint_actual_position(), dtype=float).reshape(5, 4)
        max_err = np.abs(final - target).max()
        print(f"\n✅ 半握拳到位，最大跟踪误差: {max_err * 1000:.1f} mrad")
        print("=" * 55)
        print("  手已停在半握拳状态，保持使能中。")
        print("  运行以下命令恢复张开：")
        print("    .venv-wujihand/bin/python scripts/reset_wuji_hand_fast.py")
        print("=" * 55)
        return 0

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断，强制恢复张开 ...")
        try:
            cur = np.asarray(hand.read_joint_actual_position(), dtype=float).reshape(5, 4)
            zero = np.clip(np.zeros((5, 4)), lower + 0.02, upper - 0.02)
            for step in range(1, 31):
                a = step / 30
                hand.write_joint_target_position(cur + (zero - cur) * a)
                time.sleep(0.02)
            hand.write_joint_enabled(False)
        except Exception:
            pass
        return 1
    finally:
        # 不自动去使能，保持手在半握拳状态
        pass


if __name__ == "__main__":
    raise SystemExit(main())