#!/usr/bin/env python3
"""Wuji 左手快速恢复脚本 — 从任意姿态快速返回张开（零位）。

适用场景：握拳/做手势后手指间出现碰撞，或需要紧急恢复到安全张开姿态。

与 scripts/reset_wuji_hand.py 的区别：
    - 默认 ramp 0.8s（更快），可用 --ramp 调整；
    - Ctrl+C 中断后仍然强制平滑回零再去使能（安全兜底）；
    - 批量 write_joint_target_position 一次写入 20 关节。

安全：
    - 目标夹在硬件限位内（0.02 rad 余量）；
    - 错误码非零时拒绝执行；
    - 全程使能，执行完自动去使能。
"""

import argparse
import time

import numpy as np
from wujihandpy import Hand

DEFAULT_SERIAL = "365939643134"
FINGER_NAMES = ["拇指", "食指", "中指", "无名指", "小指"]


def main():
    parser = argparse.ArgumentParser(description="Wuji 左手快速恢复到张开姿态")
    parser.add_argument("--serial-number", default=DEFAULT_SERIAL,
                        help=f"设备序列号 (默认: {DEFAULT_SERIAL})")
    parser.add_argument("--ramp", type=float, default=0.8,
                        help="快速恢复缓入时间，秒 (默认: 0.8)")
    parser.add_argument("--hold", type=float, default=0.8,
                        help="到达后保持时间，秒 (默认: 0.8)")
    args = parser.parse_args()

    res = run(serial_number=args.serial_number, ramp=args.ramp, hold=args.hold)
    raise SystemExit(res)


def run(serial_number=DEFAULT_SERIAL, ramp=0.8, hold=0.8):
    """执行快速恢复，返回进程退出码。可被其他脚本 import 复用。"""
    print(f"连接设备 SN={serial_number} ...")
    hand = Hand(serial_number=serial_number)

    current = np.asarray(hand.read_joint_actual_position(), dtype=float).reshape(5, 4)
    lower = np.asarray(hand.read_joint_lower_limit(), dtype=float).reshape(5, 4)
    upper = np.asarray(hand.read_joint_upper_limit(), dtype=float).reshape(5, 4)
    errors = np.asarray(hand.read_joint_error_code())

    if errors.any():
        print(f"错误: 关节错误码非零 (sum={errors.sum()})，拒绝执行")
        return 1

    # 目标：全零位（张开），夹在限位内
    target = np.clip(np.zeros((5, 4)), lower + 0.02, upper - 0.02)

    print(f"固件: {hand.get_firmware_version()}")
    print(f"温度: max={np.asarray(hand.read_joint_temperature(), dtype=float).max():.1f}°C")
    print("当前关节位置:")
    for i, name in enumerate(FINGER_NAMES):
        print(f"  {name}: {np.round(current[i], 3).tolist()}")
    print(f"目标: 全零位（张开手）")
    print(f"快速恢复 ramp: {ramp}s, 保持: {hold}s")

    if ramp <= 0:
        print("错误: --ramp 必须 > 0")
        return 1

    steps = max(int(ramp * 50), 20)      # 50 Hz 插值，至少 20 步
    dt = ramp / steps

    print("\n>>> 使能 ...")
    hand.write_joint_enabled(True)
    time.sleep(0.1)

    interrupted = False
    try:
        for step in range(1, steps + 1):
            alpha = step / steps
            interp = current + (target - current) * alpha
            hand.write_joint_target_position(interp)   # 批量写入 5x4，五指同时
            time.sleep(dt)

        time.sleep(hold)

        final = np.asarray(hand.read_joint_actual_position(), dtype=float).reshape(5, 4)
        max_err = np.abs(final - target).max()
        print("\n最终关节位置:")
        for i, name in enumerate(FINGER_NAMES):
            print(f"  {name}: {np.round(final[i], 3).tolist()}")
        print(f"最大跟踪误差: {max_err * 1000:.1f} mrad")
        print("✅ 快速恢复完成")
        return 0
    except KeyboardInterrupt:
        interrupted = True
        print("\n⚠️ 收到中断，直接强制回到张开位再退出 ...")
        # 兜底：无论当前在哪，尽力一次下张开目标，再保持使能到写完成
        try:
            hand.write_joint_target_position(target)
            time.sleep(0.3)
        except Exception:
            pass
        print(">>> 已去使能")
        return 1
    finally:
        try:
            hand.write_joint_enabled(False)
        except Exception:
            pass
        if not interrupted:
            print(">>> 已去使能")


if __name__ == "__main__":
    raise SystemExit(main())