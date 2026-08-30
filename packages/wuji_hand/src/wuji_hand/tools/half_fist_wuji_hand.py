#!/usr/bin/env python3
"""Wuji 左手半握拳脚本 — 从当前姿态缓入到半握拳，保持后自动恢复张开。

半握拳定义：五指自然弯曲，约为全握拳的 50%。适合测试手指协调性、
观察手指间碰撞风险，以及验证关节跟踪精度。

安全：
    - 目标 clamped 在硬件限位内（0.02 rad 余量）；
    - 错误码非零时拒绝执行；
    - 保持阶段按 Ctrl+C → 强制回零再去使能；
    - 保持结束后自动平滑张开。
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
    parser = argparse.ArgumentParser(description="Wuji 左手半握拳")
    parser.add_argument("--serial-number", default=DEFAULT_SERIAL,
                        help=f"设备序列号 (默认: {DEFAULT_SERIAL})")
    parser.add_argument("--ramp", type=float, default=3.0,
                        help="缓入到半握拳的时间，秒 (默认: 3.0)")
    parser.add_argument("--hold", type=float, default=5.0,
                        help="半握拳保持时间，秒 (默认: 5.0)")
    parser.add_argument("--recovery-ramp", type=float, default=1.2,
                        help="恢复张开的缓出时间，秒 (默认: 1.2)")
    args = parser.parse_args()

    res = run(serial_number=args.serial_number, ramp=args.ramp,
              hold=args.hold, recovery_ramp=args.recovery_ramp)
    raise SystemExit(res)


def run(serial_number=DEFAULT_SERIAL, ramp=3.0, hold=5.0, recovery_ramp=1.2):
    """执行半握拳，返回进程退出码。可被其他脚本 import 复用。"""
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
    print(f"缓入: {ramp}s, 保持: {hold}s, 恢复: {recovery_ramp}s")

    # 缓入到半握拳
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

        # 保持阶段，定期打印状态
        print(f"\n>>> 半握拳保持中 ({hold}s)，按 Ctrl+C 可快速恢复张开 ...")
        report_interval = 1.0
        elapsed = 0.0
        while elapsed < hold:
            pos = np.asarray(hand.read_joint_actual_position(), dtype=float).reshape(5, 4)
            print(f"  [t={elapsed + 1:.0f}s] 拇指={np.round(pos[0], 3).tolist()}  "
                  f"食指={np.round(pos[1], 3).tolist()}")
            time.sleep(report_interval)
            elapsed += report_interval

        # 自动恢复张开
        print(f"\n>>> 保持结束，{recovery_ramp}s 恢复张开 ...")
        final_current = np.asarray(hand.read_joint_actual_position(), dtype=float).reshape(5, 4)
        zero_target = np.clip(np.zeros((5, 4)), lower + 0.02, upper - 0.02)
        rsteps = max(int(recovery_ramp * 50), 20)
        rdt = recovery_ramp / rsteps
        for step in range(1, rsteps + 1):
            alpha = step / rsteps
            interp = final_current + (zero_target - final_current) * alpha
            hand.write_joint_target_position(interp)
            time.sleep(rdt)

        final = np.asarray(hand.read_joint_actual_position(), dtype=float).reshape(5, 4)
        max_err = np.abs(final - zero_target).max()
        print(f"最大跟踪误差: {max_err * 1000:.1f} mrad")
        print("✅ 半握拳测试完成，已自动张开")
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
        except Exception:
            pass
        return 1
    finally:
        try:
            hand.write_joint_enabled(False)
        except Exception:
            pass
        print(">>> 已去使能")


if __name__ == "__main__":
    raise SystemExit(main())