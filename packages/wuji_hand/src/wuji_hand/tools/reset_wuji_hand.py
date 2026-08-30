#!/usr/bin/env python3
"""Wuji 左手复位脚本 — 将所有关节缓入到张开（零位）姿态。

使用方法：
    .venv-wujihand/bin/python scripts/reset_wuji_hand.py

安全：
    - 默认缓入 1 秒，可通过 --ramp 调整
    - 不会跳过硬件限位
    - 完成后自动去使能
"""

import argparse
import time
import numpy as np
from wujihandpy import Hand

DEFAULT_SERIAL = "365939643134"


def main():
    parser = argparse.ArgumentParser(description="Wuji 左手复位到张开姿态")
    parser.add_argument("--serial-number", default=DEFAULT_SERIAL,
                        help=f"设备序列号 (默认: {DEFAULT_SERIAL})")
    parser.add_argument("--ramp", type=float, default=1.0,
                        help="缓入时间，秒 (默认: 1.0)")
    parser.add_argument("--hold", type=float, default=1.0,
                        help="到达后保持时间，秒 (默认: 1.0)")
    args = parser.parse_args()

    print(f"连接设备 SN={args.serial_number} ...")
    hand = Hand(serial_number=args.serial_number)

    # 读取当前状态
    current = np.asarray(hand.read_joint_actual_position(), dtype=float).reshape(20)
    lower = np.asarray(hand.read_joint_lower_limit(), dtype=float).reshape(20)
    upper = np.asarray(hand.read_joint_upper_limit(), dtype=float).reshape(20)
    errors = np.asarray(hand.read_joint_error_code())
    temps = np.asarray(hand.read_joint_temperature(), dtype=float)

    # 安全检查
    if errors.any():
        print(f"错误: 关节错误码非零 (sum={errors.sum()})，请先排查")
        return 1

    # 目标: 全零位（张开），夹在限位内
    target = np.clip(np.zeros(20), lower + 0.02, upper - 0.02)

    finger_names = ["拇指", "食指", "中指", "无名指", "小指"]
    print(f"\n固件: {hand.get_firmware_version()}")
    print(f"温度: max={temps.max():.1f}°C")
    print(f"当前关节位置:")
    for i, name in enumerate(finger_names):
        print(f"  {name}: {current[i]}")
    print(f"目标: 全零位（张开手）")
    print(f"缓入: {args.ramp}s, 保持: {args.hold}s")

    # 执行
    steps = max(int(args.ramp * 50), 20)
    dt = args.ramp / steps

    print("\n>>> 使能 ...")
    hand.write_joint_enabled(True)
    time.sleep(0.1)

    try:
        for step in range(1, steps + 1):
            alpha = step / steps
            interp = current + (target - current) * alpha
            for fj in range(20):
                hand.finger(fj // 4).joint(fj % 4).write_joint_target_position(
                    float(interp[fj]))
            time.sleep(dt)

        time.sleep(args.hold)

        final = np.asarray(hand.read_joint_actual_position(), dtype=float).reshape(20)
        max_err = np.abs(final - target).max()
        print(f"最终关节位置:")
        for i, name in enumerate(finger_names):
            print(f"  {name}: {final[i]}")
        print(f"最大跟踪误差: {max_err*1000:.1f} mrad")
        print("✅ 复位完成")
    finally:
        hand.write_joint_enabled(False)
        print(">>> 已去使能")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
