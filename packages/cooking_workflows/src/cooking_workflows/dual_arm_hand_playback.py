#!/usr/bin/env python3
"""Tianji 双臂 + Wuji 手联合 NPZ 轨迹回放。

从同一个 200Hz NPZ 文件读取左臂、右臂和右手 20 关节目标，
同时控制 Tianji 机器人（A 臂 + B 臂）和 Wuji 手。

USAGE:
    # 干跑（只看规划，不连任何设备）
    PYTHONPATH=. python3 real_robot_debug/dual_arm_hand_playback.py

    # 真机执行（双臂 + 手）
    PYTHONPATH=. python3 real_robot_debug/dual_arm_hand_playback.py --execute

    # 只控制手，不控制臂
    PYTHONPATH=. python3 real_robot_debug/dual_arm_hand_playback.py --execute --hand-only

    # 只控制臂，不控制手
    PYTHONPATH=. python3 real_robot_debug/dual_arm_hand_playback.py --execute --arm-only

Safety:
    - 默认干跑，--execute 才会连接设备
    - 各关节轨迹独立检查硬件限位
    - Ctrl+C 安全停止所有设备
    - 手部目标超出硬件限位自动裁剪
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
WUJI_SERIAL = "365939643134"

# ── 关节阻抗参数 ──
DEFAULT_JOINT_K = (8.0, 8.0, 8.0, 4.0, 2.0, 1.5, 1.0)
DEFAULT_JOINT_D = (0.8, 0.8, 0.8, 0.6, 0.4, 0.3, 0.2)

# ── 关节限位 ──
# 双臂 (degrees)
ARM_JOINT_LIMITS_DEG = [
    (-170, 170), (-100, 120), (-170, 170), (-130, 130),
    (-170, 170), (-90, 220), (-170, 170),
]
# Wuji 手 (rad) — 从硬件读取的限位
HAND_JOINT_LIMITS_RAD = {
    "lower": np.array([
        [-0.11925496, -0.25735966, -0.54967515, -0.58205475],
        [-0.26239912, -0.43876813, -0.52905068, -0.50485284],
        [-0.23443433, -0.44822569, -0.56918519, -0.57158002],
        [-0.24246004, -0.47819426, -0.57648532, -0.5434772 ],
        [-0.25753832, -0.47106056, -0.68819585, -0.57163156],
    ]),
    "upper": np.array([
        [1.67901294, 0.94978802, 1.66274323, 1.64120234],
        [1.64607823, 0.34152377, 1.67151943, 1.63989344],
        [1.6439589 , 0.33032859, 1.64428097, 1.66215395],
        [1.64423041, 0.27520675, 1.67904946, 1.68428525],
        [1.63236697, 0.2921523 , 1.53570436, 1.64613478],
    ]),
}
HAND_MARGIN = 0.02  # rad

ARMS = [
    {"sdk_arm": "A", "index": 0, "npz_key": "left_arm_target_rad",  "name": "Left  (A)"},
    {"sdk_arm": "B", "index": 1, "npz_key": "right_arm_target_rad", "name": "Right (B)"},
]

FINGER_NAMES = ["拇指", "食指", "中指", "无名指", "小指"]

# ── 信号处理 ──
_interrupted = False


def _on_sigint(signum, frame):
    global _interrupted
    _interrupted = True
    print("\n⚠️  Ctrl+C — 正在停止所有设备 ...")


signal.signal(signal.SIGINT, _on_sigint)


# ═══════════════════════════════════════════════════════════════
# 轨迹加载与安全校验
# ═══════════════════════════════════════════════════════════════


def load_trajectory(path: Path) -> dict:
    """加载 NPZ，返回双臂 + 手部轨迹数据。"""
    with np.load(path, allow_pickle=False) as data:
        time_s = np.asarray(data["time_s"], dtype=float)

        result = {
            "time_s": time_s - time_s[0],
            "total_frames": time_s.size,
            "total_duration_s": float(time_s[-1] - time_s[0]),
            "arms": {},
            "hand_rad": np.asarray(data["right_hand_target_rad"], dtype=float),
        }

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
    if np.any(intervals <= 0) or not np.allclose(intervals, intervals[0], rtol=0, atol=1e-9):
        raise ValueError("time_s must be strictly increasing and uniform")
    if result["hand_rad"].shape != (time_s.size, 20) or not np.all(np.isfinite(result["hand_rad"])):
        raise ValueError("right_hand_target_rad must be finite (N,20)")

    return result


def check_arm_limits(traj: dict):
    """检查双臂轨迹是否在硬件限位内。"""
    for arm in ARMS:
        targets_deg = traj["arms"][arm["sdk_arm"]]["targets_deg"]
        for j in range(7):
            lo, hi = ARM_JOINT_LIMITS_DEG[j]
            col = targets_deg[:, j]
            if col.min() < lo + 1.0 or col.max() > hi - 1.0:
                raise ValueError(
                    f"{arm['name']} Joint {j+1}: [{col.min():.1f}, {col.max():.1f}] deg "
                    f"超出限位 [{lo}, {hi}] (1 deg margin)"
                )


def check_and_clip_hand(traj: dict) -> np.ndarray:
    """检查手部轨迹是否在硬件限位内，超出部分自动裁剪。返回裁剪后轨迹。"""
    hand_traj = traj["hand_rad"]  # (N, 20)
    clipped = hand_traj.copy().reshape(-1, 5, 4)
    lower = HAND_JOINT_LIMITS_RAD["lower"]
    upper = HAND_JOINT_LIMITS_RAD["upper"]
    violations = []
    for i in range(5):
        for j in range(4):
            lo, hi = lower[i, j] + HAND_MARGIN, upper[i, j] - HAND_MARGIN
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


def build_entry(start_deg, first_target_deg, duration_s=10.0, control_hz=200.0):
    """quintic 平滑缓入轨迹。"""
    steps = int(round(duration_s * control_hz))
    if steps < 1:
        steps = 1
    phase = np.linspace(0, 1, steps + 1)
    smooth = 10 * phase**3 - 15 * phase**4 + 6 * phase**5
    result = start_deg + smooth[:, None] * (first_target_deg - start_deg)
    result[0] = start_deg
    result[-1] = first_target_deg
    return result


# ═══════════════════════════════════════════════════════════════
# 参数解析
# ═══════════════════════════════════════════════════════════════


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-npz", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--robot-ip", default=ROBOT_IP)
    parser.add_argument("--kine-config", type=Path, default=DEFAULT_KINE_CONFIG)
    parser.add_argument("--wuji-serial", default=WUJI_SERIAL)
    parser.add_argument("--speed-scale", type=float, default=0.05,
                        help="回放倍速 (0.01-1.0)。所有设备同步。")
    parser.add_argument("--entry-duration-s", type=float, default=15.0,
                        help="双臂缓入时长（秒）。")
    parser.add_argument("--execute", action="store_true",
                        help="连接真机并执行。不指定则干跑。")
    parser.add_argument("--hand-only", action="store_true",
                        help="只控制 Wuji 手，不控制机械臂。")
    parser.add_argument("--arm-only", action="store_true",
                        help="只控制机械臂，不控制 Wuji 手。")
    parser.add_argument("--no-keep-enabled", action="store_true",
                        help="播放完后禁用双臂。")
    args = parser.parse_args(argv)

    if not 0 < args.speed_scale <= 1.0:
        raise ValueError("--speed-scale must be in (0, 1.0]")
    if args.entry_duration_s <= 0:
        raise ValueError("--entry-duration-s must be positive")
    if not args.source_npz.is_file():
        raise ValueError(f"source NPZ not found: {args.source_npz}")
    if args.hand_only and args.arm_only:
        raise ValueError("--hand-only 和 --arm-only 不能同时使用")

    return args


# ═══════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════


def main(argv: list[str] | None = None) -> int:
    global _interrupted

    try:
        args = parse_args(argv)
    except ValueError as e:
        print(f"error: {e}")
        return 2

    # ── 加载轨迹 ──
    print(f"Loading: {args.source_npz}")
    traj = load_trajectory(args.source_npz)
    total_frames = traj["total_frames"]
    source_dt = 0.005  # 200 Hz
    play_dt = source_dt / args.speed_scale
    play_duration = traj["total_duration_s"] / args.speed_scale

    print(f"总帧数:     {total_frames}")
    print(f"原始 DT:    5 ms (200 Hz)")
    print(f"播放 DT:    {play_dt*1000:.1f} ms (speed={args.speed_scale}x)")
    print(f"原始时长:   {traj['total_duration_s']:.1f}s")
    print(f"回放时长:   {play_duration:.1f}s")
    print()

    # ── 安全检查 ──
    if not args.hand_only:
        check_arm_limits(traj)
        print("✅ 双臂轨迹在硬件限位内")

    if not args.arm_only:
        hand_clipped = check_and_clip_hand(traj)
    else:
        hand_clipped = None

    # ── 打印轨迹范围 ──
    if not args.hand_only:
        for arm in ARMS:
            targets_deg = traj["arms"][arm["sdk_arm"]]["targets_deg"]
            max_step = np.abs(np.diff(targets_deg, axis=0)).max()
            print(f"\n{arm['name']} (degrees):")
            for j in range(7):
                print(f"  J{j+1}: [{targets_deg[:,j].min():7.1f}, {targets_deg[:,j].max():7.1f}]")
            print(f"  最大帧间步长: {max_step:.2f} deg")

    if not args.arm_only:
        print(f"\nWuji 手 (rad):")
        for i in range(5):
            for j in range(4):
                col = hand_clipped[:, i * 4 + j]
                print(f"  {FINGER_NAMES[i]} J{j+1}: [{col.min():.4f}, {col.max():.4f}]")

    print()

    if not args.execute:
        print(f"[干跑] 规划完成，未连接任何设备。")
        print(f"[干跑] 双臂缓入: {args.entry_duration_s}s → 首帧")
        print(f"[干跑] 回放: {total_frames} 帧 at {args.speed_scale}x")
        print(f"[干跑] 模式: {' 手+臂' if not args.hand_only and not args.arm_only else ' 仅手' if args.hand_only else ' 仅臂'}")
        print()
        sys.stdout.flush()
        return 0

    # ═════════════════════════════════════════════════════════════
    # 真机执行
    # ═════════════════════════════════════════════════════════════

    # 按需导入
    if not args.hand_only:
        from SDK_PYTHON.fx_robot import DCSS, Marvin_Robot, Concise_Marvin_Robot
        dcss = DCSS()
        robot = None
        arms_connected = False

    if not args.arm_only:
        from wujihandpy import Hand
        wuji_hand = None
        hand_connected = False

    try:
        # ── 连接 Wuji 手 ──
        if not args.arm_only:
            print(f"\n连接 Wuji 手 SN={args.wuji_serial} ...")
            wuji_hand = Hand(serial_number=args.wuji_serial)
            hand_connected = True
            # 检查手部状态
            errors = np.asarray(wuji_hand.read_joint_error_code())
            if errors.any():
                print(f"⚠️ Wuji 手错误码非零 (sum={errors.sum()})，但继续执行")
            temps = np.asarray(wuji_hand.read_joint_temperature(), dtype=float)
            print(f"Wuji 手温度: max={temps.max():.1f}°C")
            print("✅ Wuji 手已连接")

        # ── 连接 Tianji 机器人 ──
        if not args.hand_only:
            print(f"\n连接机器人 {args.robot_ip} ...")
            base_robot = Marvin_Robot()
            ok = base_robot.connect(args.robot_ip)
            if not ok:
                raise RuntimeError("连接机器人失败")
            arms_connected = True
            base_robot.check_error_and_clear(dcss)
            base_robot.log_switch("1")
            base_robot.local_log_switch("1")
            time.sleep(0.3)

            # 读取双臂状态
            sub = base_robot.subscribe(dcss)
            current_joints = {}
            first_targets = {}
            for arm in ARMS:
                idx = arm["index"]
                raw = [float(v) for v in sub["outputs"][idx]["fb_joint_pos"]]
                current_joints[arm["sdk_arm"]] = np.array(raw)
                first_targets[arm["sdk_arm"]] = traj["arms"][arm["sdk_arm"]]["targets_deg"][0]
                cur_state = sub["states"][idx]["cur_state"]
                err_code = sub["states"][idx]["err_code"]
                max_jump = np.abs(current_joints[arm["sdk_arm"]] - first_targets[arm["sdk_arm"]]).max()
                print(f"{arm['name']}: state={cur_state}, err={err_code}, "
                      f"pos={[round(v,1) for v in raw]}, jump={max_jump:.1f}deg")

            # 断开重连，用 Concise API
            base_robot.release_robot()
            time.sleep(0.3)
            robot = Concise_Marvin_Robot()
            ok = robot.connect(args.robot_ip)
            if not ok:
                raise RuntimeError("连接简明 SDK 失败")
            print("✅ 简明 SDK 已连接")

            # 配置双臂：先 position 模式，再 impedance
            print("\n配置双臂 joint impedance ...")
            joint_k = list(DEFAULT_JOINT_K)
            joint_d = list(DEFAULT_JOINT_D)

            for arm in ARMS:
                sdk_arm = arm["sdk_arm"]
                ok = robot.set_position_state(arm=sdk_arm, velRatio=50, AccRatio=50)
                if ok is False:
                    print(f"  ⚠️ {sdk_arm} set_position_state 失败，跳过")
                else:
                    print(f"  {sdk_arm} position mode 配置成功")
                time.sleep(0.2)

            time.sleep(0.5)
            sub = robot.subscribe(dcss)
            for arm in ARMS:
                idx = arm["index"]
                raw = [float(v) for v in sub["outputs"][idx]["fb_joint_pos"]]
                current_joints[arm["sdk_arm"]] = np.array(raw)
                cur_state = sub["states"][idx]["cur_state"]
                print(f"{arm['name']} position mode: state={cur_state}, "
                      f"pos={[round(v,1) for v in raw]}")

            for arm in ARMS:
                sdk_arm = arm["sdk_arm"]
                ok = robot.set_imp_joint_state(
                    arm=sdk_arm, velRatio=100, AccRatio=100,
                    K=joint_k, D=joint_d,
                )
                if ok is False:
                    print(f"  ⚠️ {sdk_arm} set_imp_joint_state 失败，回退到 position mode")
                    robot.set_position_state(arm=sdk_arm, velRatio=50, AccRatio=50)
                else:
                    print(f"  {sdk_arm} impedance 配置成功")
                time.sleep(0.3)

            time.sleep(0.5)
            sub = robot.subscribe(dcss)
            for arm in ARMS:
                idx = arm["index"]
                raw = [float(v) for v in sub["outputs"][idx]["fb_joint_pos"]]
                current_joints[arm["sdk_arm"]] = np.array(raw)
                cur_state = sub["states"][idx]["cur_state"]
                max_jump = np.abs(current_joints[arm["sdk_arm"]] - first_targets[arm["sdk_arm"]]).max()
                print(f"{arm['name']} 最终状态: state={cur_state}, "
                      f"pos={[round(v,1) for v in raw]}, jump={max_jump:.1f}deg")
                if max_jump > 90:
                    print(f"  ⚠️ 注意: {arm['name']} 反馈可能不是真实编码器值")

            print("✅ 双臂配置完成")

        # ═══════════════════════════════════════════════════════════
        # Phase 1: 双臂缓入 + 手使能
        # ═══════════════════════════════════════════════════════════
        entry_duration = args.entry_duration_s
        entry_hz = 200.0
        entry_dt = 1.0 / entry_hz

        # 双臂缓入轨迹
        if not args.hand_only:
            entry_trajs = {}
            for sdk_arm in current_joints:
                entry_trajs[sdk_arm] = build_entry(
                    current_joints[sdk_arm],
                    first_targets[sdk_arm],
                    duration_s=entry_duration,
                    control_hz=entry_hz,
                )
            entry_frames = len(list(entry_trajs.values())[0])
        else:
            entry_frames = int(entry_duration * entry_hz)
            entry_trajs = {}

        # 手部使能
        if not args.arm_only:
            print(f"\n>>> Wuji 手使能 ...")
            wuji_hand.write_joint_enabled(True)
            time.sleep(0.1)
            print("✅ Wuji 手已使能")

        print(f"\n[Phase 1/2] 缓入: {entry_duration}s → {entry_frames} 帧")
        print("            (Ctrl+C 可随时停止)")

        # 手部缓入（从当前到手首帧）
        if not args.arm_only:
            hand_current = np.asarray(wuji_hand.read_joint_actual_position(), dtype=float).reshape(20)
            hand_first_target = hand_clipped[0]
            hand_entry = build_entry(
                hand_current, hand_first_target,
                duration_s=entry_duration, control_hz=entry_hz,
            )
        else:
            hand_entry = None

        for i in range(entry_frames):
            if _interrupted:
                break

            # 双臂命令
            if not args.hand_only:
                for sdk_arm in entry_trajs:
                    robot.set_joint_position_cmd(sdk_arm, entry_trajs[sdk_arm][i].tolist())

            # 手部命令
            if not args.arm_only and hand_entry is not None:
                wuji_hand.write_joint_target_position(hand_entry[i].reshape(5, 4))

            time.sleep(entry_dt)

        if _interrupted:
            print("缓入阶段被中断。")
        else:
            print("✅ 缓入完成")

        # ═══════════════════════════════════════════════════════════
        # Phase 2: 同步回放
        # ═══════════════════════════════════════════════════════════
        if not _interrupted:
            print(f"\n[Phase 2/2] 回放: {total_frames} 帧 "
                  f"at {args.speed_scale}x ({play_duration:.1f}s)")
            print("            (Ctrl+C 可随时停止)")

            playback_t0 = time.perf_counter()
            for fi in range(total_frames):
                if _interrupted:
                    break

                # 双臂
                if not args.hand_only:
                    for arm in ARMS:
                        sdk_arm = arm["sdk_arm"]
                        target = traj["arms"][sdk_arm]["targets_deg"][fi]
                        robot.set_joint_position_cmd(sdk_arm, target.tolist())

                # 手部
                if not args.arm_only:
                    wuji_hand.write_joint_target_position(
                        hand_clipped[fi].reshape(5, 4)
                    )

                # Wall-clock 同步
                if fi < total_frames - 1:
                    target_time = playback_t0 + (fi + 1) * play_dt
                    sleep_for = target_time - time.perf_counter()
                    if sleep_for > 0:
                        time.sleep(sleep_for)

            if _interrupted:
                print("回放阶段被中断。")
            else:
                print("✅ 回放完成")

    except Exception as e:
        print(f"❌ 错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # ── 清理 ──
        print("\n>>> 清理中 ...")

        # 去使能 Wuji 手
        if not args.arm_only and hand_connected and wuji_hand is not None:
            try:
                wuji_hand.write_joint_enabled(False)
                print("Wuji 手已去使能")
            except Exception as e:
                print(f"Wuji 手去使能失败: {e}")

        # 禁用双臂
        if not args.hand_only and arms_connected and robot is not None:
            if args.no_keep_enabled or _interrupted:
                for arm in ARMS:
                    try:
                        robot.disable(arm["sdk_arm"])
                    except Exception:
                        pass
                print("双臂已禁用")
            else:
                print("双臂保持使能 (--no-keep-enabled 可禁用)")

        # 释放机器人
        if not args.hand_only and arms_connected and robot is not None:
            try:
                robot.release_robot()
            except Exception:
                pass
            print("机器人已释放")

    if _interrupted:
        print("\n⚠️ 被 Ctrl+C 中断")
        return 130

    return 0


if __name__ == "__main__":
    raise SystemExit(main())