#!/bin/bash
# 播放镜像右手 200Hz 轨迹到实体 Wuji 手
#
# 使用方法:
#   # 默认 0.2x 速度（安全首测）
#   ./scripts/play_mirrored_hand.sh
#
#   # 指定速度
#   ./scripts/play_mirrored_hand.sh 0.1
#
#   # 指定速度和轨迹文件
#   ./scripts/play_mirrored_hand.sh 0.2 recordings/recorded_hand_guarded_chop_200hz_mirrored_right_hand.npz

set -eu

SPEED="${1:-0.2}"
NPZ="${2:-recordings/recorded_hand_guarded_chop_200hz_mirrored_right_hand.npz}"
SERIAL="${3:-365939643134}"
RAMP="${4:-3.0}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$PROJECT_DIR/.venv-wujihand/bin/python"

if [ ! -f "$PROJECT_DIR/$NPZ" ]; then
    echo "错误: 找不到轨迹文件 $PROJECT_DIR/$NPZ"
    exit 1
fi

echo "=== Wuji 镜像手轨迹回放 ==="
echo "轨迹文件: $NPZ"
echo "倍速:     ${SPEED}x"
echo "设备 SN:  $SERIAL"
echo "缓入:     ${RAMP}s"
echo

exec "$PYTHON" - "$PROJECT_DIR/$NPZ" "$SPEED" "$SERIAL" "$RAMP" << 'PYEOF'
import sys
import time
import numpy as np
from wujihandpy import Hand

npz_path, speed_str, serial, ramp_str = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
SPEED = float(speed_str)
RAMP_S = float(ramp_str)

# 加载轨迹
data = np.load(npz_path)
hand_traj = data["right_hand_target_rad"]  # (N, 20)
time_s = data["time_s"]
total_frames = len(hand_traj)
original_dt = time_s[1] - time_s[0]
play_dt = original_dt / SPEED

print(f"轨迹帧数:      {total_frames}")
print(f"原始采样周期:  {original_dt*1000:.0f} ms (200 Hz)")
print(f"回放帧间隔:    {play_dt*1000:.0f} ms")
print(f"原始时长:      {time_s[-1]:.1f}s")
print(f"回放时长:      约 {time_s[-1]/SPEED:.0f}s")
print(f"最大帧间步长:  {np.abs(np.diff(hand_traj, axis=0)).max():.4f} rad")
print()

# 连接
print(f"连接 SN={serial} ...")
hand = Hand(serial_number=serial)

# 安全检查
lower = np.asarray(hand.read_joint_lower_limit(), dtype=float).reshape(20)
upper = np.asarray(hand.read_joint_upper_limit(), dtype=float).reshape(20)
errors = np.asarray(hand.read_joint_error_code())
temps = np.asarray(hand.read_joint_temperature(), dtype=float)

if errors.any():
    print(f"❌ 关节错误码非零 (sum={errors.sum()})，请先排查")
    sys.exit(1)

MARGIN = 0.02
traj_min = hand_traj.min(axis=0)
traj_max = hand_traj.max(axis=0)
violations = []
for i in range(20):
    f, j = i // 4 + 1, i % 4 + 1
    if traj_min[i] < lower.flat[i] + MARGIN:
        violations.append(f"F{f}_J{j}: min={traj_min[i]:.4f} < {lower.flat[i]+MARGIN:.4f}")
    if traj_max[i] > upper.flat[i] - MARGIN:
        violations.append(f"F{f}_J{j}: max={traj_max[i]:.4f} > {upper.flat[i]-MARGIN:.4f}")
if violations:
    print("❌ 轨迹超出硬件限位:")
    for v in violations:
        print(f"   {v}")
    sys.exit(1)

print(f"✅ 轨迹在硬件限位内 (margin={MARGIN}m rad)")
print(f"温度: max={temps.max():.1f}°C")

# 当前到首帧差距
current_pos = np.asarray(hand.read_joint_actual_position(), dtype=float).reshape(20)
first_target = hand_traj[0]
max_jump = np.abs(first_target - current_pos).max()
print(f"当前→首帧最大差距: {max_jump:.4f} rad")
print()

# 缓入
ramp_steps = max(int(RAMP_S / play_dt), 1)
print(f">>> 使能 + {RAMP_S}s 缓入 ({ramp_steps} 步) ...")
hand.write_joint_enabled(True)
time.sleep(0.1)

start = time.time()
try:
    for step in range(1, ramp_steps + 1):
        alpha = step / ramp_steps
        interp = current_pos + (first_target - current_pos) * alpha
        for fj in range(20):
            hand.finger(fj // 4).joint(fj % 4).write_joint_target_position(
                float(interp[fj]))
        time.sleep(play_dt)

    print(f">>> 缓入完成，开始回放 {total_frames} 帧 ...")

    # 回放
    for fi in range(total_frames):
        target = hand_traj[fi]
        for fj in range(20):
            hand.finger(fj // 4).joint(fj % 4).write_joint_target_position(
                float(target[fj]))

        # 每 2 秒实际时间报告一次进度
        if fi % max(int(2.0 / play_dt), 1) == 0:
            elapsed = time.time() - start
            pct = fi / total_frames * 100
            print(f"  帧 {fi}/{total_frames} ({pct:.0f}%)  {elapsed:.0f}s")

        time.sleep(play_dt)

    elapsed = time.time() - start
    print(f">>> 回放完成 ({elapsed:.0f}s)，保持最终姿态 2s ...")
    time.sleep(2.0)

    final_pos = np.asarray(hand.read_joint_actual_position(), dtype=float).reshape(20)
    max_err = np.abs(final_pos - hand_traj[-1]).max()
    print(f"最终姿势最大跟踪误差: {max_err*1000:.1f} mrad")
    print("✅ 回放成功")

finally:
    hand.write_joint_enabled(False)
    print(">>> 已去使能")
PYEOF
