# Tianji 双臂与 Wuji Hand 联合回放

本文迁移自旧项目联合操作手册。联合入口同时拥有机械臂网络连接和 Wuji USB 设备，风险高于
单设备实验。迁移后尚未实体联合复验，必须先分别完成 Tianji 与 Wuji 教程。

## 推荐顺序

1. `ping` 机器人 IP，并运行 Tianji 只读诊断。
2. 读取 Wuji SN、手性、错误码和温度。
3. 对 NPZ 做离线 dry-run，确认左右臂、手、时间轴、首帧和限位。
4. 分别用 `--arm-only` 与 `--hand-only` 做受监督低速验证。
5. 最后才运行完整联合回放。

## Batch stream

```bash
# dry-run：不连接设备
.venv-hardware/bin/python -m cooking_workflows.dual_arm_hand_batch_stream \
  --source-npz local/data/recordings/trajectory.npz

# 只控制双臂
.venv-hardware/bin/python -m cooking_workflows.dual_arm_hand_batch_stream \
  --source-npz local/data/recordings/trajectory.npz \
  --robot-ip 192.168.1.190 \
  --execute --arm-only --entry-duration-s 15 --speed-scale 0.05

# 只控制手
.venv-hardware/bin/python -m cooking_workflows.dual_arm_hand_batch_stream \
  --source-npz local/data/recordings/trajectory.npz \
  --wuji-serial 365939643134 \
  --execute --hand-only --entry-duration-s 15 --speed-scale 0.05

# 双臂与手联合
.venv-hardware/bin/python -m cooking_workflows.dual_arm_hand_batch_stream \
  --source-npz local/data/recordings/trajectory.npz \
  --robot-ip 192.168.1.190 --wuji-serial 365939643134 \
  --execute --entry-duration-s 15 --speed-scale 0.05
```

不加 `--hand-only` 或 `--arm-only` 时控制双臂和手；两个参数不能同时使用。程序检查双臂
反馈帧、state、错误码和跟踪误差，并对手轨迹做限位处理。

`--probe-current` 需要 `--execute`，会对机械臂发送当前姿态保持命令，因此不是只读操作。

## 旧 Concise API 播放器

`cooking_workflows.dual_arm_hand_playback` 也被保留：

```bash
# dry-run
.venv-hardware/bin/python -m cooking_workflows.dual_arm_hand_playback \
  --source-npz local/data/recordings/trajectory.npz

# 真机；仅在需要复现旧 Concise API 流程时使用
.venv-hardware/bin/python -m cooking_workflows.dual_arm_hand_playback \
  --source-npz local/data/recordings/trajectory.npz \
  --robot-ip 192.168.1.190 --wuji-serial 365939643134 \
  --execute --entry-duration-s 15 --speed-scale 0.05 \
  --no-keep-enabled
```

优先使用 batch stream；旧播放器只用于兼容和对照。

## 停止

- 任一设备出现意外动作，立即使用对应物理急停/断能手段。
- `Ctrl+C` 会触发程序清理，但不能替代物理急停。
- 程序退出后分别确认双臂状态和 Wuji 使能状态，不能只看终端退出码。
