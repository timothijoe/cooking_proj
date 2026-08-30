# Tianji NPZ 轨迹播放

本文迁移自旧项目双臂回放教程。轨迹文件必须包含匹配时间轴的左右臂七关节目标，放在
`local/data/recordings/`。所有角度和 A/B 臂映射必须在执行前核对。

## A 臂两阶段阻抗播放

先 dry-run，仅加载和检查轨迹：

```bash
.venv-hardware/bin/python -m tianji_arm.experiments.a_arm_impedance_two_stage \
  --source-npz local/data/recordings/trajectory.npz \
  --speed-scale 0.05 --entry-duration-s 15
```

真机执行：

```bash
.venv-hardware/bin/python -m tianji_arm.experiments.a_arm_impedance_two_stage \
  --source-npz local/data/recordings/trajectory.npz \
  --robot-ip 192.168.1.190 \
  --execute --speed-scale 0.05 --entry-duration-s 15 \
  --no-keep-enabled
```

## A、B 或双臂播放

```bash
# dry-run
.venv-hardware/bin/python -m tianji_arm.experiments.arm_impedance_playback \
  --arm AB --source-npz local/data/recordings/trajectory.npz

# 真机，首测 0.05 倍速并在结束后禁用
.venv-hardware/bin/python -m tianji_arm.experiments.arm_impedance_playback \
  --arm AB --source-npz local/data/recordings/trajectory.npz \
  --robot-ip 192.168.1.190 \
  --execute --speed-scale 0.05 --entry-duration-s 15 \
  --no-keep-enabled
```

`--arm A`、`--arm B` 和 `--arm AB` 分别选择一条或两条臂。程序先从当前反馈用 quintic
平滑进入首帧，再按轨迹时间播放。

## 双臂 batch stream

```bash
# 离线检查
.venv-hardware/bin/python -m tianji_arm.experiments.dual_arm_batch_stream \
  --source-npz local/data/recordings/trajectory.npz

# 当前姿态探针：会连接、切换阻抗状态并发送当前姿态保持命令
.venv-hardware/bin/python -m tianji_arm.experiments.dual_arm_batch_stream \
  --source-npz local/data/recordings/trajectory.npz \
  --robot-ip 192.168.1.190 --execute --probe-current

# 真机轨迹
.venv-hardware/bin/python -m tianji_arm.experiments.dual_arm_batch_stream \
  --source-npz local/data/recordings/trajectory.npz \
  --robot-ip 192.168.1.190 \
  --execute --entry-duration-s 15 --speed-scale 0.05
```

注意：`--probe-current` 不是只读，它会发送当前关节姿态保持帧，只是不要求产生有意位移。
执行前仍按真机运动对待。
