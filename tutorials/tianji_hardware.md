# Tianji 真机入门

本页从查看参数、只读连接到受限的单关节试动逐级说明。当前诊断程序只支持 SDK A 臂；
迁移后尚未重新完成现场验收，因此第一次运动必须由设备旁操作者监督。

## 1. 现场检查

运行前确认：

- 物理急停可达并已经验证；
- 机器人周围无人，机械臂、工具、线缆和工作台之间有足够净空；
- 当前设备确实将目标机械臂映射为 SDK A 臂；
- 机器人 IP、工具安装、关节正方向和现场坐标已经核对；
- 操作者知道如何断电或用控制器停止。`Ctrl+C` 和软件 disable 不能代替物理急停。

## 2. 安装环境并查看帮助

```bash
./scripts/setup/create_hardware_env.sh
.venv-hardware/bin/python -m tianji_arm.experiments.arm_command_diagnostic --help
.venv-hardware/bin/python -m tianji_arm.experiments.plan_sampled_ik_chop --help
```

`--help` 只打印参数并退出，不连接控制器，也不会运动。

## 3. 只读真机诊断

下面的命令会连接 `192.168.1.190`，持续 5 秒读取 A 臂状态和关节反馈，但不会切换控制模式、
使能机械臂或发送关节目标：

```bash
.venv-hardware/bin/python -m tianji_arm.experiments.arm_command_diagnostic \
  --robot-ip 192.168.1.190 \
  --monitor-s 5
```

正常输出应先出现连接信息和 `READ_ONLY: no controller state or joint target was sent`，随后
持续打印递增的 `frame`、控制器 `state` 和七个 `actual_deg`。如果连接失败、帧号不更新、
关节反馈不是七个有限数值或 `err_code` 非零，程序会报错退出；不要继续运动试验。

该模式虽然不发送运动命令，但仍会连接真实控制器读取反馈。

## 4. 受限单关节真机试动

只有只读反馈正常、现场检查完成后，才可显式加入 `--execute`。以下示例让 A 臂第 1 关节
从当前反馈位置移动 `+0.5°`，然后监控 5 秒：

```bash
.venv-hardware/bin/python -m tianji_arm.experiments.arm_command_diagnostic \
  --robot-ip 192.168.1.190 \
  --execute \
  --trial-joint 1 \
  --trial-delta-deg 0.5 \
  --monitor-s 5
```

安全限制：

- 只允许 A 臂；
- `--trial-joint` 必须是 `1` 到 `7`；
- `--trial-delta-deg` 必须在 `[-1, 0)` 或 `(0, 1]` 度内；
- `--execute`、`--trial-joint` 和 `--trial-delta-deg` 缺一不可；
- 目标超过程序内关节限位会在发送前被拒绝；
- 程序只发送一个受限目标，随后打印目标、实际角度和最大误差；
- 进入执行模式后，无论正常结束或发生异常，程序都会在 `finally` 中尝试去使能 A 臂并释放连接。

首次建议从 `0.2°` 到 `0.5°` 开始。负方向试动使用负值，例如
`--trial-delta-deg -0.2`。改变关节编号或方向前重新检查该关节的实际净空。

## 5. 停止与异常

- 发现非预期运动时立即使用物理急停；不要等待程序自行退出。
- 正常情况下可按 `Ctrl+C` 中止，程序会进入清理流程，但这只是软件停止。
- 若出现 `WARNING: failed to disable A arm`，应立即通过控制器或物理安全系统确认设备状态。
- 若命令返回 `ERROR`，先解决连接、反馈、错误码或限位问题，不要反复添加 `--execute` 重试。

## 6. 其他实验入口

遥操作 dry-run 见[遥操作教程](teleoperation.md)。部分 dry-run 会连接并读取反馈，但不发送
运动。切菜、连续阻抗、轨迹进入和联合双臂属于更高风险实验，不应从本入门示例直接推导
参数；不要复制旧项目中的 IP、初始角度、工作空间或切菜参数直接运行。

所有 trace CSV 写入 `local/data/recordings/`。异常时先停止命令并使用物理急停；键盘空格或
deadman 只是软件停止，不是急停替代品。
