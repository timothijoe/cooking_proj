# Tianji Arm experiments

本目录保存 Tianji 真机实验模块。运行它们应使用仓库根目录创建的
`.venv-hardware`，而不是系统 Python、旧项目的 `real_robot_debug/` 路径或个人绝对路径。

## 安全入口

先在仓库根目录创建环境并只查看帮助：

```bash
./scripts/setup/create_hardware_env.sh
.venv-hardware/bin/python -m tianji_arm.experiments.arm_command_diagnostic --help
```

`arm_command_diagnostic` 默认只读连接；只有同时提供 `--execute`、关节编号和不超过
`1°` 的位移才会发送受限单关节目标。不要把 `--help`、dry-run 或旧实验记录视为实体设备
已经验收。

## 模块分组

- `plan_sampled_ik_chop`、`shift_right_arm_trajectory`：离线轨迹规划/变换；
- `arm_command_diagnostic`：A 臂反馈诊断和受限单关节试动；
- `keyboard_cartesian_jog`、`gamepad_cartesian_jog`、`gamepad_cartesian_jog_simstyle`：
  受监督点动，默认 dry-run；
- `a_arm_impedance_two_stage`、`arm_impedance_playback`、`dual_arm_batch_stream`：NPZ
  轨迹播放与批流；
- `real_*chop`：切菜控制实验；`bak` 仅保留作历史对照，不是推荐入口。

完整的风险等级、参数和执行顺序见
[`tutorials/hardware/`](../../../../../tutorials/hardware/README.md)；首次真机操作必须先读
[`tutorials/tianji_hardware.md`](../../../../../tutorials/tianji_hardware.md)。

所有真机执行都要求设备旁操作者、可达的物理急停、确认过的设备身份和工作空间。绝不在
未核对这些条件时添加 `--execute`。
