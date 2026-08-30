# Tianji 实验接口

模块位于 `tianji_arm.experiments`。`arm_command_diagnostic` 用于命令能力诊断；
`plan_sampled_ik_chop` 仅离线规划；sampled position、sampled joint impedance、planned
Cartesian 和 IK/cart impedance 是不同控制实验，不能在文档中混为一个“切菜接口”。

键盘和手柄工具默认 dry-run，但可能连接控制器读取启动反馈；实际发送必须加 `--execute`。
CSV 应记录目标/实际笛卡尔与关节状态、速度、力矩和六轴力反馈，并写入 `local/data/`，
禁止恢复旧 `/home/...` 绝对路径。
