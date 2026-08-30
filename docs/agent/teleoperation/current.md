# 遥操作当前状态

Tianji 提供键盘单步、手柄笛卡尔点动和仿真风格连续 FK/IK 点动；包装脚本在
`scripts/teleop/tianji/`。默认只读反馈和打印目标，实际运动要求 `--execute`。

Wuji 保留 angle-bar 仿真和实体轨迹播放工具，但 `scripts/teleop/wuji/play_mirrored_hand.sh`
仍引用旧环境和目录，尚未完成轻量仓库适配，不能作为当前推荐入口。实体设备均未在迁移后
复验。人工操作见[遥操作教程](../../../tutorials/teleoperation.md)。
