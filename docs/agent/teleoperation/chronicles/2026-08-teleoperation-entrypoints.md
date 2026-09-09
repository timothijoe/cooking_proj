# 2026-08：键盘与手柄遥操作入口

旧项目形成了 Tianji 键盘单步、手柄离散规划和仿真风格连续 FK/IK 三种入口，并统一采用
dry-run 默认、显式 `--execute`、工作空间约束和退出清理。迁移时三套 Python 模块与 Tianji
包装脚本全部保留。

Wuji 的交互仿真和实体工具也被迁入，但旧 shell 包装未随新目录完全适配。因此当前记录将
其标为待修复，而不是把“文件存在”误写成“已经可用”。

## 2026-09-06：左臂与手柄现场连通性复查

为准备左臂手柄遥操作，现场检查了迁移后的运行环境。控制器 `192.168.1.190` 的两次 ICMP
探测均成功（0% 丢包，约 1.4 ms）；`arm_command_diagnostic` 以默认只读模式成功经 Marvin
SDK 连接 SDK A 臂。七个伺服错误码均为 0，反馈帧持续更新；当时控制器报告 A 臂 `state=0`。
诊断没有请求状态切换或关节目标。

Xbox 手柄已枚举为
`/dev/input/by-id/usb-Microsoft_Xbox360_For_Windows-joystick`，并指向 `/dev/input/js0`。
左臂应使用较新的 `gamepad_simstyle_jog.sh --arm left`（连续 50 Hz FK/IK），而非较早的
离散 `gamepad_jog.sh`。首个实体运动仍待在物理急停、净空、轴方向和低速小工作空间均确认后
执行；本次记录不是运动验收。
