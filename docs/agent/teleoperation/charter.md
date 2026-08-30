# 遥操作总章程

遥操作只把有界的人机输入转换为小步或限速目标，不负责绕过规划器、安全状态或设备限位。
真机入口默认 dry-run；连续输入必须有 deadman，释放后停止产生新运动目标，退出时清理并
去使能。工作空间、速度、反馈新鲜度和控制器错误均应在每次发送前检查。

当前能力详见 [Tianji 点动](evolution/tianji-jogging.md) 与
[Wuji 交互控制](evolution/wuji-interactive-control.md)。
