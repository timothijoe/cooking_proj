# Tianji 点动

键盘入口一次按键请求一个保持姿态的基坐标小步；W/S、A/D、R/F 对应 ±X、±Y、±Z，
Q 或空格退出。手柄入口用 RB 作为 deadman，Start 退出；左摇杆控制 X/Y，右摇杆纵轴
控制 Z。

`gamepad_cartesian_jog` 使用离散规划段；`gamepad_cartesian_jog_simstyle` 在固定频率下
积分速度、做 FK/IK 并发送关节目标。二者默认 dry-run，执行模式要求显式工作空间或
启动位姿附近边界，并检查反馈、错误、规划与速度上限。软件停止不是物理急停。

对于新的人机遥操作，应优先采用 `gamepad_cartesian_jog_simstyle`：它以 50 Hz 连续循环运行，
不等待单段 MOVLA 轨迹完成，并支持将基座坐标平移与 TCP 坐标系的 yaw/pitch/roll 同时输入。
离散 `gamepad_cartesian_jog` 保留用于保守的分段对照和故障隔离。

2026-09-06 的现场检查确认了该选择的运行前提：A 臂控制器网络与只读 SDK 反馈正常、伺服
错误码为零，且默认 Xbox 手柄节点存在。实际左臂运动尚未在迁移后验收，因此首测应覆盖
`--arm left --speed-mm-s 20 --workspace-around-current-mm 50 --orientation-rate-dps 20`，并只在
确认急停、净空和单轴方向后添加 `--execute`。手柄映射为左杆上下→基座 X、左杆左右→基座 Z、
右杆上下→基座 Y、右杆左右→工具 yaw、D-pad→工具 pitch/roll；RB 是 deadman，Start 退出。
