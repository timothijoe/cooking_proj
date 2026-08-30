# Tianji 点动

键盘入口一次按键请求一个保持姿态的基坐标小步；W/S、A/D、R/F 对应 ±X、±Y、±Z，
Q 或空格退出。手柄入口用 RB 作为 deadman，Start 退出；左摇杆控制 X/Y，右摇杆纵轴
控制 Z。

`gamepad_cartesian_jog` 使用离散规划段；`gamepad_cartesian_jog_simstyle` 在固定频率下
积分速度、做 FK/IK 并发送关节目标。二者默认 dry-run，执行模式要求显式工作空间或
启动位姿附近边界，并检查反馈、错误、规划与速度上限。软件停止不是物理急停。
