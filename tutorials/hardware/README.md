# 真机实验教程索引

本目录迁移自 `tianji_robotic_project` 的真机教程，并已替换为当前分包、`.venv-hardware`
和 `local/data/` 路径。迁移后没有重新完成实体设备验收，命令的“旧项目状态”不等于当前
仓库已上机通过。

建议顺序：

1. [Tianji 诊断与单关节试动](../tianji_hardware.md)
2. [Tianji 切菜与位置实验](tianji_chopping.md)
3. [Tianji 键盘和手柄点动](tianji_teleoperation.md)
4. [Tianji NPZ 轨迹播放](tianji_trajectory_playback.md)
5. [Wuji Hand 真机操作](wuji_hand.md)
6. [双臂与 Wuji Hand 联合回放](combined_playback.md)
7. [全部真机实验入口总表](experiment_inventory.md)
8. [Wuji 手套网络接入](wuji_glove_network.md)
9. [二维压力传感器 UART 采集与热力图](sensor_uart_stream.md)

风险分级：

| 标识 | 行为 |
|---|---|
| 离线 | 不连接设备 |
| 只读连接 | 连接设备并读反馈，不发送运动目标 |
| dry-run | 按具体工具定义；可能连接读取反馈，必须阅读说明 |
| 真机执行 | 会使能或发送目标，只能在设备旁运行 |

所有真机执行前都要确认物理急停、净空、设备身份、A/B 臂或左右手映射、工具和坐标方向。
禁止仅凭旧项目中的默认 IP、序列号、初始关节角或工作空间直接执行。
