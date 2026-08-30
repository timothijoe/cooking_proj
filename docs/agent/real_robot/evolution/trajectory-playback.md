# 真机轨迹播放

规范联合 NPZ 以 `time_s` 为时间轴，机械臂使用 `left_arm_target_rad` 与
`right_arm_target_rad`（`N×7`），手轨迹按工作流使用 `left_hand_target_rad` 或
`right_hand_target_rad`（`N×20`）。所有数组必须有限、帧数一致、时间严格递增；具体工具
可能进一步要求 5 ms/200 Hz。

播放分为两阶段：从当前反馈用静止端点 quintic 轨迹缓入首帧，再按 `speed-scale` 播放原始
轨迹。首帧差、关节范围、单步变化、反馈状态和跟踪误差必须在执行前或运行中检查。

当前入口：

- `a_arm_impedance_two_stage`：A 臂两阶段播放；
- `arm_impedance_playback`：A、B 或 AB 播放；
- `dual_arm_batch_stream`：同周期批量发送双臂目标；
- `wuji_hand.tools.hand_only_stream`：有显式 `--execute` 的手部播放；
- `cooking_workflows.dual_arm_hand_batch_stream`：双臂和手联合播放；
- `dual_arm_hand_playback`：旧 Concise API 兼容入口。

`probe-current` 会切换模式并发送当前姿态保持帧，不是只读。某些播放器默认结束后保持使能，
另一些支持 `--no-keep-enabled`；调用者必须逐个确认，不能假设统一行为。人工命令以
[`tutorials/hardware/`](../../../../tutorials/hardware/README.md) 为准。

迁移后仅验证了参数、静态接口和自动测试，没有实体播放证据。旧项目的设备状态和运行结果
只能作为历史线索。

旧项目曾有从录制切菜仿真导出 200 Hz 联合 NPZ、生成镜像臂轨迹和官方 Right retarget
候选的开发入口。当前轻量仓库迁入了轨迹消费、平移和播放代码，但没有独立的完整 200 Hz
导出 CLI；现有候选 NPZ 只能作为 `local/data/` 外部资源恢复，不能声称可由当前仓库重建。
