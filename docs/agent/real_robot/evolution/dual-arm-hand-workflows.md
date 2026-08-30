# 双臂与手联合工作流

`cooking_workflows.dual_arm_hand_batch_stream` 和 `dual_arm_hand_playback` 负责组合 Tianji 双臂
与 Wuji 手轨迹。它们应只编排已验证的领域接口，不直接隐藏 SDK 生命周期、单位转换或安全门。

联合运行前必须分别通过机械臂、手、轨迹时间轴和设备所有权检查，再验证同步停止策略。
当前仅完成迁移和自动测试，未完成现场联合验证，因此不得作为默认教程入口。
