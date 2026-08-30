# Retargeting 当前状态

当前可以解析 Wuji Studio 右手骨架 MCAP，镜像为左手语义，调用官方离线 retargeter，
输出规范 NPZ 或 JointState 语义 MCAP；标准 JointState MCAP 可直接跳过重定向。还保留基于
录制动作区间的贴桌后退和双臂切菜联动。

代码与测试已迁移，旧项目的真实录制曾成功处理；新仓库内真实 MCAP 路径尚未逐项复验。
输入输出位于 `local/data/recordings*`。人工步骤见
[Retargeting 教程](../../../tutorials/retargeting.md)。
