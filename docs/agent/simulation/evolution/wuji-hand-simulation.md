# Wuji Hand 仿真

Wuji 有两条隔离路径：`twin-sim` 在双臂模型中控制挂载手；`tianji-robot sim` 使用
hand-only 官方模型进行 MCAP 回放、贴桌后退和角度条交互。hand-only 后端不得加载机械臂
或实体 SDK。

轨迹是有限值、严格递增时间戳的 20 关节数组。回放前检查关节范围和单步变化量；贴桌
后退只修正手掌位姿，保留录制中的相对指姿，并限制桌面穿透。输入与输出数据均属于
`local/data/`，不提交 Git。

迁移后已验证双臂中的 `hand-demo`；hand-only 各 Viewer/真实 MCAP 路径尚未在新目录逐项
复验，因此不能把旧项目证据写成新项目验收结果。
