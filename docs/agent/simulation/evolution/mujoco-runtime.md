# MuJoCo 运行时

权威实现位于 `packages/cooking_simulation/src/twin_sim/`。`model.py` 校验模型实体，
`robot.py` 写目标并步进，`kinematics.py` 提供 FK/Jacobian/DLS IK 与路径预检，
`trajectory.py` 生成 minimum-jerk 轨迹，`force_monitor.py` 读取并过滤力，`logging.py`
输出状态记录，`tasks/` 只编排任务。

位置执行器的增益和力限来自 MJCF，不是公开阻抗接口。接触力阈值只产生 warning；
NaN/Inf 必须立即停止。模型通过 `local/assets/` 提供，路径解析规则见
[本地资源接口](../../interfaces/local-resources.md)。

最新验证以 [当前状态](../current.md) 为准。后续若加入导纳或力控，应作为明确的上层控制器，
不能把它伪装成当前的位置控制能力。
