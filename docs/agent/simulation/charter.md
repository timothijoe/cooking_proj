# 仿真总章程

仿真用于在不接触实体设备的情况下验证模型、轨迹、状态机、安全几何和数据接口。
`twin_sim` 内部统一使用 rad、m、s、N；不得导入实体 SDK，也不得让观测阈值悄悄改变轨迹。

稳定依赖方向为“CLI/任务 → 轨迹与连续 IK → 关节目标 → MuJoCo → 状态、接触与记录”。
模型和录制等大资源只通过 `local/` 或 `COOKING_LOCAL_ROOT` 注入。新增任务必须具备
headless 路径、有限值检查、全路径预检、可解释结果以及对应回归测试。

当前实现细节以 [MuJoCo 运行时](evolution/mujoco-runtime.md)、
[护手切菜](evolution/guarded-chopping.md)和
[Wuji Hand 仿真](evolution/wuji-hand-simulation.md)为准。
