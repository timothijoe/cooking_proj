# 仿真接口

安装入口：`twin-sim = twin_sim.cli:main`。

当前子命令：`view`、`gamepad-teleop`、`hand-demo`、`joint`、`cartesian`、`chop`、
`line-chop`、`pick-place`、`guarded-chop`、`recorded-hand-guarded-chop`、
`guarded-chop-replay`。所有自动验证优先使用 `--headless`；Viewer 仅用于人工视觉验收。

包内 API 使用 SI 单位。模型依赖 `local/assets/robot_assets` 与 `local/assets/MarvinCCS`；
录制/回放文件应位于 `local/data/`。任务结果必须显式报告成功、阶段计数和安全指标。
