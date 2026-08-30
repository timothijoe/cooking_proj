# 仿真当前状态

当前保留独立手部演示、关节/笛卡尔运动、切菜、抓放、双臂护手切菜、录制手势联动切菜、
录制与状态回放。稳定入口是 `.venv/bin/twin-sim`，人工步骤见
[仿真教程](../../../tutorials/simulation.md)。

迁移后已完成完整测试：458 passed、6 skipped、4 xfailed；另有一条普通 chop 接触力
观察阈值 warning。真实 headless 验证包括 hand-demo、5 刀 4 次退手 guarded-chop，
以及 5 刀 5 段手势的 recorded-hand-guarded-chop。GUI guarded-chop 也已人工展示。

未验证项：迁移后其余 Viewer 命令未逐项人工观看；录制联动依赖本地 MCAP；仿真不等于
真机安全证明。详细状态见 `evolution/` 三篇纪传体。
