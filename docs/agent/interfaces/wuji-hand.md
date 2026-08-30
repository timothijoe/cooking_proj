# Wuji Hand 接口

安装入口：`tianji-robot = tianji_robotics.cli:main`。仿真域提供 `wuji-replay`、
`wuji-table-retreat`、`wuji-angle-bar`；硬件域当前可靠的公开安全入口是
`hardware wuji-sdk preflight`。

规范轨迹包含递增 `time_s` 与 `(N, 20)` 关节弧度。SDK 依赖从 `local/vendor` 注入；
录制从 `local/data/recordings*` 注入。实体发送工具是实验接口，不得由仿真模块导入。
