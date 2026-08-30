# Wuji 交互控制

`tianji-robot sim wuji-angle-bar` 用本地录制和 hand-only MuJoCo 提供交互式关节观察，
不连接实体手。实体侧的发送姿态、复位、半握拳和轨迹播放工具位于
`wuji_hand.tools`，需要厂商 SDK、设备身份和现场安全确认。

旧的镜像轨迹 shell 包装仍依赖 `.venv-wujihand` 与旧 recordings 布局，迁移后应先修正为
`.venv-hardware` 和 `local/data/recordings`，再做 shell 语法、dry-run/设备隔离和实体低速
验收。修正之前只可读源码，不应照教程执行。

`wuji-angle-bar` 的 MuJoCo 滑条、Save Pose 和 Record 功能可以作为离线工具使用；但当前
“Send to Hand”按钮仍硬编码旧 `.venv-wujihand` 与旧脚本路径，迁移后不可用。发送真机应
保存 NPZ 后，按人类真机教程显式运行 `.venv-hardware/bin/wuji-send-pose`，不能依赖按钮。
