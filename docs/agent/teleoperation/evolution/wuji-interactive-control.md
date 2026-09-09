# Wuji 交互控制

`tianji-robot sim wuji-angle-bar` 用本地录制和 hand-only MuJoCo 提供交互式关节观察，
不连接实体手。实体侧的发送姿态、复位、半握拳和轨迹播放工具位于
`wuji_hand.tools`，需要厂商 SDK、设备身份和现场安全确认。

旧的镜像轨迹 shell 包装仍依赖 `.venv-wujihand` 与旧 recordings 布局，迁移后应先修正为
`.venv-hardware` 和 `local/data/recordings`，再做 shell 语法、dry-run/设备隔离和实体低速
验收。修正之前只可读源码，不应照教程执行。

`wuji-angle-bar` 的 MuJoCo 滑条、Save Pose 和 Record 功能可以作为离线工具使用。

## Send to Hand 按钮（2026-09-06 已修复并验证）

面板 **Send to Hand** 按钮此前硬编码旧 `.venv-wujihand` 与旧脚本路径，迁移后不可用。
现已修复：

- `package/wuji_hand/src/tianji_robotics/simulation/local_angle_bar.py` 的
  `_WUJIHAND_PYTHON` 改为 repo 根目录 `.venv-hardware/bin/python`；
  `_SEND_POSE_SCRIPT` 改为包内 `wuji_hand/tools/send_pose_to_wuji_hand.py`。
- `send_pose_to_wuji_hand.py` 默认序列号更新为当前设备 `344D345D3533`
  （旧默认 `365939643134` 不再匹配）。
- 现场已验证：拖滑条 → 点 Send to Hand → 确认弹窗 → 真手缓入移动 → 自动去使能。

使用前核对当前设备身份（`wujihandpy.Hand` 只读查询），并保持物理急停可达。详细步骤见
[Wuji Hand 真机操作](../../../tutorials/hardware/wuji_hand.md)。