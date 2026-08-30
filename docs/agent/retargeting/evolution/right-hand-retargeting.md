# 右手 Retargeting 与真机输入边界

右手轨迹必须从原始右手骨架调用官方 `Handedness.Right` retargeter 生成。不能把已经经过
左手 retargeter 的 20 关节输出，通过逐关节复制或符号翻转“镜像”为右手；旧项目 Viewer
已经证明这种数值镜像动作严重错误。

旧数据中名称包含 `UNUSABLE_numeric_mirror` 的文件禁止作为真机输入。名称包含
`official_right_retarget_candidate` 的文件虽来自正确数据链，也仍只是离线候选：MuJoCo
只能验证模型范围和视觉方向，不能证明真机 SDK 索引、零位、电机方向、耦合、固件限位或
安装碰撞安全。

右手上机顺序必须是：只读识别设备和手性 → 单关节索引/正方向低速验证 → 零位与限位验证
→ hand-only 低速短轨迹 → 联合系统。任何 candidate 文件不得绕过该顺序直接批量播放。
