# 旧真机文档来源映射

本表用于确认 `tianji_robotic_project` 的有效真机 Markdown 已被吸收，而不是按文件数量盲目
复制。旧命令中的 `real_robot_debug/`、`.venv-wujihand`、`recordings/` 和个人绝对路径均已
转换为当前模块、`.venv-hardware` 与 `local/data/`。

| 旧文档 | 当前去向 | 处理 |
|---|---|---|
| `real_robot_debug/README.md` | `tutorials/hardware/tianji_chopping.md`、`tianji_teleoperation.md` | 命令迁移并按风险拆分 |
| `doc_zt/tianji_wuji_operation_guide.md` | `tutorials/hardware/README.md`、`wuji_hand.md`、`combined_playback.md` | 操作顺序、设备信息、安全门合并 |
| `doc_zt/wuji_sdk_playback_guide.md` | `tutorials/hardware/wuji_hand.md`、`tianji_trajectory_playback.md` | 手与双臂命令拆分 |
| `doc_zt/dual_arm_hand_batch_stream_readme.md` | `tutorials/hardware/combined_playback.md` | batch、arm-only、hand-only、probe 保留 |
| `doc_zt/hand_only_stream_readme.md` | `tutorials/hardware/experiment_inventory.md`、`wuji_hand.md` | dry-run/execute/保持使能语义保留 |
| `doc_zt/half_fist_wuji_hand.md` | `tutorials/hardware/wuji_hand.md` | 保持使能和恢复步骤保留 |
| `doc_zt/reset_wuji_hand_fast.md` | `tutorials/hardware/wuji_hand.md` | 快速恢复及局限保留 |
| `doc_zt/2026-08-05-agent-handoff.md` | 本领域 `current.md` 与三篇纪传体 | 当前入口取代旧路径；设备历史进入编年体 |
| `doc_zt/2026-08-05-sdk-control-summary.md` | `evolution/control-modes.md`、`chronicles/2026-08-05-sdk-control.md` | SDK 发现和失败过程提炼 |
| `doc_zt/2026-08-16-tianji-dual-arm-handoff.md` | `evolution/trajectory-playback.md`、`chronicles/2026-08-16-dual-arm-debugging.md` | 双臂批流和故障边界保留 |
| `doc_zt/2026-08-23-simstyle-gamepad-jog.md` | `teleoperation/evolution/tianji-jogging.md`、真机点动教程 | 轴映射、FK/IK 和 deadman 保留 |
| `doc_zt/2026-08-26-wuji-glove-network-setup.md` | `evolution/device-commissioning.md`、手套网络教程 | 历史设备身份和路由问题保留 |
| `doc_zt/2026-08-06-mujoco-right-hand-retarget-progress.md` | `retargeting/evolution/right-hand-retargeting.md` | 错误数值镜像和真机候选边界保留 |
| `docs/wuji/hardware_interfaces.md` | `interfaces/wuji-hand.md`、`evolution/wuji-hand-tools.md` | 生命周期和 preflight 边界保留 |
| `README_wuji_angle_bar.md`、`docs/simulation/wuji_local_angle_bar.md` | `teleoperation/evolution/wuji-interactive-control.md` | 仿真功能保留；旧真机子进程路径标为待修复 |

以下材料不作为“当前真机教程”迁移：`docs/superpowers/plans/`、`specs/`、原始聊天、ROS 2
文档、旧归档源码说明和厂商 SDK README。其已经落实到当前代码的长期约束，应由纪传体说明；
未落实或已被替代的方案不继续冒充当前事实。

## 真机相关旧计划/设计的吸收情况

这些文件不原样保留实施步骤，但其有效结论已经归入当前纪传体：

| 旧计划/设计主题 | 当前去向 |
|---|---|
| `real-a-arm-ik-cart-impedance`、`real-a-pln-cart-position-chop` | `evolution/control-modes.md`、真机切菜教程 |
| `a-arm-impedance-two-stage-player`、`left-arm-trajectory-entry` | `evolution/trajectory-playback.md`、轨迹播放教程 |
| `recorded-chop-200hz-trajectory-export` | `evolution/trajectory-playback.md`；当前只保留格式与消费端，导出入口未迁入 |
| `mirrored-200hz-trajectory-export`、`mirrored-recorded-hand-playback` | `retargeting/evolution/right-hand-retargeting.md`；错误数值镜像已废弃 |
| `official-right-retarget-combined-export` | `retargeting/evolution/right-hand-retargeting.md`、联合回放教程 |
| `wuji-right-index-finger-test` | 真机实验总表、`evolution/wuji-hand-tools.md` |
| `keyboard-cartesian-jog` | `teleoperation/evolution/tianji-jogging.md`、真机点动教程 |
| `tianji-wuji-integration-refactor` | `evolution/dual-arm-hand-workflows.md`、接口文档 |

`2026-08-02-ubuntu24-jazzy-wuji-setup-summary.md` 主要描述 ROS 2/Jazzy 与仿真环境，本阶段
ROS 2 明确暂缓；其中非 ROS 的 Python 3.12 和环境重建原则已进入环境教程。
