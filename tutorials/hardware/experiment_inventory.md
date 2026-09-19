# 真机实验入口总表

本表覆盖当前仓库迁移进来的全部真机相关 Python 入口。`未单列执行教程` 表示保留代码用于
兼容或开发对照，不表示可以直接上机。

## Tianji Arm

| 模块（`tianji_arm.experiments.` 后缀） | 类型 | 运行行为 | 教程 |
|---|---|---|---|
| `arm_command_diagnostic` | A 臂诊断/≤1°试动 | 默认只读连接；`--execute` 运动 | [诊断](../tianji_hardware.md) |
| `plan_sampled_ik_chop` | 轨迹规划 | 离线，不连接 | [切菜](tianji_chopping.md) |
| `shift_right_arm_trajectory` | 右臂轨迹平移 | 离线 FK/IK，不连接 | 下方命令 |
| `real_position_mode_a_arm` | 单目标位置 | 默认连接读反馈；`--execute` 运动 | [切菜](tianji_chopping.md) |
| `real_sampled_joint_impedance_chop` | sampled 阻抗切菜 | 默认连接/配置；`--execute` 运动 | [切菜](tianji_chopping.md) |
| `real_sampled_position_chop` | sampled 位置切菜 | 默认连接/配置；`--execute` 运动 | [切菜](tianji_chopping.md) |
| `real_pln_cart_position_chop` | MOVLA 切菜 | 默认连接/配置；`--execute` 运动 | [切菜](tianji_chopping.md) |
| `real_ik_cart_impedance_lateral` | 上述切菜共享实现 | `--execute` 运动 | 不直接使用，选择三个包装入口 |
| `keyboard_cartesian_jog` | 键盘点动 | 默认连接 dry-run；`--execute` 运动 | [点动](tianji_teleoperation.md) |
| `gamepad_cartesian_jog` | 手柄 MOVLA 点动 | 默认连接 dry-run；`--execute` 运动 | [点动](tianji_teleoperation.md) |
| `gamepad_cartesian_jog_simstyle` | 连续 FK/IK 点动 | 默认连接 dry-run；`--execute` 运动 | [点动](tianji_teleoperation.md) |
| `a_arm_impedance_two_stage` | A 臂 NPZ 播放 | 默认离线；`--execute` 运动 | [轨迹播放](tianji_trajectory_playback.md) |
| `arm_impedance_playback` | A/B/AB NPZ 播放 | 默认离线；`--execute` 运动 | [轨迹播放](tianji_trajectory_playback.md) |
| `dual_arm_batch_stream` | 双臂批流 | 默认离线；`--execute` 发送 | [轨迹播放](tianji_trajectory_playback.md) |
| `bak` | 历史通用控制器 | `--execute` 运动 | 未单列执行教程，保留作对照 |

离线平移右臂轨迹：

```bash
.venv-hardware/bin/python -m tianji_arm.experiments.shift_right_arm_trajectory \
  --source-npz local/data/recordings/source.npz \
  --output-npz local/data/recordings/shifted.npz \
  --shift-x-mm 30 --shift-y-mm 0 --shift-z-mm 100
```

该工具不连接机器人，但生成的新轨迹仍必须重新 dry-run 和现场审核。

## Wuji Hand

| 安装后命令/模块 | 行为 | 安全门 | 教程 |
|---|---|---|---|
| `tianji-robot hardware wuji-sdk preflight` | 文件检查，不连接 | 无需执行门 | [Wuji](wuji_hand.md) |
| `wuji_hand.tools.hand_only_stream` | NPZ 手轨迹 | 默认离线，`--execute` 后连接使能 | 下方命令 |
| `wuji-index-mcp-test`（`test_wuji_right_index_mcp`） | 食指 MCP 小动作 | 会连接做 preflight，`--execute` 后运动 | 下方命令 |
| `wuji-reset`（`reset_wuji_hand`） | 恢复张开 | 运行即连接和运动 | [Wuji](wuji_hand.md) |
| `wuji-reset-fast`（`reset_wuji_hand_fast`） | 快速恢复张开 | 运行即连接和运动 | [Wuji](wuji_hand.md) |
| `wuji-half-fist`（`half_fist_wuji_hand`） | 半握拳后恢复 | 运行即连接和运动 | [Wuji](wuji_hand.md) |
| `wuji-half-fist-hold`（`half_fist_wuji_hand_hold`） | 半握拳并保持使能 | 运行即连接和运动 | [Wuji](wuji_hand.md) |
| `wuji-send-pose`（`send_pose_to_wuji_hand`） | 单帧姿态 | 运行即连接和运动 | [Wuji](wuji_hand.md) |
| `wuji-play-trajectory`（`play_wuji_trajectory`） | NPZ 轨迹 | 运行即连接和运动 | [Wuji](wuji_hand.md) |

推荐的手部轨迹入口有显式执行门：

```bash
# dry-run，不连接
.venv-hardware/bin/python -m wuji_hand.tools.hand_only_stream \
  --source-npz local/data/recordings/trajectory.npz --speed-scale 0.1

# 真机
.venv-hardware/bin/python -m wuji_hand.tools.hand_only_stream \
  --source-npz local/data/recordings/trajectory.npz \
  --wuji-serial 344D345D3533 \
  --speed-scale 0.1 --entry-duration-s 5 --execute
```

该入口播放完成或 `Ctrl+C` 后会保持使能和当前姿态，必须随后运行复位并确认设备状态。

右手食指 MCP 测试：

```bash
# 会连接读取设备和生成计划，但不运动
.venv-hardware/bin/wuji-index-mcp-test --serial-number <RIGHT_HAND_SERIAL>

# 真机运动
.venv-hardware/bin/wuji-index-mcp-test \
  --serial-number <RIGHT_HAND_SERIAL> --execute --dwell-s 0.5
```

序列号占位符必须替换为现场确认的右手设备；不能使用左手旧序列号。

## 联合工作流

| 模块（`cooking_workflows.` 后缀） | 行为 | 教程 |
|---|---|---|
| `dual_arm_hand_batch_stream` | 默认离线；`--execute` 控制双臂/手 | [联合回放](combined_playback.md) |
| `dual_arm_hand_playback` | 旧 Concise API；默认离线；`--execute` 控制设备 | [联合回放](combined_playback.md) |

## 传感器

| 脚本 | 行为 | 教程 |
|---|---|---|
| `scripts/hardware/sensor_uart_stream.py` | 默认 dry-run；`--execute` 打开 `/dev/ttyUSB0`、发送采集指令并显示二维压力热力图 | [UART 压力传感器](sensor_uart_stream.md) |

当前现场观察可使用下列命令；`--heatmap-vmax 2` 是临时显示上限，并非量程：

```bash
.venv-hardware/bin/python scripts/hardware/sensor_uart_stream.py \
  --execute --port /dev/ttyUSB0 --baudrate 115200 \
  --heatmap-vmin 0 --heatmap-vmax 2
```

在运行该命令的终端按 `Ctrl+C` 停止采集。已知传感器最大量程后，应将 2 替换为量程值，
以保证不同时刻的颜色可以直接比较。

## 不应直接执行的文件

- `__init__.py` 不是实验入口。
- `real_ik_cart_impedance_lateral` 是多个切菜包装器的共享实现，人工操作应选择语义明确的
  wrapper。
- `bak` 是历史兼容实验，没有独立上机教程；除非重新评审，否则只查看 `--help` 和源码。
