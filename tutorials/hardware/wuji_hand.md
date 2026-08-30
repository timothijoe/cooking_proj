# Wuji Hand 真机操作

本文迁移自旧项目 Wuji SDK 操作手册。旧设备记录为左手、USB 序列号 `365939643134`、
产品序列号 `LQSQJR.260616.005`；运行前必须读取当前设备身份，不能假设仍是这只手。

重要：除 `hand_only_stream --execute` 和 `wuji-index-mcp-test --execute` 外，多个 Wuji 工具
没有 `--execute` 安全门；直接运行 `wuji-reset`、`wuji-half-fist`、`wuji-send-pose` 或
`wuji-play-trajectory` 就会连接、使能并发送目标。

## 设备和只读状态

```bash
ls -l /dev/ttyACM0

.venv-hardware/bin/python -c "
from wujihandpy import Hand
import numpy as np
hand = Hand(serial_number='365939643134')
print('SN:', hand.get_product_sn())
print('FW:', hand.get_firmware_version())
print('Handedness:', hand.get_handedness())
print('Errors:', np.asarray(hand.read_joint_error_code()))
print('Temp max:', np.asarray(hand.read_joint_temperature(), dtype=float).max())
print('Positions:', np.asarray(hand.read_joint_actual_position(), dtype=float))
"
```

该命令会连接设备并读取状态，但不使能、不发送目标。错误码非零或温度异常时不要执行动作。

## 轨迹 preflight

```bash
.venv-hardware/bin/tianji-robot hardware wuji-sdk preflight \
  local/data/recordings/trajectory.npz
```

preflight 只验证文件，不连接实体手。

## 复位到张开

以下命令会立即连接和驱动实体手：

```bash
# 1 秒缓入
.venv-hardware/bin/wuji-reset --serial-number 365939643134 --ramp 1.0

# 0.8 秒快速恢复
.venv-hardware/bin/wuji-reset-fast --serial-number 365939643134 --ramp 0.8
```

快速恢复会从当前姿态插值到零位，结束后去使能。它不是物理急停；若手指已经机械卡死，
继续发送复位目标可能增加载荷，应先切断运动并人工检查。

## 半握拳

```bash
# 半握拳后自动恢复张开
.venv-hardware/bin/wuji-half-fist --serial-number 365939643134 --ramp 3 --hold 5

# 半握拳并保持；这是高风险入口
.venv-hardware/bin/wuji-half-fist-hold --serial-number 365939643134 --ramp 3 --hold 10
```

保持版本可能在程序退出后仍保持使能。观察完立即用 `wuji-reset-fast` 恢复，并确认实际状态。

## 播放 NPZ 轨迹

先做 preflight，再从 0.1 倍速和较长缓入开始：

```bash
.venv-hardware/bin/wuji-play-trajectory \
  local/data/recordings/trajectory.npz \
  --serial-number 365939643134 \
  --speed 0.1 --ramp 5 --hold 2
```

该工具没有 `--execute`；运行命令本身就会使能并播放。它检查错误码和关节限位、缓入首帧、
播放并在 `finally` 中去使能。`Ctrl+C` 后手可能停在中间姿态。

## 单帧姿态

```bash
.venv-hardware/bin/wuji-send-pose \
  local/data/recordings/pose.npz \
  --serial-number 365939643134 --ramp 3 --hold 2
```

该工具同样没有 `--execute`；NPZ 必须包含它要求的 `joint_positions_rad` 字段。

可以用 `.venv/bin/tianji-robot sim wuji-angle-bar` 在 MuJoCo 中制作并保存姿态，但当前面板
的 **Send to Hand** 按钮仍引用旧环境和脚本路径，不要点击它控制真机。保存 NPZ 后退出
面板，再使用上面的 `wuji-send-pose` 命令，并重新核对手性和序列号。

右手轨迹不得从左手 20 关节输出做简单符号镜像。旧 `UNUSABLE_numeric_mirror` 数据禁止
上机；`official_right_retarget_candidate` 也必须先完成单关节索引、正方向、零位和限位验证。

## 常见问题

- `ERROR_BUSY`：另一个 SDK 或 ROS 进程占用同一 USB。先识别并正常停止占用者，不要盲目
  `pkill` 其他用户进程。
- 错误码非零：记录错误码并按设备手册排查；不要用反复 reset 掩盖机械卡碰。
- SDK 与 ROS 不能同时拥有设备。本项目 ROS 2 当前暂缓。
- 温度过高：停止使用并等待冷却，不能靠降低速度继续硬撑。
