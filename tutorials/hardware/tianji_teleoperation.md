# Tianji 真机键盘与手柄点动

本文迁移自旧项目真机 README。三个入口默认 dry-run，但仍会连接控制器读取启动反馈；
加入 `--execute` 后会驱动实体机械臂。

## 键盘单步

```bash
# dry-run
./scripts/teleop/tianji/keyboard_jog.sh --arm A

# 真机：以启动 TCP 为中心建立 ±100 mm 工作空间
./scripts/teleop/tianji/keyboard_jog.sh \
  --arm A --execute --step-mm 5 \
  --workspace-around-current-mm 100
```

W/S、A/D、R/F 请求 ±X、±Y、±Z；Q 或空格退出并尝试 disable。每次按键只请求一个小步，
按住按键不应产生连续运动。也可以用经过现场确认的 `--workspace-min` 与 `--workspace-max`
代替启动点边界。

## 手柄离散 MOVLA 点动

```bash
# dry-run
./scripts/teleop/tianji/gamepad_jog.sh --arm right

# 真机首测
./scripts/teleop/tianji/gamepad_jog.sh \
  --arm right --execute \
  --speed-mm-s 10 \
  --workspace-around-current-mm 50
```

`right` 对应旧设备 SDK B 臂，`left` 对应 A 臂。RB 是 deadman，释放 RB 后不再生成新请求；
Start 退出。左摇杆控制平移，具体轴映射运行前用 dry-run 核对。

## 仿真风格连续 FK/IK 点动

```bash
# dry-run
./scripts/teleop/tianji/gamepad_simstyle_jog.sh --arm right

# 真机首测
./scripts/teleop/tianji/gamepad_simstyle_jog.sh \
  --arm right --execute \
  --speed-mm-s 20 \
  --orientation-rate-dps 20 \
  --workspace-around-current-mm 50
```

该入口以固定频率积分摇杆速度、做 FK/IK 并发送关节目标。左摇杆负责基坐标平移；右摇杆和
D-pad 可同时控制工具坐标 yaw/pitch/roll。RB 仍是 deadman，Start 退出。

软件会检查反馈、错误、工作空间、规划或 IK，并限制速度；这些保护不建立真实碰撞净空。
手柄断连、方向错误、IK 连续失败或任何意外运动都应立即释放 RB，并按物理急停。
