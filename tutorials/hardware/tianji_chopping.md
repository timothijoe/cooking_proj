# Tianji 真机切菜与位置实验

本文迁移自旧项目 `real_robot_debug/README.md`。当前模块位于
`tianji_arm.experiments`。以下带 `--execute` 的命令会驱动实体机械臂；不带 `--execute`
的三个真机切菜入口仍会连接控制器、读取反馈，并可能配置控制模式，只是不发送切菜轨迹。
完全离线检查应使用 `plan_sampled_ik_chop`。示例姿态、方向和距离必须替换为现场确认值。

## 离线 sampled-IK 规划

该入口不连接机器人，只做 FK、笛卡尔采样、IK 和 CSV 输出：

```bash
.venv-hardware/bin/python -m tianji_arm.experiments.plan_sampled_ik_chop \
  --arm A \
  --init-joints "112.46,-51.20,-85.44,-72.70,47.48,-12.84,35.14" \
  --control-hz 250 \
  --dz-mm -40 \
  --hold-s 2.0 \
  --cycles 2 \
  --lateral \
  --lateral-mm 10 \
  --chop-axis y \
  --lateral-axis z \
  --lateral-phase separate \
  --stride 25 \
  --output-csv local/data/recordings/planned_chop.csv
```

## Sampled joint-impedance 切菜

旧项目的推荐实验是 SDK A 臂、真实坐标 `+Y` 下切、`Z` 方向换位。先做不发送轨迹的
连接检查（注意：它不是完全离线）：

```bash
.venv-hardware/bin/python -m tianji_arm.experiments.real_sampled_joint_impedance_chop \
  --robot-ip 192.168.1.190 \
  --arm A \
  --init-joints "112.46,-51.20,-85.44,-72.70,47.48,-12.84,35.14" \
  --control-hz 250 --dz-mm -40 --hold-s 2.0 --cycles 2 \
  --lateral --lateral-mm 10 \
  --chop-axis y --lateral-axis z --lateral-phase separate \
  --joint-k "8,8,8,4,2,1.5,1" \
  --joint-d "0.8,0.8,0.8,0.6,0.4,0.3,0.2" \
  --trace-csv local/data/recordings/joint_impedance_dry_run.csv
```

确认规划和现场后，真机命令是在相同参数中加入 `--execute`：

```bash
.venv-hardware/bin/python -m tianji_arm.experiments.real_sampled_joint_impedance_chop \
  --robot-ip 192.168.1.190 \
  --arm A \
  --init-joints "112.46,-51.20,-85.44,-72.70,47.48,-12.84,35.14" \
  --execute \
  --control-hz 250 --dz-mm -40 --hold-s 2.0 --cycles 2 \
  --lateral --lateral-mm 10 \
  --chop-axis y --lateral-axis z --lateral-phase separate \
  --joint-k "8,8,8,4,2,1.5,1" \
  --joint-d "0.8,0.8,0.8,0.6,0.4,0.3,0.2" \
  --print-feedback --feedback-stride 25 \
  --print-force-feedback --force-feedback-stride 25 \
  --trace-csv local/data/recordings/joint_impedance_trace.csv
```

CSV 包含目标/实际笛卡尔位姿、关节位置、速度、力矩和六轴力/力矩反馈。默认
`lateral-phase=separate` 表示先回到上位再横移；只有明确需要时才改为 `retract`。

## Sampled position 切菜

该入口使用相同采样笛卡尔目标和 IK，但发送位置模式关节目标：

```bash
.venv-hardware/bin/python -m tianji_arm.experiments.real_sampled_position_chop \
  --robot-ip 192.168.1.190 --arm A \
  --init-joints "109.81,-62.66,-95.69,-93.79,63.32,-2.76,12.42" \
  --execute \
  --control-hz 250 --dz-mm -20 --hold-s 2.0 --cycles 2 \
  --lateral --lateral-mm 10 \
  --chop-axis y --lateral-axis x --lateral-phase separate \
  --print-feedback --feedback-stride 25 \
  --print-force-feedback --force-feedback-stride 25 \
  --trace-csv local/data/recordings/sampled_position_trace.csv
```

## Planned Cartesian MOVLA 切菜

该入口使用较粗的 MOVLA 规划段，不是逐控制周期 sampled IK：

```bash
.venv-hardware/bin/python -m tianji_arm.experiments.real_pln_cart_position_chop \
  --robot-ip 192.168.1.190 --arm A --execute \
  --control-hz 250 --dz-mm -20 --hold-s 2.0 --cycles 5 \
  --lateral --lateral-mm 10 \
  --chop-axis y --lateral-axis x \
  --trace-csv local/data/recordings/pln_cart_trace.csv
```

## 单一关节目标位置

先读帮助；`--joints` 是七个目标角度，`--execute` 会真实发送：

```bash
.venv-hardware/bin/python -m tianji_arm.experiments.real_position_mode_a_arm --help

.venv-hardware/bin/python -m tianji_arm.experiments.real_position_mode_a_arm \
  --arm A --robot-ip 192.168.1.190 \
  --joints "J1,J2,J3,J4,J5,J6,J7" \
  --execute
```

`J1...J7` 必须替换为当前机器人经过验证的目标，不能原样复制。程序限制单次最大关节差，
但软件限位不能证明路径无碰撞。

## 公共约束

- SDK A 臂是旧设备上的左臂，B 臂是右臂；新设备必须重新确认。
- 旧设备记录中 `+Y` 是下切方向；相机画面方向不能替代机器人坐标确认。
- 程序拒绝垂直动作超过 80 mm、横移超过 50 mm，但仍需现场碰撞评审。
- 默认不加 `--keep-enabled`；不要为了方便长期保持使能。
- 非预期运动立即按物理急停。`Ctrl+C` 只触发软件清理。
