# 独立倒手 PPO 残差基线

2026-09-18。在 [参考环境](regrasp-microskill-reference.md) 上建立可训练 Gym 环境，真实运行 PPO 更新网络参数。它是保留参考阶段与 IK 的残差控制，不是从零学习关节轨迹，也没有加载旧切菜 PPO。

## 入口

```bash
OMP_NUM_THREADS=1 .venv/bin/python scripts/simulation/train_regrasp_rl.py \
  --steps 16384 --output local/outputs/potato/regrasp_rl_v1

# 已训练权重单独评估，不继续训练
OMP_NUM_THREADS=1 .venv/bin/python scripts/simulation/train_regrasp_rl.py \
  --load local/outputs/potato/regrasp_rl_v1/policy.zip --steps 0 \
  --output local/outputs/potato/regrasp_rl_v1/eval

# 明确标注 TRAINED PPO RESIDUAL 的实时窗口，自动完成三次后保持
DISPLAY=:0 OMP_NUM_THREADS=1 .venv/bin/python scripts/simulation/run_regrasp_rl.py \
  --policy local/outputs/potato/regrasp_rl_v1/policy.zip --view
```

运行器可以改用 `MUJOCO_GL=egl` 与 `--video` 导出视频。不要把旧 `run_dynamic_regrasp.py` 的切菜窗口、`run_regrasp_microskill.py` 的参考窗口和本 PPO 窗口混淆。

## Observation：354 维

每帧 118 维，拼接最近三帧（20 ms 间隔）：

- 左臂 7 + 手 20 个关节的位置、速度、目标误差：81 维。
- 三根主指总法向载荷：3 维。当前为理想触觉代理，不是电流。
- 三指 12 个关节执行器力矩/对应力矩限制：12 维。没有冒称 Wuji 实测电流。
- 腕部相对桌面的三维位置：3 维。
- 内部控制阶段 one-hot：9 维。
- 当前阶段经过时间、已完成请求数：2 维。
- 上次原始策略动作：8 维。

没有点云、土豆尺寸、精确表面、土豆位姿或速度直接输入 actor。critic 与 actor 暂时共用这些输入，没有实现特权 critic。物体真值用于奖励和仿真成功/终止判断；实机不能照搬此验收器。接触和力矩反馈尚未建模测量噪声、延迟、滤波或硬件差异。

## Action：8 维

PPO 输出限制在 [-1, 1]，经 `a_filtered += 0.2 * (a - a_filtered)` 平滑：

| 维度 | 控制内容 | 范围 |
|---|---|---|
| 0–2 | 三指独立载荷目标 | 0.4–0.9 N |
| 3 | 下降/重新接触调节速度系数 | 0.7–1.3 |
| 4 | 支撑退腕参考进度速度 | 0.75–1.25 倍 |
| 5 | 移指参考进度速度 | 0.75–1.25 倍 |
| 6 | 指尖腾空曲线高度 | 10–14 mm |
| 7 | DIP 额外屈曲幅度 | 20–28° |

移指高度/DIP 幅度在进入移指时锁定，保证全段 IK 预检与执行使用同一几何曲线；不在腾空中突然换轨迹。物理步长固定 2 ms、控制周期 20 ms，速度动作只改变参考进度。动作全零对应参考参数。不是直接电流、力矩控制；载荷调节最终通过有限力位置驱动执行。

## Reward 与 episode

从悬空开始，一个 episode 连续执行三次倒手，期间不重置物体。每回合重新采样 ±5% 三轴尺寸和不对称形变种子；质量随体积缩放，底部削平。

每步奖励：

- 时间代价 -0.01；原始动作变化平方和乘 -0.03。
- 相比上一步增加的土豆位移（m）乘 -100。
- 当前位移/5 mm 乘 -0.05；线速度/20 mm/s 乘 -0.05。
- 每指超过 2 N 的载荷合计乘 -0.02。
- 支撑/落指阶段适度接触奖励，最高 +0.015；腾空时不要求接触。
- 初次建立支撑 +2；每次实际落点及稳定性验收完成 +10。
- 故障或 episode 超时 -10。

达到三次成功或物理/IK 故障终止；1400 控制步上限截断。成功依据仍是参考环境的实际相对落点、物体位移和接触检查，不能只播放完轨迹就算成功。

## 训练和对照

CPU PPO，网络 [64,64]，rollout 512，batch 128，学习率 3e-4，gamma .995，初始 log_std=-1.5，随机种子 0。首次预算 16384 个环境步。输出 `policy.zip`、`training.monitor.csv`、`training.json`；随后固定评估种子 100/101/102/103，分别运行零动作参考和确定性 PPO，保存 `evaluation.json`。

评估同样完成三次请求。报告成功次数、最大物体位移、耗时、动作幅值与终止原因。训练日志成功率不是独立评估结果；样本很小，不能声称泛化或优于参考，必须看实际对照结果。

## 当前限制

这轮学习的是压力和参考运动参数，仍保留阶段机、载荷阈值、固定方向、固定腕部朝向及参考轨迹结构；何时换落点还受参考阶段约束，没有成为完全自由的策略决策。尚未实现真实遥控器调用 PPO、effort 标定、触觉随机化、特权 critic 或 sim2real。后续需要逐项放开动作能力，并检验策略是否真正依赖接触反馈。

## 首轮实际结果

已完成 16,384 步训练，训练耗时 154.3 s；权重位于 `local/outputs/potato/regrasp_rl_v1/policy.zip`。8 项参考/Gym 测试通过（29.73 s）。评估结果保存到同目录 `evaluation.json`，PPO 实际运行视频及逐步动作见 `rollout/ppo.mp4`、`rollout/rollout.json`。这些大文件位于忽略的 local 目录，不会随源码自动提交。

| 评估项（种子 100–103，每回合三次倒手） | 零动作参考 | 确定性 PPO |
|---|---:|---:|
| 三次全部完成的回合 | 4/4 | 4/4 |
| 平均每回合最大物体位移 | 1.2146 mm | 1.1201 mm |
| 平均执行时间（不含 reset 稳定） | 14.405 s | 14.365 s |
| 平均 episode reward | 21.9007 | 22.1838 |

逐形状最大位移（参考/PPO，mm）：100 为 1.2796/1.3980，101 为 1.0492/1.1088，102 为 1.1125/0.9889，103 为 1.4170/0.9845。PPO 在两个形状上略差，两个形状上更好；总体差异小，不能据此声称稳定提升或 sim2real 成功。种子 100 上 8 个动作的平均绝对值约 0.039–0.084，策略确实输出非零调整，但幅度较小，仍主要依赖参考动作结构。
