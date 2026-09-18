## 当前版本：连续 3 轮切菜倒手

任务改为同一动力学回合内连续完成 3 轮：刀做两次浅下移，关节后退，刀退开，三指同步抬起/后移 8 mm/落下，重新按稳，再进行下一轮。三轮累计指尖后移 24 mm。整个序列约 29.7 秒，轮间不重置土豆的位置、速度、接触状态或累计位移。

轮间安排 1.2 秒手腕跟进和 1.2 秒刀靠近：手腕跟进约 8 mm；为保持关节可达，第二轮起初始掌高增加 1 mm。每轮倒手主体仍保持手臂固定。关节目标平滑连接；实际运动由有限力驱动器执行。仍使用 3 mm 平底、无外部测试推力、拇指/小指全程移开，刀不切开土豆。

RL 仍控制三个竖直压紧残差和一个公共阶段速度，动作顺序与刀轨迹由参考规划定义。奖励包括累计位移、线速度、转角、指尖滑移、水平合力、过大接触力和动作突变的惩罚，以及阶段进度、应支撑阶段的接触和每轮验证奖励。

成功要求完成全部三轮且全过程累计位移小于 5 mm；每轮结束三根指尖法向载荷均大于 0.1 N、土豆线速度小于 0.02 m/s。检查结果保存于 `cycle_checks`，不以播放完动画代替重新按稳。刀手过大接触力、异常穿透、物体远离或数值异常仍会终止回合。

输出：`local/outputs/potato/dynamic_regrasp_three_cycles/`。单轮源码存档：`local/checkpoints/20260918-before-three-cycles/source.tar.gz`。窗口显示轮次和当前阶段，循环播放时仅在三轮全部结束后开启新的回放。

```bash
OMP_NUM_THREADS=1 .venv/bin/python scripts/simulation/run_dynamic_regrasp.py --cycles 3 --train 16384
MUJOCO_GL=egl .venv/bin/python scripts/simulation/run_dynamic_regrasp.py --cycles 3 --policy local/outputs/potato/dynamic_regrasp_three_cycles/policy.zip --export
.venv/bin/python scripts/simulation/run_dynamic_regrasp.py --cycles 3 --policy local/outputs/potato/dynamic_regrasp_three_cycles/policy.zip --view
```

本轮 PPO 训练 16,384 步后，三轮均通过重新按稳检查，全程最大位移约 3.70 mm；零残差参考约 4.15 mm。35 项相关测试通过。该结果只验证当前确定性场景，不代表泛化鲁棒性。

当前只支持 1–3 轮，以保持现有土豆长度和关节范围内的可达性。无随机化条件下评估只运行一个固定种子；当前结果不能当作跨形状或跨摩擦泛化结论。

---

## 历史版本：接触点上移，压紧方向竖直向下

为减少三指在土豆前侧斜面上向后推物体，初始三指接触点沿 +Y 移动 24 mm，靠近土豆顶部；初始手掌同步平移 24 mm 以保持关节可达，倒手过程中手臂仍固定。仅移动指尖而保持原初始手掌位置会超出当前关节约束，因此没有强行穿透或放宽关节限制。

RL 的 3 个压紧残差改为沿世界 -Z（竖直向下）的指尖雅可比映射，替代原来指向土豆中心的方向；新增水平合力惩罚。该映射是有界位置控制，实际接触力方向仍由接触几何和动力学决定，不宣称精确力控制。回放增加向后手指合力（世界 +Y）的数值显示。

保持 3 mm 平底、无测试外力、拇指/小指移开、三指同时抬起/后移/落下。新策略重新训练，输出存于 `local/outputs/potato/dynamic_regrasp_top_contact/`，修改前源码在 `local/checkpoints/20260918-before-contact-adjustment/source.tar.gz`。`contact_comparison.json` 用相同零残差动作比较接触位置，不混入旧/新策略的差别。

```bash
OMP_NUM_THREADS=1 .venv/bin/python scripts/simulation/run_dynamic_regrasp.py --train 8192
MUJOCO_GL=egl .venv/bin/python scripts/simulation/run_dynamic_regrasp.py --policy local/outputs/potato/dynamic_regrasp_top_contact/policy.zip --export
.venv/bin/python scripts/simulation/run_dynamic_regrasp.py --policy local/outputs/potato/dynamic_regrasp_top_contact/policy.zip --view
```

本次 8,192 步 PPO 的最大位移为 3.45 mm，达到当前场景下全过程小于 5 mm 的阈值。零残差对照为 3.59 mm，说明本轮改善主要来自接触位置调整。相关 32 项测试通过。当前没有形状、摩擦或状态随机化；无外力下不同评估种子产生相同环境，不能当作三个独立泛化测试。

---

## 历史版本：关闭测试外力

按用户要求，`DynamicRegraspEnv` 默认 `disturbance=False`，动态任务命令行也显式关闭外力。训练、评估和回放不再向土豆施加测试推力。平底、三指同步动作及拇指/小指移开均保持不变。

当前回放沿用上一版策略，只关闭推力，未重新训练，便于观察这一项改动的影响。输出写入 `local/outputs/potato/dynamic_regrasp_no_push/`，旧的有扰动结果保留。报告记录 `external_disturbance=false`、原策略路径和实际施加冲量；窗口显示 `NO EXTERNAL PUSH`。未来直接运行 `--train` 也会在无外力条件下训练。底层显式开启扰动的能力仅保留用于单独测试。

```bash
MUJOCO_GL=egl .venv/bin/python scripts/simulation/run_dynamic_regrasp.py --policy local/outputs/potato/dynamic_regrasp_flat_bottom/policy.zip --export
.venv/bin/python scripts/simulation/run_dynamic_regrasp.py --policy local/outputs/potato/dynamic_regrasp_flat_bottom/policy.zip --view
```

---

## 历史版本：削去 3 mm 底部，平底朝下

用户新增假设：土豆底部已削去一小块。动态任务现使用 `PotatoConfig(bottom_cut_m=.003)`，沿原网格最低点以上 3 mm 的水平面截去底部，插值得到切面边界，由同一个凸网格承担渲染与碰撞。不是把完整土豆埋入桌面，也没有焊接或固定土豆。

切面约 40.2 × 65.4 mm，土豆总高约 57.1 mm。按砧板顶面 z=0.230 m 调整物体初始高度；三指接触轨迹在新表面上重新求解。拇指、小指继续全程移开，4 维 RL 控制保持不变。质量参数仍按剩余土豆 0.22 kg 处理，摩擦参数不变。历史几何预览和旧 RL 环境默认仍可生成圆底资产。

新输出目录：`local/outputs/potato/dynamic_regrasp_flat_bottom/`。修改前源码存于 `local/checkpoints/20260918-before-flat-bottom/source.tar.gz`。

```bash
OMP_NUM_THREADS=1 .venv/bin/python scripts/simulation/run_dynamic_regrasp.py --train 8192
MUJOCO_GL=egl .venv/bin/python scripts/simulation/run_dynamic_regrasp.py --policy local/outputs/potato/dynamic_regrasp_flat_bottom/policy.zip --export
.venv/bin/python scripts/simulation/run_dynamic_regrasp.py --policy local/outputs/potato/dynamic_regrasp_flat_bottom/policy.zip --view
```

验证包括：实际编译后的碰撞网格含水平底面、自由落体后直立停稳、切除深度合法性，以及整段倒手的动力学回归。本次重新训练 8,192 步后，种子 100/101/102 下 PPO 最大位移为 15.03/10.45/6.58 mm（同形状参考轨迹为 16.53/11.61/7.16 mm），均未达到 5 mm 目标；所有评估中拇指、小指和刀手接触力均为 0 N。平底不等于已经学会抗滑，详细结果见该目录的 `evaluation.json`。

---

## 历史版本：拇指、小指移开

按用户反馈，动态环境中的拇指和小指全程停留在初始脱离土豆的位置，取消辅助夹持、释放动作及对应的接触奖励。碰撞几何仍启用，并在每个物理步检查两指所有碰撞部位的接触力。三指同步倒手轨迹、阶段时长、刀轨迹和扰动条件保持原设置，便于比较。

RL 动作空间改为 4 维：食指、中指、无名指的压紧残差，加公共阶段速度。旧 6 维策略不能用于新版；重新训练的策略与输出保存在 `local/outputs/potato/dynamic_regrasp_helpers_parked/`，旧结果未覆盖。旧动态源文件另存于 `local/checkpoints/20260918-before-parking-helpers/dynamic-source.tar.gz`。

```bash
OMP_NUM_THREADS=1 .venv/bin/python scripts/simulation/run_dynamic_regrasp.py --train 8192
MUJOCO_GL=egl .venv/bin/python scripts/simulation/run_dynamic_regrasp.py --policy local/outputs/potato/dynamic_regrasp_helpers_parked/policy.zip --export
.venv/bin/python scripts/simulation/run_dynamic_regrasp.py --policy local/outputs/potato/dynamic_regrasp_helpers_parked/policy.zip --view
```

本次 8,192 步 PPO 后，三个评估种子 100/101/102 的最大位移分别为 20.74/24.36/17.65 mm。全部 6 次参考/PPO 回合中，两辅助指最大接触力均为 0 N；仍未达到 5 mm 稳定目标。

评估中的 `peak_helper_contact_force_n` 是整段物理步上两指碰撞部位的最大接触力；`helper_loads_n` 是各输出帧上拇指、小指的总载荷。下方记录为历史版本，旧的 6 维策略需配合存档源码使用。

---

## 2026-09-18：动力学与同步倒手 PPO 基线

修改前快照：`local/checkpoints/20260918-020945-before-dynamic-regrasp/`。包含源码（含未跟踪文件）、工作区补丁、原同步预览、旧 RL 结果和参考录制；机器人本地资产、虚拟环境仍为外部依赖。原几何预览脚本没有改动。

新增 `twin_sim.dynamic_regrasp.DynamicRegraspEnv` 与 `scripts/simulation/run_dynamic_regrasp.py`：

- 仅 reset 初始化 qpos，动作通过有限力位置驱动器和 `mj_step` 执行；土豆自由运动。
- 刀面参考间隙 3 mm，关节回退期间做两次浅下移，再连续抬起退开。刀刃始终在整颗土豆上方，不模拟实际切开，也没有宣称刀面已实现接触力控制。
- 6 维动作：5 指各自最多 3 mm 的压紧方向残差（每关节最多 0.12 rad），加一个三指共用的阶段速度。指尖残差是几何雅可比映射，不是精确力控制。
- 输入：关节位置/速度、80 维理想接触触觉、土豆尺寸/位姿/速度特权信息、阶段。当前只随机水平扰动方向，未随机形状、摩擦或传感器噪声。
- 奖励惩罚土豆移动、旋转、速度、指尖相对滑移、过大接触力和动作突变；鼓励相应阶段的支撑指接触。阶段推进奖励按实际参考进度计算，避免仅因放慢而累计更多阶段奖励。
- 三指后移阶段施加约 2 N 水平外力；按参考阶段触发后，在物理时钟中持续 80 ms，冲量固定为 0.16 N·s，与阶段速度无关。
- 界面显示 5 指与土豆的法向力、接触点切向相对速度、土豆位移/转角和刀手最近间隙。视频/窗口回放的是已执行动力学的状态记录，不是再次执行策略。
- 土豆表面距离查询出现过与包围盒和实际碰撞矛盾的负值，因此刀与土豆使用保守的世界 Z 包围盒间隙；刀手仍用几何最近距离，并逐物理步检查碰撞力。

本次 PPO 8,192 步只构成初步基线。保留种子 100/101/102 中，参考最大位移约 12.88/12.17/10.18 mm，PPO 约 11.20/10.31/8.51 mm；均未达到全过程位移小于 5 mm 的成功条件。不能据此声称学会稳定倒手或证明触觉有效。尚需触觉消融、形状/摩擦随机化和更充分训练。

```bash
# 训练与评估
OMP_NUM_THREADS=1 .venv/bin/python scripts/simulation/run_dynamic_regrasp.py --train 8192
# 使用已有策略导出物理回放视频与逐帧指标
MUJOCO_GL=egl .venv/bin/python scripts/simulation/run_dynamic_regrasp.py --policy local/outputs/potato/dynamic_regrasp/policy.zip --export
# 打开窗口（空格暂停）
.venv/bin/python scripts/simulation/run_dynamic_regrasp.py --policy local/outputs/potato/dynamic_regrasp/policy.zip --view
```

输出在 `local/outputs/potato/dynamic_regrasp/`：`policy.zip`、`training.json`、`evaluation.json`、`rollout.npz`、`rollout-metrics.json`、`dynamic_regrasp.mp4`。

---

# 土豆触觉退手实验

## 当前：三指同步倒手，拇指和小指辅助扶持

用户已明确三个手指同时移动。动作顺序为：食指、中指、无名指扣住土豆；先保持指尖接触，让 PIP（近端指间关节）后收；关节与指尖前后接近后，拇指和小指先接触土豆，再让三根主指同步抬起、后移和落下，最后松开辅助两指。手掌在周期中固定，三指之间保留间隙。

新窗口入口：

```bash
.venv/bin/python scripts/simulation/view_finger_regrasp.py
# 导出分阶段图片、视频和几何测量
MUJOCO_GL=egl .venv/bin/python scripts/simulation/view_finger_regrasp.py --export
```

空格暂停，左右键查看相邻时刻，1/2/3 切换侧面/俯视/斜视，R 重播。绿色为三根主指，金色为拇指/小指辅助接触，粉色为 PIP 朝向刀的表面参考位置；黄色标尺长 100 mm。刀面半透明仅用于观察，不改变碰撞体。

原场景测得土豆约 122×69×55 mm，手掌长宽约 105×81 mm、全手伸展长度约 206 mm；无单位缩放错误。新预览保持手模型不缩放，将土豆改为沿切片方向放置的标称长宽高 130×80×60 mm、质量 220 g。旧 RL 环境和旧策略仍使用旧配置，本轮没有把新姿态冒充已训练策略。

掌根相对上一版降低约 8 mm（相对原始录制初始摆位仍抬高 16 mm），周期中左臂关节完全不变。指尖固定时 PIP 领先量从约 7 mm 减至约 1.5 mm，规划采样点最大指尖漂移小于 0.001 mm。刀让开、辅助两指接触后，三指同步抬起 12 mm、后移 8 mm，再同时落到实际网格表面。采用竖直落指，保持横向间距，不用向物体中心的投影把三指挤到一起。全过程三指碰撞代理的最小间隙约 2.29 mm。

几何回归覆盖指尖固定、关节后收、三指同步、先抬后移、辅助指持续接触、三指间隙、掌根降低、关节限位和落点。真实 MuJoCo 静态检查执行 1 秒有限力位置伺服，在后 0.5 秒统计接触：初始姿态由三根主指承担手部载荷；三指抬起并后移的姿态下，拇指/小指平均法向载荷约 0.831/0.758 N，其他手指没有承担土豆接触载荷。最初 8 mm 抬指方案在动力学检查中出现无名指再次轻触土豆，故增加为 12 mm。**这些是分姿态静态检查，不是完整运动过程的抗滑证明。** 动态抗滑、刀面接触力及新动作的 RL 训练仍待开发。

文件位于 `local/outputs/potato/finger_regrasp_sync/`，包括 `finger_regrasp.mp4`、七张分阶段图片和 `measurements.json`；上一版逐指预览文件仍保留在 `finger_regrasp/`。本轮针对新旧土豆模块的 20 项测试通过。

## 较早的 RL 管线原型

当前是第一阶段的可运行研究原型：完整刚体土豆、现有 Wuji 手和双臂、录制退手参考、分区接触力、Gymnasium 环境与 PPO 入口。**刀固定在初始抬起姿态，扰动来自作用于土豆的外力脉冲。尚未实现刀具接触扰动或验证学得的倒手策略。**

用户示范中的任务是弯曲手指扶持食材并逐步退手，不能预设为整手完全松开再抓住。当前参考仍是原有录制与程序修正后的轨迹，尚未从视频恢复接触交替的时序。提供的 YouTube 和 B 站视频均未成功获取内容；B 站返回 HTTP 412。不能把下述实现当作已复现这两个视频。

## 环境与输入

使用 Python 3.12。已有项目 `.venv` 时直接安装；需要新建时可用 `uv venv --python 3.12 .venv`。

```bash
uv pip install --python .venv/bin/python -e '.[simulation,data,dev]' \
  -e packages/robot_core -e packages/cooking_simulation -e packages/wuji_hand
# CPU 原型可先安装 CPU torch，避免默认下载 CUDA 运行库。
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e '.[potato]'
```

依赖本地机器人资产及以下录制（也可用 `--reference` 指定兼容 NPZ）：

`local/data/recordings/recorded_hand_guarded_chop_200hz_latest.npz`

数组契约：`time_s (N,)`、`left_arm_target_rad (N,7)`、`right_arm_target_rad (N,7)`、`left_hand_target_rad (N,20)`；时间从零严格递增，使用弧度。

## 运行与导出

```bash
# 录制参考的零残差对照，导出视频、接触日志、实际 qpos 与报告
MUJOCO_GL=egl .venv/bin/python -m twin_sim.potato_cli replay \
  --episodes 3 --video --output local/outputs/potato/baseline

# 首先用短训练检查流程；2048 步不代表策略收敛
.venv/bin/python -m twin_sim.potato_cli train \
  --steps 2048 --output local/outputs/potato/smoke_training

# 在相同的新种子上分别跑策略与参考对照
MUJOCO_GL=egl .venv/bin/python -m twin_sim.potato_cli evaluate \
  --policy local/outputs/potato/smoke_training/ppo_potato.zip \
  --episodes 3 --seed 100 --video --output local/outputs/potato/smoke_evaluation
.venv/bin/python -m twin_sim.potato_cli replay \
  --episodes 3 --seed 100 --output local/outputs/potato/heldout_baseline
```

无图形运行不需要 `MUJOCO_GL`。渲染需要可用的 EGL 驱动。`--no-disturbance` 关闭外力，`--no-tactile` 清零策略触觉输入。严谨的无触觉对照需要从训练开始使用该开关，不能只在评估时清零便视为公平对照。

输出目录包含视频 MP4、每步日志 NPZ 与汇总 JSON。视频标注 `knife PARKED`、土豆位移和手部接触载荷；训练输出模型 ZIP 和训练元数据 JSON。回放与评估每个种子约 4–7 秒仿真时间，策略可以改变参考时钟速度。

## 当前实现的含义

- 土豆是程序生成的不规则凸网格，视觉和碰撞一致；质量 180 g，长宽高约 120×70×56 mm，并对尺寸和外形做小幅随机化。未标定真实土豆材料和摩擦。
- 位置伺服有力限制，环境正常 step 不重设手或土豆的实际状态。reset 包含 0.5 秒接触稳定过程，然后重新计时；初始接触穿透超过 2 mm 或存在刀手接触时拒绝启动。
- 触觉是 10 个区域×8 通道：五个指垫和五个中间指节碰撞体；通道为法向载荷、局部合力 xyz、局部受力中心 xyz、接触标志。信号为理想物理接触读数，目前没有真实传感器噪声、延迟和历史堆叠。
- 策略可见触觉、本体感知、阶段，以及明确标记的土豆尺寸、相对位置、姿态和速度特权输入。当前 28 维动作是左臂 7 维关节残差、手指 20 维关节残差和 1 维参考时钟速度；不是腕部笛卡尔残差。
- 训练奖励包括位移/速度、接触、超大力和动作变化。当前成功标记只要求参考完成、末态位移小于 20 mm、手有接触且未触发失败，是调试指标，未达到研究验收要求。不能用它替代持续稳定、恢复时间及形状泛化评估。
- 各物理子步检查刀手接触和土豆穿透；失败阈值包括 150 mm 位移或 8 mm 土豆穿透。阈值仅用于该仿真原型。

接下来应先校准扶持姿态与接触区域，加入触觉历史/噪声、阶段奖励和更严格的持续稳定指标，再加入限深刀具接触、无手/反馈控制器对照，以及独立的形状泛化测试。现有短训练只能证明管线可运行。

物理测试：

```bash
.venv/bin/python -m pytest tests/simulation/test_potato_contact.py tests/simulation/test_potato_env.py -q
```

测试使用实际 MuJoCo 接触验证自由落体、可推动性、重量与接触力一致、力方向、无接触归零、可复现 reset、Gymnasium 接口和不覆盖物体动态状态。

## 2026-09-18 的实际短训练检查

PPO 训练 2048 步后保存并重新加载，在新种子 100、101、102 上评估，结果如下。这里只是三条轨迹，不代表统计显著性，也没有证明性能差异来自触觉。

| 种子 | 原始回放最大位移 | 短训练策略最大位移 | 策略成功 |
| --- | --- | --- | --- |
| 100 | 36.55 mm | 23.99 mm | 否 |
| 101 | 36.25 mm | 25.18 mm | 否 |
| 102 | 33.76 mm | 21.59 mm | 否 |

结果位于 `local/outputs/potato/heldout_baseline/` 与 `local/outputs/potato/smoke_evaluation/`。所有这些实验都使用完整土豆、理想接触读数、特权状态和外力脉冲；刀没有参与接触扰动。

本轮仿真与仓库回归：`286 passed, 5 skipped`。三条警告分别来自普通 chop 接触力超过提示阈值，以及 Gymnasium 对无限观测上下界的两条提示。新物理与环境测试共 12 项。
