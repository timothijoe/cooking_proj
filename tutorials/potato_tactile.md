## 当前版本：初始化后刀持续贴近关节

新增可选 `--continuous-knife`。只在最开头加入一次 `INITIAL_APPROACH`：刀从前方约 35 mm、上方约 25 mm 的偏移姿态平滑靠近。随后取消每轮 `KNIFE_CLEAR` 的前退/抬高，也取消轮间独立 `KNIFE_APPROACH` 阶段。刀保留切菜下刀、提刀行程，抬指和后移时继续跟随 PIP 区域。

参考轨迹和角度触发后的重新求解都保持刀近距；每个控制步另用独立 scratch data，针对当步手指控制目标求解刀的前后位置。实际模拟仍使用有限力驱动器，未覆盖真实 qpos、未禁用碰撞。初始化阶段不施加近距跟随，后续全部阶段施加。

沿用相同 PPO 的三轮实测：初始化结束后的刀与 PIP link3 碰撞区域间隙始终约 2.11–4.22 mm；切菜期间约 2.14–3.17 mm，抬指/落指时略增但不再整段退开。刀手接触力为零，刀与刚体土豆最小竖直间隙 2.64 mm。三个位置各三次支撑下刀，三轮均重新按稳，土豆最大位移 3.21 mm。零残差基准位移 7.81 mm，未通过稳定目标；本次未重新训练。

输出 `local/outputs/potato/dynamic_regrasp_continuous_knife/`，逐阶段刀手间隙见 `knife_follow_metrics.json`。新增测试检查一次初始化、无中途退刀阶段及实际动力学全程近距、无过大刀手力。

```bash
MUJOCO_GL=egl OMP_NUM_THREADS=1 .venv/bin/python scripts/simulation/run_dynamic_regrasp.py --policy local/outputs/potato/dynamic_regrasp_angle_gate/policy.zip --wrist-retreat-mm 4 --smooth-regrasp --angle-threshold-deg 80 --curl-release --landing-wrist-retreat-mm 4 --early-pip-curl --support-repeats 3 --regrasp-speed 1.6 --knife-gap-mm 3 --cut-depth-mm 24 --continuous-knife --output local/outputs/potato/dynamic_regrasp_continuous_knife --export
```

修改前源码在 `local/checkpoints/20260918-before-continuous-knife/source.tar.gz`，旧回放保留。不带新参数仍可运行上一版。

---

## 历史版本：倒手提速，刀贴近 PIP 区域展示切菜过程

修改前用户已提交为 `2364d98`，旧参数和旧回放保留。本轮添加可选参数，默认行为不变：`--regrasp-speed 1.6` 只提高三指抬起/后移/落下的参考时钟速度，物理步长和视频播放速度不变；每个位置三次支撑下刀的节奏保留。`--knife-gap-mm 3` 将参考刀手间隙从 6 mm 收紧至 3 mm，`--cut-depth-mm 24` 将刀的竖直行程由 18 mm 加深到 24 mm，仍保留不穿入土豆的几何间隙。

实测同一 PPO 三轮均重新按稳，移指总时间从 4.70 秒降到 2.96 秒，整段从 22.66 秒降到 20.92 秒；土豆最大位移 3.24 mm。支撑下刀期间，刀与 PIP 相关中节碰撞几何间隙约 1.70–2.72 mm（中位 2.23 mm），全过程刀手接触力为零，刀距土豆最小竖直间隙 2.62 mm。支撑阶段三指最大世界位置偏移 0.942 mm，持续有接触。此处的 PIP 间隙指 link3 碰撞胶囊区域，非理想关节点距离。

本次沿用现有 PPO，未重新训练。零残差三轮位移约 8.64 mm，不满足稳定目标。刀仍不切断刚体土豆，没有生成真实分离的土豆片；当前展示三次下刀、提刀、固定指尖退腕、快速移指的完整动作顺序。

```bash
MUJOCO_GL=egl OMP_NUM_THREADS=1 .venv/bin/python scripts/simulation/run_dynamic_regrasp.py --policy local/outputs/potato/dynamic_regrasp_angle_gate/policy.zip --wrist-retreat-mm 4 --smooth-regrasp --angle-threshold-deg 80 --curl-release --landing-wrist-retreat-mm 4 --early-pip-curl --support-repeats 3 --regrasp-speed 1.6 --knife-gap-mm 3 --cut-depth-mm 24 --output local/outputs/potato/dynamic_regrasp_fast_close_cut --export
```

新测试验证实际移指控制步数减少、支撑步数和次数不变、刀手/刀食物间隙与物理接触检查，而非仅改变视频播放速度。

---

## 已提交版本：每个接触位置支撑退腕 3 次，再移指

新增 `--support-repeats 3`（支持 1/2/3，默认 1 保持旧版）。每个接触位置分三次小幅退腕，每次 40 个参考移动采样加 20 个支撑采样；指尖世界目标在整个支撑段固定。刀在该段对应做三次不切入土豆的浅行程。三次全部完成后，还必须满足实际角度 ≥80°和三指接触持续 60 ms，才允许移指；不再因第一次就达到角度而提前跳走。

最终参数把支撑退腕总量设为 4 mm（每次约 1.33 mm），落指辅助退腕仍为 4 mm，两段合计与相邻指尖接触点 8 mm 的间距一致，避免轮间需要退回。轮间从上一轮真实控制终点平滑衔接，消除未重定向旧手形导致的调整。重复支撑模式下，禁止再屈曲的限制用于落指/按稳；进入下一轮的退腕支撑准备后允许关节随固定指尖几何调整，避免限制造成食指在下一支撑段开头补偿滑动。旧的单次模式不变。

同一 PPO 的三个接触位置各完成三次支撑退腕，共九次；三轮均重新按稳。各支撑段三指世界位置偏移最大 0.956 mm，最小指尖法向载荷约 0.199 N，全段有接触，没有提前抬指。土豆全程最大位移 3.36 mm。这里是实际接触保持，不是把指尖或土豆焊死。零残差三轮位移 9.33 mm，未通过稳定目标；本次未重训 PPO。

6 项相关测试通过，包括指尖参考固定、三次退腕及停留、次数完成前不放行、角度/载荷条件与连续接触。输出中 `support_repeats_completed` 及回放 `SUPPORT n/3` 显示完成次数，`support_contact_metrics.json` 保存三轮实际接触指标。

```bash
MUJOCO_GL=egl OMP_NUM_THREADS=1 .venv/bin/python scripts/simulation/run_dynamic_regrasp.py --policy local/outputs/potato/dynamic_regrasp_angle_gate/policy.zip --wrist-retreat-mm 4 --smooth-regrasp --angle-threshold-deg 80 --curl-release --landing-wrist-retreat-mm 4 --early-pip-curl --support-repeats 3 --output local/outputs/potato/dynamic_regrasp_three_support_seated --export
```

旧版回放保留，修改前源码在 `local/checkpoints/20260918-before-three-support-strokes/source.tar.gz`。早期候选 `three_support_strokes*`、`three_support_balanced`、`three_support_continuous` 中第二/三轮食指滑动较大，不是最终回放。

---

## 历史版本：加强离开初段的 PIP 屈曲

修改前认可版完整保存为 `local/checkpoints/20260918-landing-wrist-4mm-preserved/source-policy-results.tar.gz`，包含源码、测试、策略、旧回放、比较数据及说明，另有 README 恢复命令。仍依赖工作区原有资产与 Python 环境。

新增可选 `--early-pip-curl`。只在抬指前半段提前竖直腾空进度：`lift_u = u + 0.12*sin(2*pi*u)^2`（u<0.5），后半段恢复原参数。最大腾空高度仍为 12 mm，DIP 轨迹、水平后移目标、后半段落指目标和 4 mm 退腕配合保持不变。这样 PIP 较早分担屈曲，不通过加大落指后的收缩来补偿。

同一现有 PPO、同一确定性场景三轮对比：离开阶段实际 PIP 净屈曲从旧版约 −0.7–1.7° 增至 3.3–6.2°，三指均更明显屈曲。落指最大 PIP 局部回弹 0.048°（旧版 0.025°），仍小于 0.05°；轮间 DIP 最大回弹约 1.08°，没有声称全程无回弹。三轮均重新按稳，土豆最大位移 1.60 mm；零残差基准也通过三轮，位移 2.04 mm。本次沿用策略，未重新训练。

5 项直接相关测试通过，覆盖早期 PIP 增强、落指几何不变、腕部朝向、落指驱动目标只展开或保持及轮间手形一致。对比指标见 `local/outputs/potato/dynamic_regrasp_early_pip_curl/joint_comparison.json`。

```bash
MUJOCO_GL=egl OMP_NUM_THREADS=1 .venv/bin/python scripts/simulation/run_dynamic_regrasp.py --policy local/outputs/potato/dynamic_regrasp_angle_gate/policy.zip --wrist-retreat-mm 8 --smooth-regrasp --angle-threshold-deg 80 --curl-release --landing-wrist-retreat-mm 4 --early-pip-curl --output local/outputs/potato/dynamic_regrasp_early_pip_curl --export
```

不带 `--early-pip-curl` 可继续运行上一版，旧结果未覆盖。

---

## 已保留版本：落指时腕部轻退 4 mm，减小接触回弹

新增 `--landing-wrist-retreat-mm`，范围 0–4 mm，默认 0，要求 `--curl-release`。落指期间按五次平滑函数沿 +Y 后移，参考腕部高度、朝向不变；按稳阶段保持后移后的姿态，下一轮从该腕部终点平滑衔接。指尖目标同时重新求解，以维持原来的世界接触位置，不通过手腕拖动物体。

同一现有 PPO、相同确定性场景，比较 0/2/4 mm：

| 目标后移 | 落指最大 PIP 回弹 | 轮间最大 DIP 回弹 | 土豆最大位移 |
|---|---|---|---|
| 0 mm | 0.898° | 2.007° | 1.97 mm |
| 2 mm | 0.202° | 1.538° | 1.62 mm |
| 4 mm | 0.025° | 1.072° | 1.63 mm |

三者均完成三轮且重新按稳。4 mm 版本落指区间实际腕部后移约 3.74–3.77 mm，伴随约 0.36–0.38 mm 的竖直跟踪偏移，朝向变化最大 0.067°；不可称真实腕部绝对无旋转或高度完全不动。落指 DIP 没有测得反向回弹，但轮间仍有约 1° 回弹，不能报告为全程零回弹。

比较原始记录及脚本见 `local/outputs/potato/landing_wrist_comparison/`。本次未重训，保持原力矩上限和接触物理。验证了世界指尖目标不变、参考腕部只沿后退方向移动且朝向不变，以及完整动力学回合。

```bash
MUJOCO_GL=egl OMP_NUM_THREADS=1 .venv/bin/python scripts/simulation/run_dynamic_regrasp.py --policy local/outputs/potato/dynamic_regrasp_angle_gate/policy.zip --wrist-retreat-mm 8 --smooth-regrasp --angle-threshold-deg 80 --curl-release --landing-wrist-retreat-mm 4 --output local/outputs/potato/dynamic_regrasp_landing_wrist_4mm --export
```

旧版输出保留，修改前源码位于 `local/checkpoints/20260918-before-landing-wrist-retreat/source.tar.gz`。

---

## 历史版本：取消轮间重新蜷指，落指以展开为主

本轮诊断找到此前遗漏的主要问题：虽然落指阶段 DIP 已展开，但 `WRIST_FOLLOW` 又插值回旧的深屈曲起始手形，实际 DIP 重新屈曲约 40–59°。现在每轮起始和落指参考均采用 30° DIP，保留一定弯曲扣住物体，取消这次大幅手形重置。

离开时 DIP 弯曲拱形增量由 14° 增至 24°，指尖腾空拱高为 12 mm；食指接触目标横向调整 2 mm。角度触发后，使用独立 MuJoCo scratch data，按实际锁定的腕部目标重新求解后续指尖世界位置及 DIP 角度；不会覆盖真实仿真 qpos。对应刀姿也重新求解，保留 6 mm 参考刀手间隙，避免提前触发后仍使用退满腕部的旧几何。

落指、按稳及轮间衔接阶段对 PIP/DIP 驱动目标施加只能展开或保持的限制，并用实际关节角反馈限制目标；这不等于强制真实状态单调。最终实际回放中，落指 PIP 仍有小于 0.9° 的局部回弹，轮间 DIP 约 2°；不能报告为完全无收缩。轮间原先 40–59° 的主动重新蜷指已消除。

最终输出 `local/outputs/potato/dynamic_regrasp_open_landing_feedback/`。PPO 三轮全部重新按稳，土豆最大位移 1.97 mm；零残差三轮也通过，最大位移 2.19 mm。刀手实际接触力为零，最小间隙约 4.54 mm。沿用现有 PPO，未重新训练。相关回归检查通过；新增检查验证轮间 DIP 起始姿态一致、落指目标不能重新屈曲。原 PIP 离开幅度测试的 2° 下限改为 1°，因本轮加强 DIP 后 PIP 分担更少，仍要求两关节均屈曲。

```bash
MUJOCO_GL=egl OMP_NUM_THREADS=1 .venv/bin/python scripts/simulation/run_dynamic_regrasp.py --policy local/outputs/potato/dynamic_regrasp_angle_gate/policy.zip --wrist-retreat-mm 8 --smooth-regrasp --angle-threshold-deg 80 --curl-release --output local/outputs/potato/dynamic_regrasp_open_landing_feedback --export
```

修改前源码保存在 `local/checkpoints/20260918-before-cycle-recurl-fix/source.tar.gz`；此前回放和完整快照均保留。诊断数据见输出目录的 `joint_directions.txt`。MCP 本轮不作为主目标；独立指甲接触面仍未建模。

---

## 历史试验：离开时 PIP/DIP 屈曲，落指时展开

新增 `--curl-release`，依赖平滑倒手模式。此前只约束指尖位置，中指实际 DIP 在落指后仍略屈曲。现在增加 DIP 角度约束：离开时平滑增加屈曲，随后展开到参考 12°；三指仍同步沿连续轨迹后移 8 mm，腾空拱高改为 10 mm，为 PIP/DIP 同时屈曲提供空间。主三指驱动刚度为原值 2.5 倍、阻尼按平方根调整，保持原力矩上限；腕部仍按角度触发锁定。

三轮实际 PPO：离开阶段 PIP 屈曲约 4–6.3°、DIP 屈曲约 8.7–10.3°；落指阶段 PIP 展开约 2.1–4.6°、DIP 展开约 8.5–9.1°。三轮均重新按稳，土豆最大位移 2.08 mm；零残差基准为 3.11 mm，也通过三轮稳定目标。沿用现有策略，未重训。12 项相关测试通过；实际关节方向另由 `joint_motion_metrics.json` 记录。

尚未满足全部要求：MCP 在落指阶段仍增加屈曲约 6–7.7°，没有实现用户期望的轻微伸展。当前接触是半径 11 mm 的球形指腹，没有独立指甲几何或指甲接触测量，不能称为已经实现指甲扣住。当前只是解决 PIP/DIP 先收后展；MCP 配合和末端接触几何需要后续工作。

```bash
MUJOCO_GL=egl OMP_NUM_THREADS=1 .venv/bin/python scripts/simulation/run_dynamic_regrasp.py --policy local/outputs/potato/dynamic_regrasp_angle_gate/policy.zip --wrist-retreat-mm 8 --smooth-regrasp --angle-threshold-deg 80 --curl-release --output local/outputs/potato/dynamic_regrasp_curl_tracking --export
```

修改前源码保存于 `local/checkpoints/20260918-before-curl-release/source.tar.gz`。旧版回放及完整的退腕倒手快照均保留。首次 8 mm 高度且未加强跟踪的候选在 `dynamic_regrasp_curl_release/`，该候选中部分 PIP 实际方向仍错误，不是最终版。

---

## 历史试验：80° 提前触发，连续弧线移指

用户要求保留已认可版本后再调整。修改前完整快照为 `local/checkpoints/20260918-wrist-then-fingers-105658/source-policy-results.tar.gz`，包含仿真源码、脚本、测试、说明、PPO 策略及旧版回放，另有恢复运行命令。快照依赖工作区已有资产和 Python 环境，不包含整套资产或虚拟环境。

新增 `--angle-threshold-deg 80 --smooth-regrasp`，与 `--wrist-retreat-mm 8` 配合使用。角度门槛从 85° 改为 80°，仍要求三指全部达到门槛、接触载荷 >0.1 N 持续 60 ms，实际触发最低角约 81°。保持按住时退腕、触发后锁定腕部目标。

把分开的抬指/移指/落指插值改成单条路径：水平用五次平滑函数，竖直叠加 5 mm 正弦平方拱起；三个阶段标签只是同一条曲线的分段显示，中途不停车。原 2 秒退关节保持缩至 0.2 秒，并去掉 0.4 秒抬指前等待；刀退开阶段保留，轮间仍有跟进动作，因此不是完全取消所有阶段。

最终沿用 PPO 的三轮均重新按稳，全程土豆最大位移 3.20 mm，刀手接触力为零；相比保留版的 2.49 mm，动作节奏更连贯但物体位移略增。本次未重训。零残差三轮基准触发刀手接触失败，不能声称此新参考无需策略也稳定。12 项相关测试通过，删除等待后重跑平滑轨迹测试通过。

```bash
MUJOCO_GL=egl OMP_NUM_THREADS=1 .venv/bin/python scripts/simulation/run_dynamic_regrasp.py --policy local/outputs/potato/dynamic_regrasp_angle_gate/policy.zip --wrist-retreat-mm 8 --smooth-regrasp --angle-threshold-deg 80 --output local/outputs/potato/dynamic_regrasp_smooth_80_final --export
```

原版仍可用不带新参数的旧命令运行，原结果没有覆盖。新参数默认关闭，便于比较或恢复。

---

## 已保留版本：指尖支撑时退腕，角度触发后停腕移指

新增 `--wrist-retreat-mm 8`：切菜准备阶段手腕沿世界 +Y 后移，朝向和高度保持，三指参考接触点固定。三指实际中节指骨角度均 ≥85°且载荷 >0.1 N 持续 60 ms 后，锁定触发时的腕部驱动目标。旧 `KNUCKLE_BACK` 在本模式下仅保持姿态、等待刀退开，不再执行额外退关节动作。随后三指同步抬高 5 mm、后移 8 mm、落下；移指期间手腕目标固定。直接从近垂直指形抬高 12 mm 不可达，因此本模式降低抬指高度。

连续三轮仍保留轮间平滑跟进。因实际角度提前达到，每轮准备阶段手腕实测后移 6.2–6.3 mm；轮间补齐到下一轮起点，始终继续向后推进。三轮移指期间实际腕部位置偏移最大约 0.22 mm，轮内朝向变化约 0.033°。参考指尖固定不等于真实接触不滑：按住阶段指尖实际三维位置变化约 1.3–2.6 mm，尚需优化。

沿用现有 PPO 未重训；三轮均重新按稳，土豆最大位移 2.49 mm。零残差基准位移 8.47 mm，未通过稳定目标。11 项相关测试通过，包括参考固定指尖/后移手腕及角度触发后腕部控制目标保持。结果仅为当前确定性场景。

```bash
MUJOCO_GL=egl OMP_NUM_THREADS=1 .venv/bin/python scripts/simulation/run_dynamic_regrasp.py --policy local/outputs/potato/dynamic_regrasp_angle_gate/policy.zip --wrist-retreat-mm 8 --output local/outputs/potato/dynamic_regrasp_wrist_then_fingers --export
```

输出目录含 `wrist_motion_metrics.json`，记录实际腕部和指尖运动。默认参数仍保留基础动作，使用上面参数运行本试验；不与抬腕或自由腕部 PIP 补偿混用。

---

## 历史试验：只竖直抬腕 5 mm，保持朝向

新增 `--wrist-lift-mm`（默认 0，范围 0–5 mm），不启用已撤回的 `pip_guard`。三指同步抬起时手腕平滑竖直抬高，后移阶段保持高度，落指时回到原高度。手腕目标朝向及水平位置固定，三指仍抬起 12 mm、后移 8 mm。切菜准备和角度触发逻辑未改，不能声称已消除全部关节前探。

实际 PPO 回放抬指阶段 PIP 前探从食指/中指/无名指约 4.25/4.00/4.96 mm 降至 2.06/1.41/2.22 mm；切菜准备阶段前探基本未变。

5 mm 版本沿用现有 PPO，连续三轮全部重新按稳，土豆最大位移 2.75 mm。轮内实际手腕最大朝向偏差约 0.054°；参考姿态偏差小于 0.00004°，参考水平偏移小于 0.001 mm。10 项相关测试通过，包括朝向固定、升降幅度及三指抬起/后移幅度。结果只代表当前确定性场景，本次没有重新训练。

```bash
MUJOCO_GL=egl OMP_NUM_THREADS=1 .venv/bin/python scripts/simulation/run_dynamic_regrasp.py --policy local/outputs/potato/dynamic_regrasp_angle_gate/policy.zip --wrist-lift-mm 5 --output local/outputs/potato/dynamic_regrasp_vertical_wrist --export
```

---

## 基础版本：恢复手指倒手，撤回腕部补偿

用户回看指出 PIP 约束导致拧手腕，破坏了原有倒手效果。因此默认关闭 `pip_guard`，恢复角度触发版本的手指轨迹和驱动增益。每轮倒手主体的手臂参考保持固定，三指同步抬起、后移、落下；轮间仍保留原有手腕跟进。接近垂直的角度触发、三轮连续执行、拇指小指移开和平底土豆均保留。

沿用 `dynamic_regrasp_angle_gate/policy.zip`，本次没有重新训练。输出目录为 `local/outputs/potato/dynamic_regrasp_restored/`。恢复版实测连续三轮全部重新按稳，PPO 全程最大土豆位移 2.75 mm。这次恢复动作形态，并未解决旧版几毫米的关节前探问题；不能再为消除前探而自由改变腕部姿态。用户允许必要时轻微竖直抬腕，但不允许拧腕；当前恢复版尚未新增抬腕。

之前的腕部补偿实验保留代码和结果，仅通过 `--pip-guard` 显式启用，使用时应指定独立 `--output` 目录。

---

## 已撤回的实验：避免 PIP 在倒手前向前探

用户指出近侧指间关节在后移前刻意前伸。诊断确认旧策略回放中，`CUT_ADVANCE` 的 PIP 前向位移约 2–3 mm，`TRIO_LIFT` 约 4–5 mm。

新增 `PipGuardSolver` 联合求解左臂与三指：保持指尖目标位置，同时约束 PIP 的世界 Y 位置。在切菜准备和抬指/移指阶段不先向前探；关节后退阶段向后移动。角度约束直接使用中节指骨与竖直方向的三维夹角，继续由实际角度 ≥85°和接触持续 60 ms 触发。手腕需要配合，已不再承诺每轮整个手臂固定。

为使参考姿态能在动力学中执行，求解加入左臂碰撞部位与砧板至少 3 mm 的几何间隙，避免腕部穿过砧板的不可执行姿态。轨迹按最大关节目标步长 0.008 rad/20 ms 重采样；三指位置驱动增益为原来的 2.5 倍，阻尼相应调整，原力矩上限、摩擦系数、土豆质量、全部碰撞均保留。仍无测试外力，拇指/小指移开。

新输出目录 `local/outputs/potato/dynamic_regrasp_pip_guard/`。本次沿用 `dynamic_regrasp_angle_gate/policy.zip`，未重新训练；参考轨迹、协调控制与接触跟踪进行了调整。零残差参考轨迹累计滑动约 12.3 mm，不能将它报告为稳定策略；沿用现有 PPO 策略的三轮最大位移约 3.76 mm，并通过每轮重新按稳检查。

报告新增 `pip_forward_mm`（相对当前阶段起点）、`max_pre_retreat_pip_forward_mm`（切菜准备、关节后退、抬指和移指阶段的最大前向偏移）。实际动力学仍可能有小幅跟踪误差，不能把参考位置约束说成真实状态被强制固定。落指阶段的数据另保留在逐帧记录中。

最终 PPO 三轮回放中，上述阶段食指、中指、无名指 PIP 最大前向偏移分别为 0.189、0.115、0.278 mm；三轮均重新按稳，拇指和小指接触力为零。连续倒手的 6 项测试通过，其余相关测试在此前运行中通过。轮间连续性检查专门验证轮次边界不重置物体，不限制轮内真实接触导致的逐帧运动。

```bash
MUJOCO_GL=egl .venv/bin/python scripts/simulation/run_dynamic_regrasp.py --policy local/outputs/potato/dynamic_regrasp_angle_gate/policy.zip --export
.venv/bin/python scripts/simulation/run_dynamic_regrasp.py --policy local/outputs/potato/dynamic_regrasp_angle_gate/policy.zip --view
```

修改前源码在 `local/checkpoints/20260918-before-pip-guard/source.tar.gz`。诊断记录在 `local/outputs/potato/pip_direction_diagnostics/`。

---

## 历史版本：中节指骨角度触发倒手

按用户明确的动作顺序：切菜阶段，中节指骨（模型 joint3/PIP 到 joint4/DIP）的角度增大，接近垂直于砧板时才开始关节后退；后退后角度减小到约 52°是正常的，再执行三指同步抬起/后移/落下。

实现方式：

- 新增 `CUT_ADVANCE` 参考阶段，指尖保持接触，中节指骨逐渐接近竖直。食指接触点向外调整 4 mm，以减小横向倾斜造成的可达角度限制。
- 运行时从实际 MuJoCo 关节锚点测量中节指骨与水平砧板的三维夹角，90°表示竖直。必须三指都达到 85°（距离竖直 5°以内），且三指法向载荷均超过 0.1 N，连续三个 20 ms 控制采样满足条件后才触发。
- 条件提前达到就转入 `KNUCKLE_BACK`，无需等参考阶段播完。过渡用衰减的关节目标偏移衔接，不重置实际关节状态；刀保持在土豆上方。
- 条件未达到时保持在 `ANGLE_GATE`；等待超过 5 秒以 `angle_gate_timeout` 终止。每轮只触发一次，避免阈值附近重复触发。
- 删除已不用的辅助指建立/释放等待，保留落指后的稳定等待。拇指、小指仍移开，仍然无测试外力、3 mm 平底、连续三轮且不重置物体。
- PPO 输入新增实际三指角度和持续满足条件的进度；策略仍控制三指压紧量与公共动作速度。触发条件由显式状态逻辑约束，不能声称它是 RL 自行发现的规则。旧策略观察维度不同，需使用新训练结果。

窗口显示 PIP–DIP 指骨角度及 `ARMED/TRIGGERED` 状态。输出目录 `local/outputs/potato/dynamic_regrasp_angle_gate/`，触发记录在评估/逐帧文件的 `angle_gate_events`；另导出每轮触发和结束截图。修改前源码存档在 `local/checkpoints/20260918-before-angle-gate/source.tar.gz`。

```bash
OMP_NUM_THREADS=1 .venv/bin/python scripts/simulation/run_dynamic_regrasp.py --train 16384
MUJOCO_GL=egl .venv/bin/python scripts/simulation/run_dynamic_regrasp.py --policy local/outputs/potato/dynamic_regrasp_angle_gate/policy.zip --export
.venv/bin/python scripts/simulation/run_dynamic_regrasp.py --policy local/outputs/potato/dynamic_regrasp_angle_gate/policy.zip --view
```

本次 PPO 训练 16,384 步后，三轮实际触发角度均在约 85–89°，全程最大位移 2.75 mm（零残差基准 4.52 mm），三轮均通过重新按稳检查。37 项相关测试通过，并额外验证了提前触发和连续三个采样满足条件。不可达角度测试确认会等待并超时，不会继续倒手。

仍以三轮全部重新按稳、累计位移小于 5 mm 为稳定目标。角度触发正确和抓持稳定是分别验证的指标。当前场景无随机化，刀不切断土豆。

---

## 历史版本：连续 3 轮切菜倒手

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
