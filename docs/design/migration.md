# 轻量迁移设计

## 目标

将最新 `tianji_robotic_project` 的三条能力线迁入 `cooking_proj`：

1. 双臂切菜 MuJoCo 仿真；
2. Wuji Hand 数据处理、仿真、遥操作和真机工具；
3. Tianji 双臂真机调试、轨迹执行和遥操作。

ROS 2 暂不迁移。旧仓库保持只读；所有现有脚本先迁移整理，后续再筛选。

## 目录边界

```text
packages/
  robot_core/
  cooking_simulation/
  wuji_hand/
  tianji_arm/
  cooking_workflows/
experiments/
  simulation/
  real_robot/
scripts/
  setup/
  simulation/
  hardware/
  teleop/
configs/
manifests/
local/
docs/
tests/
```

- `packages` 保存可复用实现，包之间保持明确依赖方向。
- `experiments` 保存具体仿真和真机实验入口及简短说明。
- `scripts` 只保存稳定的短入口和环境脚本。
- `configs` 只保存可公开模板，不保存设备密钥、序列号或机器绝对路径。
- `manifests` 记录外部资源的版本、来源、哈希和安装位置。
- `local` 保存模型、纹理、厂商 SDK、录制数据和运行输出，整个目录被 Git 忽略。
- `docs` 只保存长期有效的快速开始、能力说明和安全边界，不迁移历史聊天、过程计划和迁移台账。
- `tests` 只保留直接保护现有能力和安全门的测试。

## Python 环境

- `.venv`：Python 3.12，仿真、数据处理和测试。
- `.venv-hardware`：Python 3.12，Tianji/Wuji 厂商 SDK 和真机操作。
- Git 只跟踪 `.python-version`、`pyproject.toml`、依赖锁文件和环境创建脚本。
- 虚拟环境、缓存、构建产物和本机 SDK 均不进入 Git。

## 迁移原则

- 以 `/home/linux/september_folder/tianji_robotic_project` 当前工作区为最新来源，包括尚未提交的新开发文件。
- 不复制旧 Git 历史、虚拟环境、ROS 构建产物、录制数据、压缩包、厂商二进制和大模型资产。
- 所有现有真机脚本先迁移并分类；成熟度标记为稳定、实验中或尚未真机验证。
- 遥操作覆盖 MuJoCo Xbox、Tianji 键盘/Xbox、Wuji 角度条和手套数据回放。
- 真机入口默认不运动；保留 dry-run、显式执行授权、限位、反馈检查和退出清理。
- 当前分支可以产生本地提交，但未经用户明确确认不得推送远端。

## 验收

- 新仓库不跟踪大型资产、SDK、录制文件或虚拟环境。
- 三条能力线均有清晰代码归属、入口和简短文档。
- 现有脚本都有迁移去向，没有静默遗漏。
- 核心离线测试和无窗口仿真入口能够运行。
- Git 状态和文件体积检查能够证明仓库保持轻量。
