# Lightweight Capability Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将最新旧仓库的仿真、Wuji 和 Tianji 能力整理进轻量 `cooking_proj`，同时将大型本地资源完全排除在 Git 外。

**Architecture:** 保留现有成熟模块的行为，按 `robot_core`、`twin_sim`、`wuji_hand`、`tianji_arm` 和 `cooking_workflows` 分包。运行时资源统一从 `local/` 解析，Git 中只保存 manifest、配置模板、入口和关键测试。

**Tech Stack:** Python 3.12、setuptools、pytest、NumPy、MuJoCo、标准 `venv`

## Global Constraints

- ROS 2 暂不迁移。
- 旧仓库只读。
- 全部现有真机脚本均迁移并分类，后续再筛选。
- `local/`、虚拟环境、SDK、模型、录制和输出不得被 Git 跟踪。
- 真机入口默认不运动。
- 只允许本地提交；不得推送。

---

### Task 1: 仓库骨架与资源边界

**Files:**
- Modify: `.gitignore`
- Create: `.python-version`
- Create: `pyproject.toml`
- Create: `tests/repository/test_layout.py`
- Create: `configs/local.example.toml`
- Create: `manifests/README.md`
- Create: `scripts/setup/create_env.sh`
- Create: `scripts/setup/create_hardware_env.sh`

- [ ] 先写仓库边界测试，验证五个包、四类脚本目录、`local/` 忽略规则和 ROS 2 缺席。
- [ ] 运行测试并确认因目录和配置尚不存在而失败。
- [ ] 创建最小骨架、环境元数据、配置模板和资源说明。
- [ ] 运行边界测试并确认通过。

### Task 2: 仿真能力迁移

**Files:**
- Create: `packages/cooking_simulation/src/twin_sim/`
- Create: `packages/cooking_simulation/pyproject.toml`
- Create: `scripts/simulation/`
- Create: `experiments/simulation/README.md`
- Create: `tests/simulation/`
- Create locally only: `local/assets/simulation/`

- [ ] 先迁移一组核心 CLI、遥操作和无窗口测试，确认资源缺失时给出明确提示。
- [ ] 迁移最新 `twin_sim` 实现及仿真入口，排除 ROS 2 模块和历史过程文件。
- [ ] 将模型和网格复制到被忽略的 `local/assets/simulation`，生成跟踪的资源 manifest。
- [ ] 运行核心仿真测试和 CLI 帮助检查。

### Task 3: Wuji Hand 能力迁移

**Files:**
- Create: `packages/wuji_hand/src/wuji_hand/`
- Create: `packages/wuji_hand/pyproject.toml`
- Create: `scripts/hardware/wuji/`
- Create: `scripts/teleop/wuji/`
- Create: `experiments/real_robot/wuji_hand/README.md`
- Create: `tests/wuji_hand/`

- [ ] 先写关节模型、NPZ、默认无设备访问和脚本清单测试。
- [ ] 迁移数据模型、MCAP/NPZ、重定向、MuJoCo 后端、桌面后退和角度条能力。
- [ ] 迁移全部 Wuji 真机脚本并改用 `local/vendor`、`local/data` 配置。
- [ ] 运行 Wuji 离线和安全边界测试。

### Task 4: Tianji 真机与遥操作能力迁移

**Files:**
- Create: `packages/tianji_arm/src/tianji_arm/`
- Create: `packages/tianji_arm/pyproject.toml`
- Create: `scripts/hardware/tianji/`
- Create: `scripts/teleop/tianji/`
- Create: `experiments/real_robot/tianji_arm/README.md`
- Create: `tests/tianji_arm/`

- [ ] 先写全部旧真机脚本映射、dry-run 和执行授权测试。
- [ ] 迁移全部 Tianji 真机脚本，按诊断、轨迹、切菜、批处理和遥操作分类。
- [ ] 消除开发机器绝对路径，改用本地配置和资源根目录。
- [ ] 运行纯离线规划、参数校验和安全门测试。

### Task 5: 跨设备工作流、文档与总体验证

**Files:**
- Create: `packages/cooking_workflows/src/cooking_workflows/`
- Create: `packages/cooking_workflows/pyproject.toml`
- Modify: `README.md`
- Create: `docs/quickstart.md`
- Create: `docs/capabilities.md`
- Create: `docs/safety.md`
- Create: `manifests/migration-map.yaml`

- [ ] 迁移双臂加手同步执行、录制回放和轨迹生成工作流。
- [ ] 建立旧文件到新位置的简洁映射，确认没有遗漏脚本。
- [ ] 更新 README、快速开始、能力成熟度和安全边界。
- [ ] 运行全部离线测试、CLI 帮助、Git 大文件扫描和仓库体积检查。
- [ ] 本地提交最终迁移结果，不推送。
