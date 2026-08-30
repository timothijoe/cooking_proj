# 快速开始

本页只保留最短入口。完整人工步骤见 [`tutorials/`](../tutorials/README.md)，开发维护信息见
[`docs/agent/`](agent/README.md)。

## 本地目录

首次运行前确认以下资源已经恢复：

```text
local/
├── assets/robot_assets/
├── assets/MarvinCCS/
├── vendor/SDK_PYTHON/
├── vendor/tianji_test/
└── data/recordings/
```

可以通过 `COOKING_LOCAL_ROOT` 将整个 `local` 根目录放在其他磁盘。

## 仿真环境

```bash
./scripts/setup/create_env.sh
.venv/bin/twin-sim hand-demo --headless
.venv/bin/twin-sim guarded-chop --headless --final-hold 0
```

## 真机环境

```bash
./scripts/setup/create_hardware_env.sh
.venv-hardware/bin/python -m tianji_arm.experiments.arm_command_diagnostic --help
```

先运行帮助和 dry-run。只有设备旁操作者确认后才添加 `--execute`。
