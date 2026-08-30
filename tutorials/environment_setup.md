# 安装环境

要求 Linux 和 Python 3.12。不要复制别人的虚拟环境，它包含本机绝对路径。

仿真、数据和测试环境：

```bash
./scripts/setup/create_env.sh
.venv/bin/python --version
.venv/bin/twin-sim --help
```

真机环境：

```bash
./scripts/setup/create_hardware_env.sh
.venv-hardware/bin/python -m tianji_arm.experiments.arm_command_diagnostic --help
```

运行前按 `manifests/local-assets.yaml` 恢复 `local/`。资源在其他磁盘时：

```bash
export COOKING_LOCAL_ROOT=/absolute/path/to/cooking-local
```

常见错误：找不到模型或 MCAP 表示本地资源未恢复；找不到 CLI 表示对应环境尚未创建；
Viewer 无法打开时先运行带 `--headless` 的命令检查代码与资源。
