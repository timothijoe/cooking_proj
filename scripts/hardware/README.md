# Hardware scripts

此目录预留给稳定的真机 shell 包装；当前推荐入口安装为 Python 模块和 `wuji-*` 命令，而
不是直接运行本目录的文件。

先创建 `.venv-hardware`：

```bash
./scripts/setup/create_hardware_env.sh
.venv-hardware/bin/python -m tianji_arm.experiments.arm_command_diagnostic --help
```

真机命令的风险等级、只读检查和执行条件见
[`tutorials/hardware/`](../../tutorials/hardware/README.md)。未经设备旁操作者明确确认，不要
添加 `--execute`，也不要运行会直接驱动 Wuji 手的 `wuji-*` 命令。
