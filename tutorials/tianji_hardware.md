# Tianji 真机入门

先创建真机环境，并确认物理急停、净空、工具、坐标方向、A/B 臂映射和网络地址。第一步
只看帮助或做离线规划：

```bash
./scripts/setup/create_hardware_env.sh
.venv-hardware/bin/python -m tianji_arm.experiments.arm_command_diagnostic --help
.venv-hardware/bin/python -m tianji_arm.experiments.plan_sampled_ik_chop --help
```

遥操作 dry-run 见[遥操作教程](teleoperation.md)。部分 dry-run 会连接并读取反馈，但不发送
运动。实际运动需要 `--execute`；本仓库迁移后尚未完成现场真机验收，因此不要复制旧项目
中的 IP、初始角度、工作空间或切菜参数直接运行。

所有 trace CSV 写入 `local/data/recordings/`。异常时先停止命令并使用物理急停；键盘空格或
deadman 只是软件停止，不是急停替代品。
