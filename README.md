# Tianji Cooking

轻量化的 Tianji 双臂与 Wuji Hand 烹饪机器人工作区。

当前整理三条能力线：

- 双臂、灵巧手与切菜任务的 MuJoCo 仿真；
- Wuji Hand 数据处理、仿真、遥操作与真机工具；
- Tianji 双臂轨迹、诊断、遥操作与真机实验。

ROS 2 暂不包含在本阶段。模型、厂商 SDK、录制数据和运行输出统一放在被 Git
忽略的 `local/`；仓库只保存代码、配置模板和资源清单。

## 快速开始

```bash
./scripts/setup/create_env.sh
.venv/bin/twin-sim --help
.venv/bin/tianji-robot --help
```

真机工具使用独立环境：

```bash
./scripts/setup/create_hardware_env.sh
```

真机入口默认不运动。任何实际运动都必须显式提供相应的执行参数，并在设备旁完成
急停、空间和反馈检查。

实际操作从[使用教程](tutorials/README.md)开始。Agent 和维护者查阅
[开发知识库](docs/agent/README.md)；项目总览另见[能力清单](docs/capabilities.md)和
[安全边界](docs/safety.md)。

希望在其他项目复用同一套整理方法时，复制
[通用项目文档整理 Prompt](PROMPTS/documentation_organizer.md)并填写开头的项目参数。
