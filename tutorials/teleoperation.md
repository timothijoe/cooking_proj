# 键盘与手柄遥操作

Tianji 遥操作使用真机环境，但默认 dry-run。先只查看帮助：

```bash
./scripts/teleop/tianji/keyboard_jog.sh --help
./scripts/teleop/tianji/gamepad_jog.sh --help
./scripts/teleop/tianji/gamepad_simstyle_jog.sh --help
```

键盘 W/S、A/D、R/F 请求 ±X、±Y、±Z 单步，Q 或空格退出。手柄以 RB 为 deadman，
Start 退出。dry-run 仍可能连接控制器读取当前姿态，但不会发送运动命令。

不要在未核对工作空间、坐标方向、设备臂映射和物理急停时添加 `--execute`。Wuji 的旧
`play_mirrored_hand.sh` 尚未适配新目录，本教程暂不推荐运行。
