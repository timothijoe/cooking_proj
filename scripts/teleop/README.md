# Teleoperation scripts

Tianji 的当前包装入口位于 `tianji/`：

```bash
./scripts/teleop/tianji/keyboard_jog.sh --help
./scripts/teleop/tianji/gamepad_jog.sh --help
./scripts/teleop/tianji/gamepad_simstyle_jog.sh --help
```

它们都使用 `.venv-hardware`，默认不发送运动目标；dry-run 仍可能连接控制器读取反馈。
执行前先阅读 [`tutorials/teleoperation.md`](../../tutorials/teleoperation.md) 和
[`tutorials/hardware/tianji_teleoperation.md`](../../tutorials/hardware/tianji_teleoperation.md)。

`wuji/play_mirrored_hand.sh` 仍包含旧环境和旧录制目录假设，**不是当前支持的入口**。需要
Wuji 手操作时使用 `tianji-robot hardware wuji-sdk preflight` 做离线检查，并遵循
[`tutorials/hardware/wuji_hand.md`](../../tutorials/hardware/wuji_hand.md)。
