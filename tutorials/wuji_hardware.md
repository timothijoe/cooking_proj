# Wuji 真机入门

当前推荐只做离线 preflight：

```bash
./scripts/setup/create_hardware_env.sh
.venv-hardware/bin/tianji-robot hardware wuji-sdk preflight \
  local/data/recordings/trajectory.npz
```

该命令只验证轨迹，不搜索、连接、使能或控制实体手。实体工具位于
`wuji_hand.tools`，但迁移后尚未上机复验；旧 `play_mirrored_hand.sh` 也仍有目录假设，
不要把它当作当前的一键运行入口。

未来上机前必须确认序列号、左右手、20 关节顺序、限位、温度、错误码、缓入速度和异常
去使能，并让物理急停保持可达。
