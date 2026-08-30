# 运行 MuJoCo 仿真

先完成[环境安装](environment_setup.md)。最短无窗口检查：

```bash
.venv/bin/twin-sim hand-demo --headless
.venv/bin/twin-sim guarded-chop --headless --final-hold 0
```

第二条命令应正常退出并报告 5 刀、4 次退手和安全距离。打开双臂切菜 Viewer：

```bash
./scripts/simulation/run_guarded_chop.sh
```

关闭 Viewer 即可停止。录制手势联动需要先把对应 MCAP 放入 `local/data/recordings/`，再用：

```bash
.venv/bin/twin-sim recorded-hand-guarded-chop \
  --hand-mcap local/data/recordings/input.mcap \
  --headless --final-hold 0
```

使用项目默认录制并打开 Viewer 时，也可直接运行
`./scripts/simulation/run_recorded_hand_guarded_chop.sh`；该脚本的可选第一个参数是 MCAP 路径。

其他案例先查看 `.venv/bin/twin-sim --help` 和子命令 `--help`。仿真不会连接实体机械臂或手。
