# Simulation scripts

这些 shell 脚本是 `twin-sim` 的短包装；稳定且最适合自动化的入口仍是
`.venv/bin/twin-sim`。先在仓库根目录运行 `./scripts/setup/create_env.sh`，并优先从
[`tutorials/simulation.md`](../../tutorials/simulation.md) 的 headless 示例开始。

- `run_guarded_chop.sh`：打开 plane 场景的护手切菜 Viewer；
- `run_guarded_chop_record.sh [output.npz]`：运行并记录状态；
- `replay_guarded_chop_2x.sh [recording.npz]`：以两倍速度回放状态；
- `run_guarded_chop_record_replay.sh`：运行带录制/回放选项的护手切菜；
- `run_recorded_hand_guarded_chop.sh [input.mcap]`：用录制手势驱动切菜。

录制文件应放在 `local/data/recordings/`。若 `local/` 位于其他磁盘，先设置
`COOKING_LOCAL_ROOT`。这些脚本只启动 MuJoCo 仿真，不会连接实体设备。
