# 录制与回放

运行并保存 guarded-chop 状态：

```bash
./scripts/simulation/run_guarded_chop_record.sh local/data/recordings/guarded_chop.npz
```

只回放已有状态，不重新运行控制器：

```bash
./scripts/simulation/replay_guarded_chop_2x.sh local/data/recordings/guarded_chop.npz
```

也可以直接使用 CLI：

```bash
.venv/bin/twin-sim guarded-chop-replay \
  --recording local/data/recordings/guarded_chop.npz --rate 2.0
```

MCAP、NPZ、JSON、CSV 和 Viewer 运行输出都放在 `local/data/`。保留原始录制，派生文件使用
新文件名；不要覆盖唯一原件，也不要把大数据加入 Git。
