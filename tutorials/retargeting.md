# 手套 Retargeting

把 Wuji Studio MCAP 放到 `local/data/recordings/`。先查看实际参数：

```bash
.venv/bin/tianji-robot sim wuji-replay --help
```

典型无窗口转换与回放：

```bash
.venv/bin/tianji-robot sim wuji-replay local/data/recordings/input.mcap \
  --headless \
  --npz local/data/recordings/output.npz \
  --joint-state-mcap local/data/recordings/output_joint_states.mcap
```

程序按 topic 判断原始手套骨架或标准 joint states。输出仍是本地数据，不要提交 Git。
若报官方 retargeter 缺失，检查 `local/vendor/` 与 `COOKING_LOCAL_ROOT`。
