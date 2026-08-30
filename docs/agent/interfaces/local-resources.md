# 本地资源接口

默认资源根为仓库内 `local/`，可设置 `COOKING_LOCAL_ROOT=/absolute/path` 整体替换。约定：

```text
local/assets/robot_assets/   # 通用机器人/MuJoCo 模型
local/assets/MarvinCCS/      # Tianji 双臂模型
local/vendor/SDK_PYTHON/     # Tianji 厂商 SDK
local/vendor/tianji_test/    # 厂商/历史辅助代码
local/data/recordings/       # 原始录制和生成输出
local/data/recordings_extra/ # 补充录制
```

`local/`、虚拟环境、MCAP、NPZ 和运行日志均不进入 Git。可提交的只有清单、许可说明、
小型配置模板和生成这些资源的代码。资源来源见 `manifests/local-assets.yaml`。
