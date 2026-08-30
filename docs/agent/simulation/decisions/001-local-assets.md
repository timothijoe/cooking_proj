# 001：大资源采用本地注入

状态：accepted

模型、厂商 SDK、MCAP、NPZ 和运行输出可能很大或受许可限制，因此统一放入被 Git 忽略的
`local/`，仓库只跟踪代码、配置模板和 `manifests/` 清单。默认根目录可由
`COOKING_LOCAL_ROOT` 覆盖。

结果是 clone 保持轻量，但首次运行必须先恢复资源；任何代码都不得重新依赖旧机器的绝对
路径。资源布局见[接口文档](../../interfaces/local-resources.md)。
