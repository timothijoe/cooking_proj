# Tianji Arm 接口

Python 模块位于 `tianji_arm`，实验入口使用：

```bash
.venv-hardware/bin/python -m tianji_arm.experiments.<module> --help
```

SDK 的 A/B 臂、左右臂映射和角度/弧度转换必须在适配层明确完成。离线规划不得连接设备；
dry-run 若会读取控制器反馈，必须在输出和教程中说明。运动接口必须要求 `--execute` 并保证
退出去使能或安全停止。
