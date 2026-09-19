# 二维压力传感器 UART 采集与热力图

`sensor_uart_stream.py` 通过 UART 读取二维压力帧，并显示实时热力图。该工具会向传感器发送
`begin` 开始采集；按 `Ctrl+C` 退出时会尽量发送 `end`。

## 现场运行

在仓库根目录执行：

```bash
.venv-hardware/bin/python scripts/hardware/sensor_uart_stream.py \
  --execute \
  --port /dev/ttyUSB0 \
  --baudrate 115200 \
  --heatmap-vmin 0 \
  --heatmap-vmax 2
```

每收到一帧有效二维数据，热力图会刷新。当前现场使用的颜色范围为 0 到 2：值达到或超过 2
会显示为色图顶端颜色。该值是便于观察的临时设定，不是传感器量程；得知传感器最大量程后，
应将 `--heatmap-vmax 2` 替换为该量程，以便跨时间直接比较压力。

## 停止与排查

- 在运行该命令的终端按 `Ctrl+C` 停止。
- 如果命令行持续打印全 0 帧，表示当前收到的数据值为 0；调整热力图上限不会改变传感器读数。
- 未弹出窗口时，先确认图形桌面可用；程序会在启动时报告所选图形后端。
