#!/usr/bin/env python3
"""传感器 UART 数据采集程序 — 通过 /dev/ttyUSB0 读取传感器数据。

依据《数据通信协议_M0305.pdf》：
  - 有线 UART（USB—TTL）通信，波特率默认 115200
  - 每包数据为 MxN 的字符串数组，结尾加换行符
  - 示例： "[[92, 74, 50, 38, 66], [65, 77, 0, 3, 6], [1, 70, 46, 99, 53]]\\n"
  - 3x5 数组共 15 个数据点
  - 开始采集指令："begin\\n"
  - 结束采集指令："end\\n"

默认干跑（DRY_RUN），不打开串口；加 --execute 才真正连接设备采集。
Ctrl+C 退出时会尽量发送 "end\\n" 停止采集。

Usage:
    # 干跑（只打印配置，不连接设备）
    .venv-hardware/bin/python scripts/hardware/sensor_uart_stream.py

    # 真机采集 5 秒并打印每一帧
    .venv-hardware/bin/python scripts/hardware/sensor_uart_stream.py --execute --duration-s 5

    # 持续采集直到 Ctrl+C，并把每一帧写入 CSV
    .venv-hardware/bin/python scripts/hardware/sensor_uart_stream.py \
        --execute --output-csv /tmp/sensor_uart_trace.csv

    # 自定义端口/波特率
    .venv-hardware/bin/python scripts/hardware/sensor_uart_stream.py \
        --execute --port /dev/ttyUSB0 --baudrate 115200
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

DEFAULT_PORT = "/dev/ttyUSB0"
DEFAULT_BAUDRATE = 115200
BEGIN_CMD = b"begin\n"
END_CMD = b"end\n"
# 协议示例是 3x5 = 15 个数据点；这里只做校验提示，不强制形状，
# 因为不同传感器 M/N 可能不同。形状不对时只告警，仍原样输出。
EXPECTED_ROWS = 3
EXPECTED_COLS = 5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="通过 UART(/dev/ttyUSB0) 读取传感器数据并解析为二维数组。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", default=DEFAULT_PORT,
                        help="串口设备路径")
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE,
                        help="波特率")
    parser.add_argument("--duration-s", type=float, default=None,
                        help="采集时长（秒）。不指定则持续到 Ctrl+C")
    parser.add_argument("--execute", action="store_true",
                        help="连接设备并采集。不指定则干跑")
    parser.add_argument("--output-csv", type=Path, default=None,
                        help="把每一帧写入 CSV（可选）")
    parser.add_argument("--print-stride", type=int, default=1,
                        help="每 N 帧打印一次，避免刷屏。0 表示不打印")
    parser.add_argument("--read-timeout-s", type=float, default=1.0,
                        help="单行读取超时（秒），超时即重试")
    parser.add_argument("--raw-log", type=Path, default=None,
                        help="把每一条原始行（含 SCANNING 等状态文本）按 "
                             "'timestamp_s\\traw_text' 写入该文件，便于排查设备行为")
    parser.add_argument("--quiet-drops", action="store_true", default=True,
                        help="对连续相同的丢帧告警只打印一次并计数（避免被 "
                             "SCANNING...... 刷屏）")
    parser.add_argument("--no-heatmap", dest="heatmap", action="store_false",
                        help="禁用实时热力图（默认 --execute 采集时自动开启；"
                             "需要 matplotlib 与图形环境）")
    parser.add_argument("--heatmap-vmin", type=float, default=0.0,
                        help="热力图颜色下限（默认 0）")
    parser.add_argument("--heatmap-vmax", type=float, default=2.0,
                        help="热力图颜色上限（临时默认 2；可按传感器量程覆盖）")
    parser.add_argument("--heatmap-title", default="Sensor Heatmap",
                        help="热力图标题（帧信息会追加在标题后）")
    args = parser.parse_args(argv)
    if args.duration_s is not None and args.duration_s <= 0:
        raise ValueError("--duration-s 必须为正数")
    if args.print_stride < 0:
        raise ValueError("--print-stride 不能为负")
    return args


def parse_frame(line: bytes) -> list[list[int]]:
    """把一帧字节串解析为二维整数数组。

    协议每包以 \\n 结尾，内容是 JSON 可解析的二维数组字符串。
    """
    text = line.decode("utf-8", errors="replace").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"无法解析为 JSON: {text!r} ({exc})") from exc
    if not isinstance(data, list) or not data or not isinstance(data[0], list):
        raise ValueError(f"期望二维数组，得到: {text!r}")
    return data


def open_serial(port: str, baudrate: int, read_timeout_s: float):
    """打开串口。延迟导入 serial，避免干跑时缺库报错。"""
    try:
        import serial  # pyserial
    except ImportError as exc:
        raise RuntimeError(
            "未安装 pyserial，请先安装： pip install pyserial"
        ) from exc
    try:
        return serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=read_timeout_s,
            write_timeout=1.0,
        )
    except (PermissionError, serial.SerialException) as exc:
        # errno 13 / EACCES：当前用户无权访问该设备节点
        import os
        has_perm = isinstance(exc, PermissionError) or getattr(exc, "errno", None) == 13
        if has_perm:
            raise RuntimeError(
                f"无权访问 {port}（Permission denied）。\n"
                f"该设备通常属于 dialout 组。把自己加入该组后重新登录即可：\n"
                f"    sudo usermod -aG dialout $USER   # 之后注销/重启，或用 newgrp dialout\n"
                f"或临时用 sudo 运行本脚本测试。\n"
                f"原始错误: {exc}"
            ) from exc
        raise


def drain_input(ser) -> None:
    """打开串口后清空接收缓冲区，避免读到上一会话的残留帧。"""
    try:
        waiting = ser.in_waiting
        if waiting:
            ser.read(waiting)
    except Exception:
        pass


def write_cmd(ser, cmd: bytes, name: str) -> None:
    try:
        ser.write(cmd)
        ser.flush()
    except Exception as exc:
        print(f"⚠️ 发送 {name} 指令失败: {exc}", file=sys.stderr)


def open_csv(path: Path | None):
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(fh)
    writer.writerow(["timestamp_s", "frame_index", "rows", "cols", "flat_data"])
    return fh, writer


def setup_heatmap(vmin: float, vmax, title: str = "Sensor Heatmap"):
    """初始化 matplotlib 实时热力图窗口。

    返回 (plt, fig, ax, im)，供 update_heatmap 与关闭使用。
    自动尝试 Qt/Tk 交互式后端；都没有时退回 Agg（不弹窗，采集/CSV 照常），
    并屏蔽 matplotlib 的 non-interactive 重复告警，避免每帧刷屏。
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    chosen = None
    for be in ("QtAgg", "Qt5Agg", "TkAgg"):
        try:
            mpl.use(be, force=True)
            _t = plt.figure()      # 强制加载后端模块，验证是否真的可用
            plt.close(_t)
            chosen = be
            break
        except Exception:
            continue
    if chosen is not None:
        print(f"✅ 热力图窗口已打开（后端 {chosen}）")
    else:
        mpl.use("Agg", force=True)  # 回退到非交互后端，确保后续 figure 不报错
        import warnings
        warnings.filterwarnings("ignore", message=".*non-interactive.*")
        print("⚠️ 未找到图形后端（缺 tkinter/Qt），热力图窗口不会显示；"
              "采集/CSV 照常工作。", file=sys.stderr)
        print("   如需窗口： pip install PySide6  或  sudo apt install python3.12-tk",
              file=sys.stderr)
    fig, ax = plt.subplots(figsize=(6, 4))
    data = np.zeros((EXPECTED_ROWS, EXPECTED_COLS))
    im = ax.imshow(data, cmap="viridis",
                   vmin=vmin, vmax=vmax if vmax is not None else 1.0,
                   aspect="auto", origin="upper")
    ax.set_title(f"{title} (waiting for data ...)")
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("value")
    plt.ion()
    plt.show(block=False)
    return plt, fig, ax, im


def update_heatmap(plt_mod, fig, ax, im, frame, frame_index, elapsed,
                   vmin, vmax, title: str = "Sensor Heatmap"):
    """用新解析到的帧刷新热力图。"""
    import numpy as np
    arr = np.array(frame, dtype=float)
    if vmax is None:
        lo = float(arr.min()) if arr.size else 0.0
        hi = float(arr.max()) if arr.size else 1.0
        if hi - lo < 1e-6:
            hi = lo + 1.0
        im.set_clim(lo, hi)
    im.set_data(arr)
    ax.set_title(f"{title} #{frame_index} [{elapsed:.2f}s]")
    fig.canvas.draw_idle()
    plt_mod.pause(0.001)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    print("=" * 60)
    print("传感器 UART 采集")
    print(f"  串口:     {args.port}")
    print(f"  波特率:   {args.baudrate}")
    print(f"  时长:     {args.duration_s if args.duration_s else '持续到 Ctrl+C'}")
    print(f"  输出 CSV: {args.output_csv or '无'}")
    print(f"  打印步幅: {'不打印' if args.print_stride == 0 else args.print_stride}")
    print(f"  热力图:   {'开' if args.heatmap else '关'}")
    print(f"  热力图标题: {args.heatmap_title}")
    print("=" * 60)

    if not args.execute:
        print("\nDRY_RUN: 配置已就绪，未连接设备。加 --execute 开始采集。")
        print("         （采集时会自动显示热力图，加 --no-heatmap 关闭）")
        return 0

    ser = None
    csv_handle = None
    csv_writer = None
    raw_log = None
    try:
        csv_handle, csv_writer = open_csv(args.output_csv) or (None, None)
    except OSError as exc:
        print(f"ERROR: 无法打开 CSV: {exc}")
        return 2
    if args.raw_log is not None:
        try:
            args.raw_log.parent.mkdir(parents=True, exist_ok=True)
            raw_log = args.raw_log.open("w", encoding="utf-8")
        except OSError as exc:
            print(f"ERROR: 无法打开 raw-log: {exc}")
            return 2

    frame_index = 0
    drop_state = {"msg": None, "count": 0}  # 连续相同丢帧的合并计数
    t0 = time.perf_counter()
    deadline = (t0 + args.duration_s) if args.duration_s else None
    heatmap_state = None  # (plt, fig, ax, im)
    try:
        print(f"\n打开串口 {args.port} @ {args.baudrate} ...")
        ser = open_serial(args.port, args.baudrate, args.read_timeout_s)
        drain_input(ser)
        print("✅ 串口已打开，发送 begin 指令开始采集")
        write_cmd(ser, BEGIN_CMD, "begin")

        if args.heatmap:
            try:
                heatmap_state = setup_heatmap(args.heatmap_vmin, args.heatmap_vmax,
                                              args.heatmap_title)
            except Exception as exc:
                print(f"⚠️ 无法初始化热力图（需要 matplotlib 与图形环境）: {exc}",
                      file=sys.stderr)
                print("   将继续采集但不显示图形。", file=sys.stderr)
                heatmap_state = None

        print("采集进行中，Ctrl+C 停止\n")
        while True:
            if deadline is not None and time.perf_counter() >= deadline:
                print("\n⏱ 到达指定时长，停止采集")
                break
            try:
                line = ser.readline()
            except Exception as exc:
                print(f"⚠️ 读取异常: {exc}", file=sys.stderr)
                continue
            if not line:
                # 读超时，继续循环；同时让 GUI 保持响应
                if heatmap_state is not None:
                    heatmap_state[0].pause(0.001)
                continue
            elapsed = time.perf_counter() - t0
            if raw_log is not None:
                raw_text = line.decode("utf-8", errors="replace").rstrip("\r\n")
                raw_log.write(f"{elapsed:.6f}\t{raw_text}\n")
                raw_log.flush()
            try:
                frame = parse_frame(line)
            except ValueError as exc:
                msg = str(exc)
                if args.quiet_drops:
                    if msg == drop_state["msg"]:
                        drop_state["count"] += 1
                        continue
                    if drop_state["count"] > 0:
                        print(f"   ↳ 上一告警重复 {drop_state['count']} 次", file=sys.stderr)
                    drop_state["msg"] = msg
                    drop_state["count"] = 1
                    print(f"⚠️ 丢帧: {msg}", file=sys.stderr)
                else:
                    print(f"⚠️ 丢帧: {msg}", file=sys.stderr)
                continue
            # 解析成功：先把之前累积的重复丢帧计数收尾打印
            if drop_state["count"] > 1:
                print(f"   ↳ 上一告警重复 {drop_state['count'] - 1} 次 "
                      f"(设备可能一直在发状态文本如 SCANNING......)", file=sys.stderr)
            drop_state["msg"] = None
            drop_state["count"] = 0
            if not frame:
                continue

            rows = len(frame)
            cols = len(frame[0]) if rows else 0
            flat = [int(v) for row in frame for v in row]
            if rows != EXPECTED_ROWS or cols != EXPECTED_COLS:
                print(f"⚠️ 帧 #{frame_index} 形状 {rows}x{cols} 与协议 {EXPECTED_ROWS}x{EXPECTED_COLS} 不符，仍记录")

            if heatmap_state is not None:
                plt_mod, fig_hm, ax_hm, im_hm = heatmap_state
                update_heatmap(plt_mod, fig_hm, ax_hm, im_hm,
                               frame, frame_index, elapsed,
                               args.heatmap_vmin, args.heatmap_vmax,
                               args.heatmap_title)

            if csv_writer is not None:
                csv_writer.writerow([
                    f"{elapsed:.6f}", frame_index, rows, cols,
                    ";".join(str(v) for v in flat),
                ])
                csv_handle.flush()

            if args.print_stride and (frame_index % args.print_stride == 0):
                print(f"[{elapsed:7.3f}s #{frame_index:>6}] {frame}")

            frame_index += 1

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断，正在停止采集 ...")
    except RuntimeError as exc:
        # open_serial 抛出的可读错误（如无权限、缺 pyserial），直接打印，不弹 traceback
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        # 收尾打印仍在累积的重复丢帧计数
        if drop_state["count"] > 1:
            print(f"   ↳ 上一告警重复 {drop_state['count'] - 1} 次 "
                  f"(设备可能一直在发状态文本如 SCANNING......)", file=sys.stderr)
        if heatmap_state is not None:
            try:
                heatmap_state[0].close("all")
            except Exception:
                pass
            print("✅ 已关闭热力图窗口")
        if ser is not None:
            write_cmd(ser, END_CMD, "end")
            try:
                ser.close()
            except Exception:
                pass
            print("✅ 已发送 end 指令并关闭串口")
        if csv_handle is not None:
            csv_handle.close()
            print(f"✅ 已写入 CSV: {args.output_csv} ({frame_index} 帧)")
        if raw_log is not None:
            raw_log.close()
            print(f"✅ 已写入 raw-log: {args.raw_log}")
        if frame_index:
            rate = frame_index / max(time.perf_counter() - t0, 1e-6)
            print(f"采集统计: {frame_index} 帧, 平均 {rate:.2f} 帧/秒")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
