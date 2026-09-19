"""Regression tests for the standalone UART sensor utility."""

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/hardware/sensor_uart_stream.py"
SPEC = importlib.util.spec_from_file_location("sensor_uart_stream", SCRIPT)
sensor_uart_stream = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(sensor_uart_stream)


def test_heatmap_uses_provisional_fixed_vmax_by_default():
    args = sensor_uart_stream.parse_args([])

    assert args.heatmap_vmax == 2.0
