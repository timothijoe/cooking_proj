from html import escape
from pathlib import Path
from typing import Sequence

import numpy as np

from twin_sim.logging import SimulationSample


_WIDTH = 1000
_HEIGHT = 480
_PANEL_Y = 62
_PANEL_W = 400
_PANEL_H = 330


def write_trajectory_svg(
    path: str | Path,
    samples: Sequence[SimulationSample],
) -> None:
    destination = Path(path)
    if not destination.parent.is_dir():
        raise ValueError(
            f"SVG parent directory does not exist: {destination.parent}"
        )
    if not samples:
        raise ValueError("samples must not be empty")

    target = np.asarray(
        [sample.target_pose[:3, 3] for sample in samples],
        dtype=float,
    )
    actual = np.asarray(
        [sample.actual_pose[:3, 3] for sample in samples],
        dtype=float,
    )
    if target.shape != actual.shape or not np.isfinite(
        np.concatenate((target, actual))
    ).all():
        raise ValueError("sample poses must contain finite XYZ positions")

    top_target, top_actual, top_mapper = _top_points(target, actual)
    side_target, side_actual = _side_points(target, actual)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_WIDTH}" '
        f'height="{_HEIGHT}" viewBox="0 0 {_WIDTH} {_HEIGHT}">',
        '<rect width="100%" height="100%" fill="#101722"/>',
        _text(250, 32, "Top View (X-Y)", size=20, anchor="middle"),
        _text(750, 32, "Side View (Path-Z)", size=20, anchor="middle"),
        _panel(50),
        _panel(550),
        _polyline(top_target, "#f4b942", dashed=True),
        _polyline(top_actual, "#48cae4"),
        _polyline(side_target, "#f4b942", dashed=True),
        _polyline(side_actual, "#48cae4"),
    ]
    seen: set[int] = set()
    for sample in samples:
        if (
            sample.phase != "HOLD"
            or sample.cut_index <= 0
            or sample.cut_index in seen
        ):
            continue
        seen.add(sample.cut_index)
        x, y = top_mapper(sample.target_pose[0, 3], sample.target_pose[1, 3])
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="#ff6b6b"/>')
        parts.append(
            _text(
                x + 8,
                y - 8,
                f"Cut {sample.cut_index}",
                size=12,
                anchor="start",
            )
        )
    parts.extend(
        (
            _legend_line(65, 430, "#f4b942", "Target", dashed=True),
            _legend_line(185, 430, "#48cae4", "Actual"),
            _text(250, 462, "X / Y position (m)", size=12, anchor="middle"),
            _text(750, 462, "Horizontal path distance / Z (m)", size=12, anchor="middle"),
            "</svg>",
        )
    )
    with destination.open("w", encoding="utf-8") as stream:
        stream.write("\n".join(parts))
        stream.write("\n")


def _top_points(
    target: np.ndarray,
    actual: np.ndarray,
):
    combined = np.vstack((target[:, :2], actual[:, :2]))
    lower, upper = _limits(combined)

    def mapper(x: float, y: float) -> tuple[float, float]:
        px = 50 + (x - lower[0]) / (upper[0] - lower[0]) * _PANEL_W
        py = _PANEL_Y + _PANEL_H - (
            (y - lower[1]) / (upper[1] - lower[1]) * _PANEL_H
        )
        return float(px), float(py)

    return (
        [mapper(*point[:2]) for point in target],
        [mapper(*point[:2]) for point in actual],
        mapper,
    )


def _side_points(
    target: np.ndarray,
    actual: np.ndarray,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    target_s = _horizontal_progress(target)
    actual_s = _horizontal_progress(actual)
    combined = np.column_stack(
        (
            np.concatenate((target_s, actual_s)),
            np.concatenate((target[:, 2], actual[:, 2])),
        )
    )
    lower, upper = _limits(combined)

    def mapper(distance: float, z: float) -> tuple[float, float]:
        px = 550 + (distance - lower[0]) / (upper[0] - lower[0]) * _PANEL_W
        py = _PANEL_Y + _PANEL_H - (
            (z - lower[1]) / (upper[1] - lower[1]) * _PANEL_H
        )
        return float(px), float(py)

    return (
        [mapper(s, z) for s, z in zip(target_s, target[:, 2], strict=True)],
        [mapper(s, z) for s, z in zip(actual_s, actual[:, 2], strict=True)],
    )


def _horizontal_progress(xyz: np.ndarray) -> np.ndarray:
    increments = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(increments)))


def _limits(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    span = upper - lower
    padding = np.maximum(span * 0.08, 0.002)
    return lower - padding, upper + padding


def _panel(x: int) -> str:
    return (
        f'<rect x="{x}" y="{_PANEL_Y}" width="{_PANEL_W}" '
        f'height="{_PANEL_H}" fill="#172233" stroke="#53657d" stroke-width="1"/>'
    )


def _polyline(
    points: list[tuple[float, float]],
    color: str,
    *,
    dashed: bool = False,
) -> str:
    coordinates = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    dash = ' stroke-dasharray="8 5"' if dashed else ""
    return (
        f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
        f'stroke-width="2.5"{dash}/>'
    )


def _text(
    x: float,
    y: float,
    value: str,
    *,
    size: int,
    anchor: str,
) -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" fill="#e5edf7" '
        f'font-family="sans-serif" font-size="{size}" '
        f'text-anchor="{anchor}">{escape(value)}</text>'
    )


def _legend_line(
    x: int,
    y: int,
    color: str,
    label: str,
    *,
    dashed: bool = False,
) -> str:
    dash = ' stroke-dasharray="8 5"' if dashed else ""
    return (
        f'<line x1="{x}" y1="{y}" x2="{x + 36}" y2="{y}" '
        f'stroke="{color}" stroke-width="3"{dash}/>'
        + _text(x + 44, y + 4, label, size=13, anchor="start")
    )
