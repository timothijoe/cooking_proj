"""Stable paths to vendored simulation assets."""

from pathlib import Path

from robot_core.paths import asset_root


def official_wuji_left_mjcf() -> Path:
    path = asset_root() / "robot_assets/mujoco/wuji_hand_standalone/mjcf/left.xml"
    if not path.is_file():
        raise FileNotFoundError(f"official Wuji left-hand model is missing: {path}")
    return path


def official_wuji_hand_mjcf(side: str) -> Path:
    if side not in {"left", "right"}:
        raise ValueError("hand side must be 'left' or 'right'")
    path = asset_root() / "robot_assets/mujoco/wuji_hand_standalone/mjcf" / f"{side}.xml"
    if not path.is_file():
        raise FileNotFoundError(f"official Wuji {side}-hand model is missing: {path}")
    return path
