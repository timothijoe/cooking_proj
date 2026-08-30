#!/usr/bin/env python3
"""Generate a local, offline-only left-arm entry trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tianji_robotics.data.left_arm_entry import (
    generate_left_arm_entry,
    save_left_arm_entry_npz,
)


def parse_start_deg(value: str) -> np.ndarray:
    try:
        result = np.asarray([float(item) for item in value.split(",")], dtype=float)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--start-deg must contain seven comma-separated numbers"
        ) from error
    if result.shape != (7,) or not np.all(np.isfinite(result)):
        raise argparse.ArgumentTypeError(
            "--start-deg must contain seven finite comma-separated numbers"
        )
    return result


def load_destination_rad(source_npz: Path) -> np.ndarray:
    try:
        with np.load(source_npz, allow_pickle=False) as data:
            values = np.asarray(data["left_arm_target_rad"], dtype=float)
    except (KeyError, OSError, ValueError) as error:
        raise ValueError(
            "source NPZ must contain a readable left_arm_target_rad array"
        ) from error
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != 7 or not np.all(np.isfinite(values)):
        raise ValueError("left_arm_target_rad must be a finite (N, 7) array with N >= 1")
    return values[0].copy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-npz", required=True, type=Path)
    parser.add_argument("--start-deg", required=True, type=parse_start_deg)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--sample-rate-hz", type=float, default=200.0)
    args = parser.parse_args()
    try:
        trajectory = generate_left_arm_entry(
            np.deg2rad(args.start_deg),
            load_destination_rad(args.source_npz),
            duration_s=args.duration_s,
            sample_rate_hz=args.sample_rate_hz,
        )
        saved = save_left_arm_entry_npz(
            trajectory,
            source_npz=args.source_npz,
            destination=args.output,
            start_deg=args.start_deg,
            duration_s=args.duration_s,
            sample_rate_hz=args.sample_rate_hz,
        )
    except ValueError as error:
        parser.error(str(error))
    print(f"path: {saved}")
    print(f"frame_count: {trajectory.time_s.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
