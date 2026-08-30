import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SimulationSample:
    time_s: float
    phase: str
    target_joints_rad: np.ndarray
    actual_joints_rad: np.ndarray
    target_pose: np.ndarray
    actual_pose: np.ndarray
    raw_force_n: float
    filtered_force_n: float
    force_over_threshold: bool
    cut_index: int = 0


_FIELDS = (
    "time_s",
    "phase",
    "cut_index",
    *(f"target_q_{index}" for index in range(7)),
    *(f"actual_q_{index}" for index in range(7)),
    *(f"target_pose_{index}" for index in range(16)),
    *(f"actual_pose_{index}" for index in range(16)),
    "raw_force_n",
    "filtered_force_n",
    "force_over_threshold",
)


def write_csv(path: str | Path, samples: Iterable[SimulationSample]) -> None:
    with CsvLogger(path) as logger:
        for sample in samples:
            logger.write(sample)


class CsvLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.parent.is_dir():
            raise ValueError(f"CSV parent directory does not exist: {self.path.parent}")
        self._stream = None
        self._writer = None

    def __enter__(self) -> "CsvLogger":
        self._stream = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._stream, fieldnames=_FIELDS)
        self._writer.writeheader()
        self._stream.flush()
        return self

    def write(self, sample: SimulationSample) -> None:
        if self._writer is None or self._stream is None:
            raise RuntimeError("CsvLogger must be opened before writing")
        self._writer.writerow(_row(sample))
        self._stream.flush()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._stream is not None:
            self._stream.close()
        self._stream = None
        self._writer = None


def _row(sample: SimulationSample) -> dict[str, object]:
    target_joints = _array(sample.target_joints_rad, (7,), "target_joints_rad")
    actual_joints = _array(sample.actual_joints_rad, (7,), "actual_joints_rad")
    target_pose = _array(sample.target_pose, (4, 4), "target_pose").reshape(-1)
    actual_pose = _array(sample.actual_pose, (4, 4), "actual_pose").reshape(-1)
    if not np.isfinite(sample.time_s) or sample.time_s < 0.0:
        raise ValueError("time_s must be non-negative and finite")
    if not isinstance(sample.phase, str) or not sample.phase:
        raise ValueError("phase must be a non-empty string")
    if (
        isinstance(sample.cut_index, bool)
        or not isinstance(sample.cut_index, (int, np.integer))
        or sample.cut_index < 0
    ):
        raise ValueError("cut_index must be a non-negative integer")
    if not np.isfinite(sample.raw_force_n) or not np.isfinite(
        sample.filtered_force_n
    ):
        raise ValueError("force values must be finite")
    row: dict[str, object] = {
        "time_s": float(sample.time_s),
        "phase": sample.phase,
        "cut_index": int(sample.cut_index),
    }
    row.update(
        {f"target_q_{index}": float(value) for index, value in enumerate(target_joints)}
    )
    row.update(
        {f"actual_q_{index}": float(value) for index, value in enumerate(actual_joints)}
    )
    row.update(
        {f"target_pose_{index}": float(value) for index, value in enumerate(target_pose)}
    )
    row.update(
        {f"actual_pose_{index}": float(value) for index, value in enumerate(actual_pose)}
    )
    row.update(
        {
            "raw_force_n": float(sample.raw_force_n),
            "filtered_force_n": float(sample.filtered_force_n),
            "force_over_threshold": bool(sample.force_over_threshold),
        }
    )
    return row


def _array(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must have shape {shape} and finite values") from error
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must have shape {shape} and finite values")
    return array
