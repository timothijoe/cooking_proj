"""Analysis helpers for recorded Wuji finger motion."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MotionInterval:
    start_frame: int
    end_frame: int


def detect_motion_interval(trajectory) -> MotionInterval:
    positions = np.asarray(trajectory.positions_rad, dtype=float)
    if len(positions) < 4:
        return MotionInterval(0, len(positions) - 1)
    energy = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    window = min(15, len(energy) if len(energy) % 2 else len(energy) - 1)
    window = max(1, window)
    smooth_energy = np.convolve(energy, np.ones(window) / window, mode="same")
    peak = int(np.argmax(smooth_energy))
    threshold = float(smooth_energy[peak]) * 0.5
    energy_start = peak
    while energy_start > 0 and smooth_energy[energy_start - 1] >= threshold:
        energy_start -= 1
    trailing = np.flatnonzero(smooth_energy[peak:] >= threshold)
    energy_end = peak + int(trailing[-1])

    coupled = np.linalg.norm(positions[:, 8:20] - positions[0, 8:20], axis=1)
    smooth_coupled = np.convolve(coupled, np.ones(window) / window, mode="same")
    coupled_peak = int(np.argmax(smooth_coupled))
    coupled_threshold = float(smooth_coupled[coupled_peak]) * 0.75
    crossings = np.flatnonzero(smooth_coupled[: coupled_peak + 1] >= coupled_threshold)
    displacement_start = int(crossings[0]) if len(crossings) else energy_start

    start = min(energy_start, displacement_start)
    end = min(len(positions) - 1, energy_end + 1)
    if end <= start:
        return MotionInterval(0, len(positions) - 1)
    return MotionInterval(start, end)


def finger_displacement_correlations(positions_rad) -> dict[str, float]:
    positions = np.asarray(positions_rad, dtype=float)
    displacement = positions - positions[0]
    magnitudes = np.stack(
        [np.linalg.norm(displacement[:, base : base + 4], axis=1) for base in range(0, 20, 4)],
        axis=1,
    )

    def correlation(first: int, second: int) -> float:
        if np.std(magnitudes[:, first]) <= 1e-12 or np.std(magnitudes[:, second]) <= 1e-12:
            return 0.0
        return float(np.corrcoef(magnitudes[:, first], magnitudes[:, second])[0, 1])

    return {
        "index_middle": correlation(1, 2),
        "middle_ring": correlation(2, 3),
        "ring_little": correlation(3, 4),
    }


def smooth_joint_step_outliers(positions_rad, *, limit_rad: float = 0.12) -> np.ndarray:
    positions = np.asarray(positions_rad, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 20:
        raise ValueError("recorded positions must have shape (frames, 20)")
    if not np.isfinite(limit_rad) or limit_rad <= 0:
        raise ValueError("joint step limit must be positive and finite")
    corrected = positions.copy()
    for index in range(1, len(corrected)):
        corrected[index] = corrected[index - 1] + np.clip(
            corrected[index] - corrected[index - 1], -limit_rad, limit_rad
        )
    return corrected
