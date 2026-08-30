"""Offline quintic left-arm trajectory entry generation."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EntryTrajectory:
    """Sampled left-arm trajectory and its finite-difference peak bounds."""

    time_s: np.ndarray
    left_arm_target_rad: np.ndarray
    peak_velocity_rad_s: np.ndarray
    peak_acceleration_rad_s2: np.ndarray


def validate_joint_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    """Return an independent seven-joint finite floating-point vector."""
    result = np.asarray(values, dtype=float)
    if result.shape != (7,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain seven finite joint values")
    return result.copy()


def generate_left_arm_entry(
    start_rad: np.ndarray,
    destination_rad: np.ndarray,
    *,
    duration_s: float = 20.0,
    sample_rate_hz: float = 200.0,
) -> EntryTrajectory:
    """Generate a zero-boundary-derivative quintic left-arm entry trajectory."""
    start = validate_joint_vector(start_rad, name="start_rad")
    destination = validate_joint_vector(destination_rad, name="destination_rad")
    if not np.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be positive and finite")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive and finite")

    intervals = int(round(float(duration_s) * float(sample_rate_hz)))
    if intervals < 1 or not np.isclose(intervals / sample_rate_hz, duration_s):
        raise ValueError("duration_s * sample_rate_hz must be a positive integer")

    time_s = np.linspace(0.0, float(duration_s), intervals + 1)
    phase = time_s / float(duration_s)
    blend = 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
    joints = start + blend[:, None] * (destination - start)
    joints[0], joints[-1] = start, destination

    velocity = np.gradient(joints, time_s, axis=0)
    acceleration = np.gradient(velocity, time_s, axis=0)
    return EntryTrajectory(
        time_s,
        joints,
        np.max(np.abs(velocity), axis=0),
        np.max(np.abs(acceleration), axis=0),
    )


def save_left_arm_entry_npz(
    trajectory: EntryTrajectory,
    *,
    source_npz: Path,
    destination: Path,
    start_deg: np.ndarray,
    duration_s: float,
    sample_rate_hz: float,
) -> Path:
    """Write an offline-only left-arm entry trajectory NPZ."""
    destination = Path(destination)
    if "offline_entry_only" not in destination.stem:
        raise ValueError("output filename must contain 'offline_entry_only'")
    entry_start_deg = validate_joint_vector(start_deg, name="start_deg")
    if not np.allclose(
        np.deg2rad(entry_start_deg), trajectory.left_arm_target_rad[0]
    ):
        raise ValueError("start_deg must match the trajectory's initial joint target")

    np.savez(
        destination,
        format_version=np.asarray(1, dtype=np.int64),
        time_s=trajectory.time_s,
        left_arm_target_rad=trajectory.left_arm_target_rad,
        source_npz_path=np.asarray(str(Path(source_npz))),
        entry_start_deg=entry_start_deg,
        entry_destination_rad=trajectory.left_arm_target_rad[-1],
        duration_s=np.asarray(duration_s, dtype=float),
        sample_rate_hz=np.asarray(sample_rate_hz, dtype=float),
        interpolation=np.asarray("quintic_smoothstep"),
        peak_velocity_rad_s=trajectory.peak_velocity_rad_s,
        peak_acceleration_rad_s2=trajectory.peak_acceleration_rad_s2,
    )
    return destination
