"""Timestamp-based replay for validated Wuji Hand trajectories."""

from dataclasses import dataclass
import time

import numpy as np

from tianji_robotics.wuji_hand.interfaces import WujiHandBackend
from tianji_robotics.wuji_hand.models import HandTrajectory
from tianji_robotics.wuji_hand.validation import validate_trajectory


@dataclass(frozen=True)
class ReplaySummary:
    frame_count: int
    duration_s: float


def replay_trajectory(
    trajectory: HandTrajectory,
    backend: WujiHandBackend,
    *,
    realtime: bool,
    max_step_rad: float = 0.2,
) -> ReplaySummary:
    """Command all frames and advance the backend between their timestamps."""
    trajectory = _preflight_for_backend(
        trajectory, backend, max_step_rad=max_step_rad
    )
    timestep_s = float(getattr(backend, "timestep_s", 0.0))
    if not np.isfinite(timestep_s) or timestep_s <= 0.0:
        raise ValueError("backend timestep_s must be positive and finite")

    backend.command_position_rad(trajectory.positions_rad[0])
    first_timestamp_ns = int(trajectory.timestamps_ns[0])
    executed_steps = 0
    for frame_index in range(1, trajectory.positions_rad.shape[0]):
        interval_s = float(
            trajectory.timestamps_ns[frame_index]
            - trajectory.timestamps_ns[frame_index - 1]
        ) / 1_000_000_000.0
        elapsed_s = float(
            int(trajectory.timestamps_ns[frame_index]) - first_timestamp_ns
        ) / 1_000_000_000.0
        target_steps = round(elapsed_s / timestep_s)
        step_count = max(0, target_steps - executed_steps)
        started = time.monotonic()
        for _ in range(step_count):
            backend.step(timestep_s)
        executed_steps += step_count
        if realtime and not bool(getattr(backend, "realtime_paced", False)):
            remaining = interval_s - (time.monotonic() - started)
            if remaining > 0.0:
                time.sleep(remaining)
        backend.command_position_rad(trajectory.positions_rad[frame_index])

    duration_s = float(
        trajectory.timestamps_ns[-1] - trajectory.timestamps_ns[0]
    ) / 1_000_000_000.0
    return ReplaySummary(
        frame_count=trajectory.positions_rad.shape[0], duration_s=duration_s
    )


def _preflight_for_backend(
    trajectory: HandTrajectory,
    backend: WujiHandBackend,
    *,
    max_step_rad: float,
) -> HandTrajectory:
    ranges = dict(backend.joint_ranges_rad)
    tolerance = float(getattr(backend, "range_tolerance_rad", 0.0))
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("backend range_tolerance_rad must be finite and non-negative")
    expanded = {
        name: (float(bounds[0]) - tolerance, float(bounds[1]) + tolerance)
        for name, bounds in ranges.items()
    }
    validate_trajectory(trajectory, expanded, max_step_rad=max_step_rad)
    if tolerance == 0.0:
        return trajectory

    clipped = trajectory.positions_rad.copy()
    for joint_index, name in enumerate(trajectory.joint_names):
        lower, upper = ranges[name]
        clipped[:, joint_index] = np.clip(
            clipped[:, joint_index], float(lower), float(upper)
        )
    normalized = HandTrajectory(
        timestamps_ns=trajectory.timestamps_ns,
        positions_rad=clipped,
        joint_names=trajectory.joint_names,
        metadata=trajectory.metadata,
    )
    return validate_trajectory(normalized, ranges, max_step_rad=max_step_rad)
