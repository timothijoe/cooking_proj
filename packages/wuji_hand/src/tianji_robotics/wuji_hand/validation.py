"""Safety validation for offline Wuji hand trajectories."""

from collections.abc import Mapping

import numpy as np

from .models import HandTrajectory


def validate_trajectory(
    trajectory: HandTrajectory,
    joint_ranges_rad: Mapping[str, tuple[float, float]],
    *,
    max_step_rad: float,
) -> HandTrajectory:
    """Reject targets outside configured ranges or with unsafe frame-to-frame jumps."""
    if not np.isfinite(max_step_rad) or max_step_rad <= 0:
        raise ValueError("max_step_rad must be finite and positive")

    limits: list[tuple[float, float]] = []
    for name in trajectory.joint_names:
        joint_range = joint_ranges_rad.get(name)
        if joint_range is None:
            raise ValueError(f"missing range for joint {name}")
        try:
            lower, upper = joint_range
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid range for joint {name}") from error
        if not np.isfinite((lower, upper)).all() or lower > upper:
            raise ValueError(f"invalid range for joint {name}")
        limits.append((lower, upper))

    for frame_index, frame in enumerate(trajectory.positions_rad):
        for joint_index, (lower, upper) in enumerate(limits):
            value = frame[joint_index]
            if value < lower or value > upper:
                name = trajectory.joint_names[joint_index]
                raise ValueError(
                    f"frame {frame_index} joint {name} is outside range [{lower}, {upper}]"
                )

    for frame_index in range(1, len(trajectory.positions_rad)):
        deltas = np.abs(
            trajectory.positions_rad[frame_index] - trajectory.positions_rad[frame_index - 1]
        )
        excess = np.flatnonzero(deltas > max_step_rad)
        if excess.size:
            joint_index = int(excess[0])
            name = trajectory.joint_names[joint_index]
            raise ValueError(
                f"frame {frame_index} joint {name} exceeds max step {max_step_rad}"
            )

    return trajectory
