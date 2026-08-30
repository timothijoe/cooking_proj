"""Pure orchestration for offline right-glove to left-hand retargeting."""

import numpy as np

from tianji_robotics.wuji_hand.interfaces import Retargeter, SkeletonSource
from tianji_robotics.wuji_hand.models import HandTrajectory
from tianji_robotics.wuji_hand.names import HAND_JOINT_NAMES
from tianji_robotics.wuji_hand.transforms import mirror_right_to_left


def retarget_recording(
    source: SkeletonSource, retargeter: Retargeter
) -> HandTrajectory:
    """Retarget every right-hand skeleton frame without file or device I/O."""
    timestamps: list[int] = []
    positions: list[np.ndarray] = []
    source_frame_id: str | None = None

    for frame_index, frame in enumerate(source.frames()):
        if frame.side != "right":
            raise ValueError(f"frame {frame_index} must be right-hand input")
        command = np.asarray(
            retargeter.step(mirror_right_to_left(frame.keypoints_m))
        )
        if command.shape != (20,) or not np.issubdtype(command.dtype, np.number):
            raise ValueError(
                f"retargeter frame {frame_index} must return 20 numeric joints"
            )
        if np.issubdtype(command.dtype, np.complexfloating) or not np.isfinite(
            command
        ).all():
            raise ValueError(
                f"retargeter frame {frame_index} must return 20 finite real joints"
            )
        timestamps.append(frame.timestamp_ns)
        positions.append(np.asarray(command, dtype=np.float64))
        if source_frame_id is None:
            source_frame_id = frame.frame_id

    if not positions:
        raise ValueError("recording contains no skeleton frames")

    return HandTrajectory(
        timestamps_ns=np.asarray(timestamps, dtype=np.int64),
        positions_rad=np.asarray(positions, dtype=np.float64),
        joint_names=HAND_JOINT_NAMES,
        metadata={
            "source_frame_id": source_frame_id,
            "input_hand_side": "right",
            "input_transform": "mirror_y_right_wrist_to_left_wrist",
            "target_hand_model": "WujiHand",
            "target_hand_side": "left",
            "unit": "rad",
            "frame_count": len(positions),
            "offline_only": True,
        },
    )
