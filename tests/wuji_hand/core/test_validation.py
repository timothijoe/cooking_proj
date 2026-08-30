import numpy as np
import pytest

from tianji_robotics.wuji_hand.models import HandTrajectory
from tianji_robotics.wuji_hand.names import HAND_JOINT_NAMES
from tianji_robotics.wuji_hand.validation import validate_trajectory


def ranges():
    return {name: (-1.0, 1.0) for name in HAND_JOINT_NAMES}


def valid_positions(frame_count=3):
    return np.zeros((frame_count, 20), dtype=np.float64)


def trajectory_with(positions):
    return HandTrajectory(
        np.arange(len(positions), dtype=np.int64) * 10 + 10,
        positions,
        HAND_JOINT_NAMES,
        {"source": "fixture"},
    )


def test_validation_returns_the_validated_trajectory():
    trajectory = trajectory_with(valid_positions())

    assert validate_trajectory(trajectory, ranges(), max_step_rad=0.2) is trajectory


def test_validation_reports_first_out_of_range_joint():
    positions = valid_positions()
    positions[0, 3] = 2.0
    trajectory = trajectory_with(positions)

    with pytest.raises(ValueError, match="frame 0.*left_finger1_joint4"):
        validate_trajectory(trajectory, ranges(), max_step_rad=0.2)


def test_validation_reports_excessive_frame_step():
    positions = valid_positions(frame_count=2)
    positions[1, 7] += 0.3
    trajectory = trajectory_with(positions)

    with pytest.raises(ValueError, match="frame 1.*max step"):
        validate_trajectory(trajectory, ranges(), max_step_rad=0.2)


def test_validation_rejects_large_int64_step_without_integer_overflow():
    positions = np.zeros((2, 20), dtype=np.int64)
    positions[0, 0] = np.iinfo(np.int64).min
    trajectory = trajectory_with(positions)
    wide_ranges = {name: (-1e20, 1e20) for name in HAND_JOINT_NAMES}

    with pytest.raises(ValueError, match="frame 1.*max step"):
        validate_trajectory(trajectory, wide_ranges, max_step_rad=0.2)


def test_validation_requires_a_range_for_every_canonical_joint():
    with pytest.raises(ValueError, match="left_finger5_joint4"):
        validate_trajectory(
            trajectory_with(valid_positions()),
            ranges() | {"left_finger5_joint4": None},
            max_step_rad=0.2,
        )
