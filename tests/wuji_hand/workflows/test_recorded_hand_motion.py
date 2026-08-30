from pathlib import Path

import numpy as np
import pytest

from tianji_robotics.workflows.recorded_hand_motion import (
    detect_motion_interval,
    finger_displacement_correlations,
    smooth_joint_step_outliers,
)
from tianji_robotics.wuji_hand.models import HandTrajectory
from tianji_robotics.wuji_hand.names import HAND_JOINT_NAMES


def _trajectory(positions):
    return HandTrajectory(
        np.arange(len(positions), dtype=np.int64) * 10_000_000,
        np.asarray(positions, dtype=float),
        HAND_JOINT_NAMES,
        {},
    )


def test_detects_dominant_interval_without_changing_samples():
    positions = np.zeros((499, 20))
    progress = np.linspace(0.0, 1.0, 138)
    positions[320:458, 8:20] = progress[:, None]
    positions[458:, 8:20] = 1.0
    trajectory = _trajectory(positions)

    interval = detect_motion_interval(trajectory)

    assert abs(interval.start_frame - 320) <= 15
    assert abs(interval.end_frame - 457) <= 15
    np.testing.assert_array_equal(trajectory.positions_rad, positions)


def test_correlations_keep_index_separate_and_long_fingers_coupled():
    time = np.linspace(0.0, 1.0, 100)
    positions = np.zeros((100, 20))
    positions[:, 4:8] = np.sin(time * 8.0)[:, None]
    positions[:, 8:12] = time[:, None]
    positions[:, 12:16] = (time * 1.05)[:, None]
    positions[:, 16:20] = (time * .9)[:, None]

    correlations = finger_displacement_correlations(positions)

    assert correlations["middle_ring"] >= .85
    assert correlations["ring_little"] >= .75
    assert correlations["index_middle"] < correlations["middle_ring"]


def test_smoothing_changes_only_steps_above_limit():
    positions = np.zeros((4, 20))
    positions[2:, 7] = .3

    corrected = smooth_joint_step_outliers(positions, limit_rad=.12)

    assert np.max(np.abs(np.diff(corrected, axis=0))) <= .12
    np.testing.assert_array_equal(corrected[:, :7], positions[:, :7])
    np.testing.assert_array_equal(positions[2:, 7], [.3, .3])


def test_real_recording_motion_evidence_when_available():
    path = Path(
        "../../recordings/wuji/august_02/"
        "session_20260802_174440_936_right_to_left_wuji_hand.npz"
    ).resolve()
    if not path.exists():
        pytest.skip("local ignored Wuji recording is unavailable")
    with np.load(path, allow_pickle=False) as archive:
        positions = archive["left_joint_positions_rad"]
        timestamps = archive["timestamps_ns"]
    trajectory = HandTrajectory(timestamps, positions, HAND_JOINT_NAMES, {})

    interval = detect_motion_interval(trajectory)
    correlations = finger_displacement_correlations(positions)

    assert len(positions) == 499
    assert abs(interval.start_frame - 320) <= 20
    assert abs(interval.end_frame - 457) <= 20
    assert correlations["middle_ring"] >= .85
    assert correlations["ring_little"] >= .75
