from pathlib import Path

import numpy as np
import pytest

from tianji_robotics.data.mcap import write_joint_state_mcap
from tianji_robotics.wuji_hand.models import HandTrajectory
from tianji_robotics.wuji_hand.names import HAND_JOINT_NAMES
from twin_sim.recorded_hand_guard import (
    load_recorded_guard_cycle,
    shape_pip_led_guard_hand,
)
from twin_sim.robot import RightArmRobot


def _canonical_mcap(tmp_path):
    timestamps = np.arange(8, dtype=np.int64) * 10_000_000
    positions = np.zeros((8, 20))
    positions[:, 4:8] = np.linspace(0.0, .07, 8)[:, None]
    trajectory = HandTrajectory(timestamps, positions, HAND_JOINT_NAMES, {})
    return write_joint_state_mcap(trajectory, tmp_path / "hand.mcap")


def test_adapter_preserves_recorded_endpoints_and_resamples_to_control_clock(tmp_path):
    cycle = load_recorded_guard_cycle(_canonical_mcap(tmp_path), control_dt_s=.01)

    assert cycle.source_kind == "joint_states"
    assert cycle.source_frame_count == 8
    np.testing.assert_allclose(cycle.hand_positions_rad[0, 4:8], 0.0)
    np.testing.assert_allclose(cycle.hand_positions_rad[-1, 4:8], .07)
    np.testing.assert_allclose(np.diff(cycle.timestamps_s), .01, atol=1e-12)
    assert cycle.relative_palm_transforms.shape == (8, 4, 4)
    assert cycle.initial_palm_transform.shape == (4, 4)
    np.testing.assert_allclose(cycle.relative_palm_transforms[0], np.eye(4), atol=1e-9)


def test_adapter_rejects_non_positive_control_period(tmp_path):
    with pytest.raises(ValueError, match="control_dt_s"):
        load_recorded_guard_cycle(_canonical_mcap(tmp_path), control_dt_s=0.0)


def test_real_recording_evidence_when_available():
    path = Path(__file__).resolve().parents[2] / Path(
        "recordings/wuji/august_02/"
        "session_20260802_174440_936_right_to_left_wuji_hand.mcap"
    )
    if not path.exists():
        pytest.skip("local ignored Wuji recording is unavailable")

    cycle = load_recorded_guard_cycle(path, control_dt_s=.01)

    assert cycle.source_frame_count == 499
    assert cycle.source_duration_s == pytest.approx(4.150116, abs=1e-5)
    assert cycle.motion_start_frame == 311
    assert cycle.motion_end_frame == 464
    assert cycle.retreat_distance_m == pytest.approx(.03, abs=.002)
    assert cycle.maximum_joint_correction_rad == pytest.approx(0.0)


def test_real_recording_is_reshaped_to_pip_led_guard_when_available():
    path = Path(__file__).resolve().parents[2] / Path(
        "recordings/wuji/august_02/"
        "session_20260802_174440_936_right_to_left_wuji_hand.mcap"
    )
    if not path.exists():
        pytest.skip("local ignored Wuji recording is unavailable")
    cycle = load_recorded_guard_cycle(path, control_dt_s=.01)
    robot = RightArmRobot(viewer=False)
    try:
        limits = robot.sim.model.actuator_ctrlrange[robot.sim.hand.actuator_ids]
        shaped = shape_pip_led_guard_hand(
            cycle.hand_positions_rad, cycle.phases, limits
        )
    finally:
        robot.close()

    assert shaped.shape == cycle.hand_positions_rad.shape
    protected = (0, 1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19)
    np.testing.assert_array_equal(
        shaped[:, protected], cycle.hand_positions_rad[:, protected]
    )
    retreat = np.asarray(cycle.phases) == "RETREAT"
    for mcp, pip in zip((4, 8, 12, 16), (6, 10, 14, 18), strict=True):
        assert np.max(shaped[retreat, mcp]) <= .30
        assert np.min(shaped[retreat, pip] - shaped[retreat, mcp]) >= .25
    assert np.all(shaped >= limits[:, 0] - 1e-8)
    assert np.all(shaped <= limits[:, 1] + 1e-8)
    assert np.max(np.abs(np.diff(shaped, axis=0))) <= .12
    assert np.corrcoef(shaped[:, 10], shaped[:, 14])[0, 1] > 0.0
    assert not np.array_equal(shaped[:, 6], shaped[:, 10])
