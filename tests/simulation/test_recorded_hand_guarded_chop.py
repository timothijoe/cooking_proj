from pathlib import Path

import pytest

from twin_sim.tasks.recorded_hand_guarded_chop import (
    RecordedHandGuardedChopConfig,
    run_recorded_hand_guarded_chop,
)


def _real_mcap() -> Path:
    path = Path(__file__).resolve().parents[2] / Path(
        "recordings/wuji/august_02/"
        "session_20260802_174440_936_right_to_left_wuji_hand.mcap"
    )
    if not path.exists():
        pytest.skip("local ignored Wuji recording is unavailable")
    return path


def test_default_run_completes_five_synchronized_cut_and_guard_cycles():
    result = run_recorded_hand_guarded_chop(
        RecordedHandGuardedChopConfig(surface_offsets_m=(.04,)),
        hand_mcap=_real_mcap(),
        viewer=False,
    )
    assert result.success
    assert result.completed_cuts == 5
    assert result.completed_hand_cycles == 5
    for cycle in range(1, 6):
        assert (cycle, "SYNC_CYCLE_START") in result.events
        assert (cycle, "SYNC_CYCLE_COMPLETE") in result.events
        assert (cycle, "HAND_SAFE") not in result.events
    assert result.selected_clearance_tier_m in (.020, .010)
    assert result.minimum_distance_m >= result.selected_clearance_tier_m
    assert result.minimum_lateral_spacing_m >= result.selected_lateral_spacing_m - .003
    assert result.maximum_lateral_spacing_m <= result.selected_lateral_spacing_m + .003
    assert result.maximum_depth_mismatch_m <= .010
    assert result.maximum_hand_penetration_m <= .0005
    assert result.minimum_thumb_clearance_m >= .002
    assert min(result.minimum_long_pad_clearances_m) >= -1e-6
    assert max(result.minimum_long_pad_clearances_m) <= .005
