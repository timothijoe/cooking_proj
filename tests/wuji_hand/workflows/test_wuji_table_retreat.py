import numpy as np
import pytest

from tianji_robotics.simulation.tabletop_wuji_hand import TabletopWujiHand
from tianji_robotics.workflows.wuji_table_retreat import (
    TableRetreatConfig,
    LoopedTableRetreat,
    _calibrated_palm_down_quaternion,
    _fit_palm_down_quaternion,
    _settle_to_table_contact,
    build_looped_table_retreat,
    build_recorded_table_retreat,
    build_table_retreat,
    replay_table_retreat,
)
from tianji_robotics.wuji_hand.models import HandTrajectory
from tianji_robotics.wuji_hand.names import HAND_JOINT_NAMES


FEASIBLE_POSE = np.array([0.144227609, -0.138699993, 1.557536125, -0.008293608, 1.26238215, -0.370000005, 0.97574544, 0.343102217, 1.26902473, -0.213412717, 0.811683357, 0.350952178, 1.41588736, -0.21959424, 0.369700611, 0.648590744, 1.46277189, -0.033096798, 0.353335321, 0.720260799])


def _trajectory():
    return HandTrajectory(np.array([0, 1_000_000]), np.vstack((FEASIBLE_POSE, FEASIBLE_POSE)), HAND_JOINT_NAMES, {})


def test_builds_contact_retreat_and_uses_earliest_equal_candidate():
    backend = TabletopWujiHand(viewer=False)
    try:
        corrected, report = build_table_retreat(_trajectory(), backend, TableRetreatConfig(place_duration_s=.1, retreat_duration_s=.2, hold_duration_s=.1, candidate_stride=1))
    finally:
        backend.close()
    assert report.source_frame == 0
    assert report.actual_retreat_m == pytest.approx(.03)
    assert report.minimum_thumb_clearance_m >= .010
    assert report.maximum_retreat_fingertip_lift_m > 0
    assert report.maximum_hand_penetration_m <= .0005
    assert corrected.phases[0] == "PLACE" and "RETREAT" in corrected.phases and corrected.phases[-1] == "HOLD"
    assert np.max(np.abs(np.diff(corrected.positions_rad, axis=0))) <= .12


def test_fitted_palm_frame_makes_anatomical_axes_horizontal():
    backend = TabletopWujiHand(viewer=False)
    try:
        quaternion = _fit_palm_down_quaternion(backend, FEASIBLE_POSE)
        backend.set_kinematic_pose(FEASIBLE_POSE, np.zeros(3), quaternion)
        roots = backend.long_finger_root_positions_m()
        tips = backend.fingertip_positions_m()
        assert abs((tips.mean(axis=0) - roots.mean(axis=0))[2]) <= 0.005
        assert abs((roots[-1] - roots[0])[2]) <= 0.005
    finally:
        backend.close()


def test_calibrated_place_has_palmar_side_below_dorsal_side():
    backend = TabletopWujiHand(viewer=False)
    try:
        quaternion = _calibrated_palm_down_quaternion(backend, FEASIBLE_POSE)
        backend.set_kinematic_pose(FEASIBLE_POSE, [0, 0, .2], quaternion)
        roots = backend.long_finger_root_positions_m()
        assert abs(roots.mean(axis=0)[2] - .2) <= .005
        assert backend.fingertip_positions_m()[:, 2].mean() < roots[:, 2].mean()
        assert (
            backend.palmar_reference_position_m()[2]
            < backend.dorsal_reference_position_m()[2]
        )
    finally:
        backend.close()


def test_prepare_pose_settles_to_first_table_contact_without_thumb_contact():
    backend = TabletopWujiHand(viewer=False)
    try:
        quaternion = _calibrated_palm_down_quaternion(backend, FEASIBLE_POSE)
        palm = _settle_to_table_contact(
            backend, FEASIBLE_POSE, np.array([0.0, 0.0, 0.2]), quaternion
        )
        backend.set_kinematic_pose(FEASIBLE_POSE, palm, quaternion)
        assert -0.0005 <= backend.minimum_hand_table_clearance_m() <= 0.0
        assert backend.thumb_position_m()[2] >= 0.010
    finally:
        backend.close()


def test_place_keeps_real_hand_geometry_above_table():
    backend = TabletopWujiHand(viewer=False)
    try:
        corrected, _ = build_table_retreat(
            _trajectory(),
            backend,
            TableRetreatConfig(
                place_duration_s=.1,
                retreat_duration_s=.2,
                hold_duration_s=.1,
                candidate_stride=1,
            ),
        )
        place_end = corrected.phases.index("RETREAT") - 1
        backend.set_kinematic_pose(
            corrected.positions_rad[place_end],
            corrected.palm_positions_m[place_end],
            corrected.palm_quaternions_wxyz[place_end],
        )
        assert backend.maximum_table_penetration_m() <= .0005
        assert backend.thumb_position_m()[2] >= .010
    finally:
        backend.close()


def test_retreat_releases_tip_contact_but_never_crosses_table():
    backend = TabletopWujiHand(viewer=False)
    try:
        corrected, report = build_table_retreat(
            _trajectory(),
            backend,
            TableRetreatConfig(
                place_duration_s=.1,
                retreat_duration_s=.2,
                hold_duration_s=.1,
                candidate_stride=1,
            ),
        )
        retreat = [
            index for index, phase in enumerate(corrected.phases)
            if phase == "RETREAT"
        ]
        heights = []
        for index in retreat:
            backend.set_kinematic_pose(
                corrected.positions_rad[index],
                corrected.palm_positions_m[index],
                corrected.palm_quaternions_wxyz[index],
            )
            assert backend.maximum_table_penetration_m() <= .0005
            heights.append(backend.fingertip_positions_m()[:, 2].copy())
        assert np.max(heights[-1] - heights[0]) > 0
        assert report.maximum_hand_penetration_m <= .0005
    finally:
        backend.close()


def test_explicit_source_frame_is_range_checked():
    backend = TabletopWujiHand(viewer=False)
    try:
        with pytest.raises(ValueError, match="outside trajectory"):
            build_table_retreat(_trajectory(), backend, TableRetreatConfig(source_frame=5, place_duration_s=.1, retreat_duration_s=.2, hold_duration_s=.1))
    finally:
        backend.close()


def test_recorded_workflow_preserves_samples_and_source_timing():
    backend = TabletopWujiHand(viewer=False)
    recording = _trajectory()
    try:
        corrected, report = build_recorded_table_retreat(
            recording, backend, source_kind="joint_states"
        )
    finally:
        backend.close()

    assert len(corrected.positions_rad) == len(recording.positions_rad)
    np.testing.assert_array_equal(
        np.diff(corrected.timestamps_ns), np.diff(recording.timestamps_ns)
    )
    assert report.source_kind == "joint_states"
    assert report.actual_retreat_m == pytest.approx(.03, abs=.002)
    assert report.palm_down_verified is True
    assert report.maximum_joint_correction_rad < .12


def test_recorded_workflow_never_zeros_or_replaces_recorded_pose():
    backend = TabletopWujiHand(viewer=False)
    recording = _trajectory()
    try:
        corrected, report = build_recorded_table_retreat(recording, backend)
    finally:
        backend.close()

    np.testing.assert_allclose(corrected.positions_rad, recording.positions_rad)
    assert report.maximum_hand_penetration_m <= .0005


def test_looped_retreat_defaults_to_three_forward_passes_and_two_resets():
    backend = TabletopWujiHand(viewer=False)
    try:
        corrected, _ = build_recorded_table_retreat(_trajectory(), backend)
        looped = build_looped_table_retreat(corrected, backend)
    finally:
        backend.close()

    assert set(looped.loop_indices) == {0, 1, 2}
    assert np.count_nonzero(np.diff(looped.loop_indices) > 0) == 2
    reset_runs = sum(
        phase == "RESET" and (index == 0 or looped.phases[index - 1] != "RESET")
        for index, phase in enumerate(looped.phases)
    )
    assert reset_runs == 2


def test_loop_reset_is_continuous_and_table_safe():
    backend = TabletopWujiHand(viewer=False)
    try:
        corrected, _ = build_recorded_table_retreat(_trajectory(), backend)
        looped = build_looped_table_retreat(corrected, backend, loops=2)
        assert np.max(np.abs(np.diff(looped.positions_rad, axis=0))) <= .12
        for joints, palm, quaternion in zip(
            looped.positions_rad,
            looped.palm_positions_m,
            looped.palm_quaternions_wxyz,
            strict=True,
        ):
            backend.set_kinematic_pose(joints, palm, quaternion)
            assert backend.maximum_table_penetration_m() <= .0005
            assert backend.thumb_position_m()[2] >= .010
    finally:
        backend.close()


def test_each_reset_transition_lasts_at_least_half_a_second():
    backend = TabletopWujiHand(viewer=False)
    try:
        corrected, _ = build_recorded_table_retreat(_trajectory(), backend)
        looped = build_looped_table_retreat(corrected, backend, loops=3)
    finally:
        backend.close()

    reset_runs = []
    start = None
    for index, phase in enumerate(looped.phases):
        if phase == "RESET" and start is None:
            start = index
        if phase != "RESET" and start is not None:
            reset_runs.append((start, index - 1))
            start = None
    assert len(reset_runs) == 2
    for start, end in reset_runs:
        duration_s = (end - start + 1) * backend.timestep_s
        assert duration_s >= .5


@pytest.mark.parametrize("loops", [0, -1, True, 1.5])
def test_looped_retreat_rejects_non_positive_integer_counts(loops):
    backend = TabletopWujiHand(viewer=False)
    try:
        corrected, _ = build_recorded_table_retreat(_trajectory(), backend)
        with pytest.raises(ValueError, match="positive integer"):
            build_looped_table_retreat(corrected, backend, loops=loops)
    finally:
        backend.close()


class RecordingTableBackend:
    timestep_s = .002
    has_viewer = False

    def __init__(self, *, close_after_steps=None):
        self.commands = []
        self.steps = []
        self.close_after_steps = close_after_steps

    def command_pose(self, joints, palm, quaternion):
        self.commands.append(np.asarray(joints).copy())

    def step(self, duration):
        self.steps.append(duration)

    def viewer_is_running(self):
        return self.close_after_steps is None or len(self.steps) < self.close_after_steps


def _looped_replay_fixture():
    return LoopedTableRetreat(
        timestamps_ns=np.array([0, 10_000_000, 20_000_000]),
        positions_rad=np.zeros((3, 20)),
        palm_positions_m=np.zeros((3, 3)),
        palm_quaternions_wxyz=np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)),
        phases=("PREPARE", "RETREAT", "HOLD"),
        loop_indices=(0, 0, 0),
    )


def test_replay_uses_cumulative_timestamps_not_one_step_per_frame():
    backend = RecordingTableBackend()

    summary = replay_table_retreat(_looped_replay_fixture(), backend)

    assert backend.steps == [.002] * 10
    assert len(backend.commands) == 3
    assert summary.scheduled_duration_s == pytest.approx(.02)
    assert summary.stopped_early is False


def test_replay_stops_when_viewer_closes_during_playback():
    backend = RecordingTableBackend(close_after_steps=3)
    backend.has_viewer = True

    summary = replay_table_retreat(_looped_replay_fixture(), backend)

    assert len(backend.steps) == 3
    assert len(backend.commands) < 3
    assert summary.loop_count == 0
    assert summary.stopped_early is True
