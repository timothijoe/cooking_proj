import numpy as np
import pytest

import twin_sim.tasks.guarded_chop as guarded_chop
from twin_sim.guarded_chop_safety import (
    CAT_PAW_RAD,
    GUARD_RELAXED_RAD,
    GUARD_RETRACTED_RAD,
    SafetyDecision,
)
from twin_sim.robot import RightArmRobot
from twin_sim.tasks.guarded_chop import (
    GuardedChopConfig,
    GuardedChopPhase,
    _preflight_guarded_chop,
    run_guarded_chop,
)


def test_plane_mode_retreats_guard_before_diagonal_knife_shift():
    robot = RightArmRobot(viewer=False)
    try:
        plan = _preflight_guarded_chop(robot, GuardedChopConfig())
    finally:
        robot.close()
    result = run_guarded_chop(
        GuardedChopConfig(final_hold_s=0.0), viewer=False
    )

    assert result.success, result.reason
    assert result.final_phase is GuardedChopPhase.COMPLETE
    assert result.completed_cuts == 5
    assert result.completed_shifts == 4
    assert np.isclose(result.total_shift_m, 0.08, atol=0.002)
    assert result.minimum_distance_m >= 0.02
    assert max(
        sample.guard_cube_contact_count for sample in result.samples
    ) == 0
    assert max(
        sample.guard_cube_penetration_m for sample in result.samples
    ) <= 0.003
    assert max(
        sample.guard_cube_normal_force_n for sample in result.samples
    ) <= 35.0
    assert all(
        not sample.cut_allowed
        for sample in result.samples
        if sample.phase is not GuardedChopPhase.CUT_DOWN
    )
    assert max(
        sample.blade_hand_contact_count for sample in result.samples
    ) == 0
    assert max(
        sample.left_speed_rad_s
        for sample in result.samples
        if sample.phase is GuardedChopPhase.CUT_DOWN
    ) <= 0.05
    assert max(
        sample.hand_speed_rad_s
        for sample in result.samples
        if sample.phase is GuardedChopPhase.CUT_DOWN
    ) <= 0.05
    down = [
        sample
        for sample in result.samples
        if sample.phase is GuardedChopPhase.CUT_DOWN
    ]
    assert min(sample.knife_height_m for sample in down) >= 0.328
    for index in range(1, 6):
        cut = [sample for sample in down if sample.cut_index == index]
        targets = np.asarray([sample.left_target_rad for sample in cut])
        assert np.max(np.ptp(targets, axis=0)) <= 1e-12

    clear = [
        sample for sample in result.samples
        if sample.phase is GuardedChopPhase.KNIFE_CLEAR
    ]
    assert clear
    assert max(sample.left_speed_rad_s for sample in clear) <= 0.05
    assert max(sample.hand_speed_rad_s for sample in clear) <= 0.05

    guard_ready = [
        sample
        for sample in result.samples
        if sample.phase is GuardedChopPhase.GUARD_READY
    ]
    guard_endpoints = [guard_ready[-1].guard_position]
    for index in range(1, 5):
        expected_phase = (
            GuardedChopPhase.FINGER_RETRACT
            if index % 2 == 1
            else GuardedChopPhase.ARM_RESET_RELAX
        )
        guard_motion = [
            sample
            for sample in result.samples
            if sample.cut_index == index
            and sample.phase is expected_phase
        ]
        lift = [
            sample
            for sample in result.samples
            if sample.cut_index == index
            and sample.phase is GuardedChopPhase.KNIFE_LIFT_SHIFT
        ]
        assert guard_motion and lift
        assert guard_motion[-1].time_s < lift[0].time_s
        low = tuple(guard_motion)
        right_targets = np.asarray(
            [sample.right_target_rad for sample in low]
        )
        assert np.max(np.ptp(right_targets, axis=0)) <= 1e-12
        assert max(sample.right_speed_rad_s for sample in low) <= 0.05
        assert guard_motion[-1].knife_guard_distance_m >= (
            guard_motion[0].knife_guard_distance_m - 2e-6
        )
        left_motion = np.asarray(
            [sample.left_target_rad for sample in guard_motion]
        )
        hand_motion = np.asarray(
            [sample.hand_target_rad for sample in guard_motion]
        )
        if index % 2 == 1:
            assert np.max(np.ptp(left_motion, axis=0)) <= 1e-12
            assert np.max(np.ptp(hand_motion, axis=0)) > 0.05
            np.testing.assert_allclose(
                guard_motion[-1].hand_target_rad,
                GUARD_RETRACTED_RAD,
                atol=1e-9,
            )
        else:
            assert np.max(np.ptp(left_motion, axis=0)) > 0.01
            assert np.max(np.ptp(hand_motion, axis=0)) > 0.05
            np.testing.assert_allclose(
                guard_motion[-1].hand_target_rad,
                GUARD_RELAXED_RAD,
                atol=1e-9,
            )
        guard_endpoints.append(guard_motion[-1].guard_position)

        left_targets = np.asarray(
            [sample.left_target_rad for sample in lift]
        )
        hand_targets = np.asarray(
            [sample.hand_target_rad for sample in lift]
        )
        assert np.max(np.ptp(left_targets, axis=0)) <= 1e-12
        assert np.max(np.ptp(hand_targets, axis=0)) <= 1e-12
        assert max(sample.left_speed_rad_s for sample in lift) <= 0.05
        assert max(sample.hand_speed_rad_s for sample in lift) <= 0.05
        knife_positions = np.asarray(
            [sample.knife_position for sample in lift]
        )
        assert np.ptp(knife_positions[:, 2]) > 0.05
        assert np.linalg.norm(
            knife_positions[-1, :2] - knife_positions[0, :2]
        ) > 0.01

    advances = np.diff(np.asarray(guard_endpoints), axis=0)
    advance_lengths = np.linalg.norm(advances, axis=1)
    assert np.allclose(advance_lengths, 0.02, atol=0.005)
    assert np.linalg.norm(
        np.asarray(guard_endpoints[-1]) - np.asarray(guard_endpoints[0])
    ) == pytest.approx(0.08, abs=0.005)
    direction = advances[0] / np.linalg.norm(advances[0])
    assert np.all(advances @ direction >= 0.015)


def test_plane_scene_keeps_guard_cube_hidden_and_collision_disabled():
    robot = RightArmRobot(viewer=False)
    try:
        configure = getattr(
            guarded_chop, "_configure_guarded_scene", None
        )
        assert callable(configure)
        configure(robot, "plane")
        cube = robot.sim.require_geom("guarded_chop_cube")
        cube_body = int(robot.sim.model.geom_bodyid[cube])

        assert robot.sim.model.geom_rgba[cube, 3] == 0.0
        assert robot.sim.model.geom_contype[cube] == 0
        assert robot.sim.model.geom_conaffinity[cube] == 0
        assert robot.sim.model.body_contype[cube_body] == 0
        assert robot.sim.model.body_conaffinity[cube_body] == 0
    finally:
        robot.close()


def test_object_preflight_keeps_conservative_serial_trajectory(monkeypatch):
    def reject_plane_coupling(*args, **kwargs):
        raise AssertionError("object preflight used plane coupling")

    monkeypatch.setattr(
        guarded_chop, "_resample_trajectory", reject_plane_coupling
    )
    robot = RightArmRobot(viewer=False)
    try:
        plan = _preflight_guarded_chop(
            robot, GuardedChopConfig(scene_mode="object")
        )
    finally:
        robot.close()

    assert len(plan.guard_shifts) == 4


def test_object_mode_retains_contact_latched_hand():
    robot = RightArmRobot(viewer=False)
    try:
        plan = _preflight_guarded_chop(
            robot, GuardedChopConfig(scene_mode="object")
        )
    finally:
        robot.close()
    result = run_guarded_chop(
        GuardedChopConfig(scene_mode="object", final_hold_s=0.0),
        viewer=False,
    )

    assert result.success, result.reason
    assert max(
        sample.guard_cube_contact_count for sample in result.samples
    ) >= 1
    shifts = [
        sample
        for sample in result.samples
        if sample.phase is GuardedChopPhase.LOW_GUARD_SHIFT
    ]
    assert shifts
    assert max(
        np.linalg.norm(sample.hand_target_rad - CAT_PAW_RAD)
        for sample in shifts
    ) > 1e-3
    for index in range(1, 5):
        clear = [
            sample for sample in result.samples
            if sample.cut_index == index
            and sample.phase is GuardedChopPhase.KNIFE_CLEAR
        ]
        origin_xy = clear[0].knife_position[:2]
        lateral = [
            sample for sample in clear
            if np.linalg.norm(sample.knife_position[:2] - origin_xy) > 0.001
        ]
        assert lateral
        assert min(sample.knife_height_m for sample in lateral) >= (
            plan.safe_knife_height_m
        )


def test_rejected_safety_decision_aborts_both_arm_sequence():
    class RejectCoordinator:
        minimum_distance_m = 0.02

        def evaluate(self, observation):
            return SafetyDecision(False, "injected safety rejection")

    result = run_guarded_chop(
        GuardedChopConfig(final_hold_s=0.0),
        viewer=False,
        coordinator=RejectCoordinator(),
    )

    assert not result.success
    assert result.final_phase is GuardedChopPhase.ABORTED
    assert result.completed_cuts == 0
    assert result.completed_shifts == 0
    assert result.reason == "injected safety rejection"


def test_replay_requires_viewer():
    with pytest.raises(ValueError, match="replay requires a Viewer"):
        run_guarded_chop(
            GuardedChopConfig(final_hold_s=0.0),
            viewer=False,
            replay_rate=2.0,
        )


def test_record_path_saves_complete_state_sequence(monkeypatch, tmp_path):
    saved = []

    def save(recording, path):
        saved.append((recording, path))

    monkeypatch.setattr(
        "twin_sim.guarded_chop_recording.save_recording", save
    )
    target = tmp_path / "guarded.npz"

    result = run_guarded_chop(
        GuardedChopConfig(final_hold_s=0.0),
        viewer=False,
        record_path=target,
    )

    assert result.success, result.reason
    assert len(saved) == 1
    recording, path = saved[0]
    assert path == target
    assert len(recording.frames) == len(result.samples)
    assert recording.frames[-1].phase == "complete"
