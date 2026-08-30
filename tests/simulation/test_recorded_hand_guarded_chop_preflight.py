from pathlib import Path

import mujoco
import numpy as np
import pytest

import twin_sim.tasks.recorded_hand_guarded_chop as module
from twin_sim.raised_work_surface import work_surface_height_m
from twin_sim.recorded_hand_guard import load_recorded_guard_cycle
from twin_sim.robot import RightArmRobot
from twin_sim.tasks.guarded_chop import _blade_bottom_height
from twin_sim.tasks.recorded_hand_guarded_chop import (
    RecordedHandGuardedChopConfig,
    _chopping_aligned_palm_rotation,
    _configure_table_only_scene,
    _lateral_guard_anchor_xy,
    _preflight_recorded_hand_guarded_chop,
    _select_clearance_candidate,
)


def _real_cycle():
    path = Path(__file__).resolve().parents[2] / Path(
        "recordings/wuji/august_02/"
        "session_20260802_174440_936_right_to_left_wuji_hand.mcap"
    )
    if not path.exists():
        pytest.skip("local ignored Wuji recording is unavailable")
    return load_recorded_guard_cycle(path)


def test_default_surface_search_actually_raises_the_board():
    config = RecordedHandGuardedChopConfig()
    assert config.surface_offsets_m[0] == pytest.approx(.04)
    assert config.surface_offsets_m[-1] == pytest.approx(.12)


def test_chopping_alignment_rotates_complete_palm_frame_minus_ninety_degrees():
    initial = _real_cycle().initial_palm_transform[:3, :3]
    quarter_turn = np.array(((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    expected = quarter_turn @ np.diag((-1.0, -1.0, 1.0)) @ initial

    np.testing.assert_allclose(
        _chopping_aligned_palm_rotation(initial), expected, atol=1e-12
    )


def test_lateral_guard_anchor_uses_left_arm_side_of_cut_line():
    np.testing.assert_allclose(
        _lateral_guard_anchor_xy(np.array((.62, -.08)), .12),
        (.54, .04),
    )


def test_table_only_scene_hides_props_without_changing_other_robot_instances():
    configured = RightArmRobot(viewer=False)
    untouched = RightArmRobot(viewer=False)
    hidden = (
        "guarded_chop_cube",
        "pick_source_pedestal",
        "pick_target_pedestal",
        "pick_cube_geom",
        "pick_target_region",
    )
    try:
        original = {
            name: (
                untouched.sim.model.geom_rgba[untouched.sim.require_geom(name)].copy(),
                int(untouched.sim.model.geom_contype[untouched.sim.require_geom(name)]),
                int(untouched.sim.model.geom_conaffinity[untouched.sim.require_geom(name)]),
            )
            for name in hidden
        }
        _configure_table_only_scene(configured)
        for name in hidden:
            geom = configured.sim.require_geom(name)
            assert configured.sim.model.geom_rgba[geom, 3] == 0.0
            assert configured.sim.model.geom_contype[geom] == 0
            assert configured.sim.model.geom_conaffinity[geom] == 0
            untouched_geom = untouched.sim.require_geom(name)
            np.testing.assert_array_equal(
                untouched.sim.model.geom_rgba[untouched_geom], original[name][0]
            )
            assert untouched.sim.model.geom_contype[untouched_geom] == original[name][1]
            assert untouched.sim.model.geom_conaffinity[untouched_geom] == original[name][2]
        board = configured.sim.require_geom("chopping_board")
        assert configured.sim.model.geom_rgba[board, 3] > 0.0
        assert configured.sim.model.geom_contype[board] != 0
    finally:
        configured.close()
        untouched.close()


def test_preflight_selects_lowest_feasible_shared_surface(monkeypatch):
    sentinel = object()

    def candidate(_robot, _cycle, _config, offset):
        if offset < .06:
            raise ValueError("left IK")
        return sentinel

    monkeypatch.setattr(module, "_build_candidate_plan", candidate)
    result = _preflight_recorded_hand_guarded_chop(
        object(), object(), RecordedHandGuardedChopConfig(surface_offsets_m=(.04, .05, .06))
    )
    assert result is sentinel


def test_clearance_selection_exhausts_preferred_tier_before_fallback():
    fallback_only = (.030, "fallback", (.015,))
    preferred = (.045, "preferred", (.021,))

    selected = _select_clearance_candidate(
        (fallback_only, preferred), preferred_m=.020, fallback_m=.010
    )
    assert selected == (.020, *preferred)

    selected_without_preferred = _select_clearance_candidate(
        (fallback_only,), preferred_m=.020, fallback_m=.010
    )
    assert selected_without_preferred == (.010, *fallback_only)


def test_real_plan_contains_five_safe_recorded_cycles_and_five_right_cuts():
    robot = RightArmRobot(viewer=False)
    try:
        plan = _preflight_recorded_hand_guarded_chop(
            robot,
            _real_cycle(),
            RecordedHandGuardedChopConfig(surface_offsets_m=(.04,)),
        )
        assert plan.surface_offset_m == pytest.approx(.04)
        assert len(plan.left_cycles) == 5
        assert len(plan.right_cuts) == 5
        assert len(plan.synchronized_cycles) == 5
        for synchronized in plan.synchronized_cycles:
            assert len(synchronized.right) == len(synchronized.left)
            assert len(synchronized.left) == len(synchronized.hand)
            assert len(synchronized.hand) == len(synchronized.knife_targets)
            assert "CUT_DOWN" in synchronized.knife_phases
            assert "KNIFE_RETRACT" in synchronized.knife_phases
        knife_y = np.concatenate(
            [
                synchronized.knife_targets[:, 1, 3]
                for synchronized in plan.synchronized_cycles
            ]
        )
        assert np.min(np.diff(knife_y)) >= -.0015 - 1e-9, np.min(
            np.diff(knife_y)
        )
        assert .010 <= knife_y[-1] - knife_y[0] <= .040
        assert len(plan.left_transitions) == 0
        assert sum(len(value.hand) for value in plan.left_cycles) == len(
            _real_cycle().hand_positions_rad
        )
        assert all(len(value.hand) == len(value.left) for value in plan.left_cycles)
        assert plan.selected_clearance_tier_m in (.020, .010)
        assert plan.minimum_planned_distance_m >= plan.selected_clearance_tier_m
        assert plan.selected_lateral_spacing_m >= .030
        assert plan.minimum_lateral_spacing_m >= (
            plan.selected_lateral_spacing_m - .003
        )
        assert plan.maximum_lateral_spacing_m <= (
            plan.selected_lateral_spacing_m + .003
        )
        assert plan.maximum_depth_mismatch_m <= .010
        assert plan.maximum_hand_penetration_m <= .0005
        assert plan.minimum_thumb_clearance_m >= .002
        assert np.all(plan.minimum_long_pad_clearances_m >= -1e-6)
        assert np.all(plan.minimum_long_pad_clearances_m <= .005)
        cycle_starts_y = np.asarray(
            [value.palm_targets[0, 1, 3] for value in plan.left_cycles]
        )
        assert np.all(np.diff(cycle_starts_y) >= 0.0)
        total_wrist_retreat = (
            plan.left_cycles[-1].palm_targets[-1, 1, 3]
            - plan.left_cycles[0].palm_targets[0, 1, 3]
        )
        assert .025 <= total_wrist_retreat <= .040
        assert plan.minimum_pad_step_y_m >= -.0005 - 1e-9
        assert np.all(plan.pad_net_retreats_m > 0.0)
        planned_left = np.concatenate([value.left for value in plan.left_cycles])
        planned_hand = np.concatenate([value.hand for value in plan.left_cycles])
        planned_phases = np.asarray(
            sum((value.phases for value in plan.left_cycles), ())
        )
        active = planned_phases != "PREPARE"
        distal_angles = []
        distal_bodies = [
            robot.sim.require_body(f"left_finger{finger}_link4")
            for finger in range(2, 6)
        ]
        for left_rad, hand_rad in zip(planned_left, planned_hand, strict=True):
            robot.sim.data.qpos[robot.sim.left.qpos_ids] = left_rad
            robot.sim.data.qpos[robot.sim.hand.qpos_ids] = hand_rad
            mujoco.mj_forward(robot.sim.model, robot.sim.data)
            axes = robot.sim.data.xmat[distal_bodies].reshape(-1, 3, 3)[:, :, 2]
            distal_angles.append(
                np.degrees(
                    np.arccos(np.clip(axes @ np.array((0.0, 0.0, -1.0)), -1.0, 1.0))
                )
            )
        active_distal_angles = np.asarray(distal_angles)[active]
        assert np.min(active_distal_angles) >= 20.0
        assert np.max(active_distal_angles) <= 50.0
        pip_ranges = np.ptp(planned_hand[:, (6, 10, 14, 18)], axis=0)
        dip_ranges = np.ptp(planned_hand[:, (7, 11, 15, 19)], axis=0)
        previous_pip_ranges = np.asarray((.384813, .260536, .310028, .358330))
        assert np.all(pip_ranges >= .75 * previous_pip_ranges)
        assert np.all(pip_ranges <= .85 * previous_pip_ranges)
        assert np.all(dip_ranges >= .25)
        assert np.all(dip_ranges <= .50)
        assert np.corrcoef(planned_hand[:, 10], planned_hand[:, 14])[0, 1] > 0.0
        assert not np.array_equal(planned_hand[:, 6], planned_hand[:, 10])
        assert np.max(np.abs(np.diff(planned_hand, axis=0))) <= .12
        board_top = work_surface_height_m(robot.sim)
        for cut, left_cycle in zip(
            plan.right_cuts, plan.left_cycles, strict=True
        ):
            hand_start = left_cycle.palm_targets[0, :2, 3]
            hand_end = left_cycle.palm_targets[-1, :2, 3]
            assert hand_end[1] > hand_start[1]
            blade_bottom = _blade_bottom_height(
                robot, cut.descent[-1].joints_rad
            )
            assert abs(blade_bottom - board_top) <= .01
        synchronized_knife_y = np.concatenate(
            [cycle.knife_targets[:, 1, 3] for cycle in plan.synchronized_cycles]
        )
        synchronized_palm_y = np.concatenate(
            [cycle.palm_targets[:, 1, 3] for cycle in plan.synchronized_cycles]
        )
        assert synchronized_knife_y[-1] > synchronized_knife_y[0]
        assert synchronized_palm_y[-1] - synchronized_palm_y[0] == pytest.approx(
            .032, abs=1e-6
        )
        cut_direction = (
            plan.right_cuts[-1].descent[-1].target_pose[:2, 3]
            - plan.right_cuts[0].descent[-1].target_pose[:2, 3]
        )
        cut_direction /= np.linalg.norm(cut_direction)
        for value in plan.left_cycles:
            retreat = value.palm_targets[-1, :2, 3] - value.palm_targets[0, :2, 3]
            retreat_direction = retreat / np.linalg.norm(retreat)
            angular_error = np.arccos(
                np.clip(np.dot(retreat_direction, cut_direction), -1.0, 1.0)
            )
            assert angular_error <= np.deg2rad(1.0)
            assert retreat[1] > 0.0
            assert abs(retreat[0]) <= .0005
        maximum_hand_step = max(
            np.max(np.abs(np.diff(value.hand, axis=0))) for value in plan.left_cycles
        )
        assert maximum_hand_step <= .12
    finally:
        robot.close()
