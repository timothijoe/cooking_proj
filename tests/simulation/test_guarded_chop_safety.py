import pytest

from twin_sim.guarded_chop_safety import (
    SafetyCoordinator,
    SafetyObservation,
)


def observation(**overrides):
    values = dict(
        phase="CUT_DOWN",
        knife_height_m=0.36,
        safe_knife_height_m=0.34,
        knife_guard_distance_m=0.03,
        left_target_stationary=True,
        right_target_stationary=False,
        left_speed_rad_s=0.0,
        right_speed_rad_s=0.0,
        hand_speed_rad_s=0.0,
        guard_cube_penetration_m=0.0,
        guard_cube_normal_force_n=0.0,
        finite_state=True,
    )
    values.update(overrides)
    return SafetyObservation(**values)


def test_cut_requires_stationary_guard_and_clearance():
    coordinator = SafetyCoordinator(minimum_distance_m=0.02)

    assert coordinator.evaluate(observation()).allowed
    assert not coordinator.evaluate(
        observation(left_target_stationary=False)
    ).allowed
    decision = coordinator.evaluate(
        observation(knife_guard_distance_m=0.019)
    )
    assert not decision.allowed
    assert "distance" in decision.reason


def test_knife_clear_rejects_guard_motion_below_safe_height():
    coordinator = SafetyCoordinator()

    decision = coordinator.evaluate(
        observation(
            phase="KNIFE_CLEAR",
            knife_height_m=0.33,
            safe_knife_height_m=0.34,
            left_target_stationary=False,
            left_speed_rad_s=0.2,
            hand_speed_rad_s=0.2,
        )
    )

    assert not decision.allowed
    assert "guard moved before knife clearance" in decision.reason


@pytest.mark.parametrize(
    "phase",
    (
        "LOW_GUARD_OPEN",
        "LOW_GUARD_SHIFT",
        "LOW_GUARD_CLOSE",
        "GUARD_SETTLE",
        "FINGER_RETRACT",
        "ARM_RESET_RELAX",
    ),
)
def test_low_guard_motion_requires_stationary_knife(phase):
    coordinator = SafetyCoordinator()
    allowed = observation(
        phase=phase,
        knife_height_m=0.33,
        right_target_stationary=True,
        left_target_stationary=False,
        left_speed_rad_s=0.2,
        right_speed_rad_s=0.0,
    )

    assert coordinator.evaluate(allowed).allowed
    decision = coordinator.evaluate(
        observation(
            phase=phase,
            knife_height_m=0.33,
            right_target_stationary=False,
            left_target_stationary=False,
            left_speed_rad_s=0.2,
            right_speed_rad_s=0.2,
        )
    )
    assert not decision.allowed
    assert "knife moved during low guard shift" in decision.reason


def test_lift_shift_requires_stationary_guard():
    decision = SafetyCoordinator().evaluate(
        observation(
            phase="KNIFE_LIFT_SHIFT",
            left_target_stationary=False,
            left_speed_rad_s=0.2,
            hand_speed_rad_s=0.2,
        )
    )

    assert not decision.allowed
    assert "guard moved during knife lift shift" in decision.reason


def test_nonfinite_state_is_rejected():
    decision = SafetyCoordinator().evaluate(
        observation(finite_state=False)
    )

    assert not decision.allowed
    assert "non-finite" in decision.reason


def test_excessive_guard_contact_is_rejected():
    coordinator = SafetyCoordinator()

    assert not coordinator.evaluate(
        observation(guard_cube_penetration_m=0.0031)
    ).allowed
    assert not coordinator.evaluate(
        observation(guard_cube_normal_force_n=35.1)
    ).allowed


def test_actual_guard_motion_blocks_cut_phase():
    coordinator = SafetyCoordinator()

    assert not coordinator.evaluate(
        observation(left_speed_rad_s=0.06)
    ).allowed
    assert not coordinator.evaluate(
        observation(hand_speed_rad_s=0.06)
    ).allowed
