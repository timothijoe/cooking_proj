import pytest

from twin_sim.gamepad_teleop import TeleopConfig, run_gamepad_teleop, shaped_axis


def test_deadzone_suppresses_small_axis_noise():
    assert shaped_axis(3000, 0.15) == 0.0
    assert shaped_axis(-3000, 0.15) == 0.0


def test_axis_is_normalized_after_deadzone():
    assert shaped_axis(32767, 0.15) == pytest.approx(1.0)
    assert shaped_axis(-32767, 0.15) == pytest.approx(-1.0)


def test_rejects_speed_above_simulation_limit_before_opening_device():
    with pytest.raises(ValueError, match="speed-mm-s"):
        run_gamepad_teleop(TeleopConfig(speed_mm_s=301.0))


def test_rejects_workspace_radius_above_simulation_limit_before_opening_device():
    with pytest.raises(ValueError, match="workspace-radius-mm"):
        run_gamepad_teleop(TeleopConfig(workspace_radius_mm=350.1))


def test_rejects_unknown_arm_before_opening_device():
    with pytest.raises(ValueError, match="arm must"):
        run_gamepad_teleop(TeleopConfig(arm="both"))  # type: ignore[arg-type]


def test_rejects_orientation_rate_above_limit():
    with pytest.raises(ValueError, match="orientation-rate-dps"):
        run_gamepad_teleop(TeleopConfig(orientation_rate_dps=91.0))


def test_rejects_orientation_rate_zero():
    with pytest.raises(ValueError, match="orientation-rate-dps"):
        run_gamepad_teleop(TeleopConfig(orientation_rate_dps=0.0))


def test_default_orientation_rate_is_valid():
    # Default config (30.0 dps) should pass validation
    assert TeleopConfig().orientation_rate_dps == 30.0
