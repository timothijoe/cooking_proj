import numpy as np
import pytest

from tianji_arm.experiments.gamepad_cartesian_jog import (
    GamepadJogConfig,
    ARM_TO_SDK,
    gamepad_delta_mm,
    parse_args,
    validate_config,
)


def test_default_arm_is_right_sdk_b():
    assert GamepadJogConfig().arm == "right"
    assert ARM_TO_SDK[GamepadJogConfig().arm] == "B"


def test_left_option_maps_to_sdk_a():
    assert parse_args(["--arm", "left"]).arm == "left"
    assert ARM_TO_SDK["left"] == "A"


def test_full_deflection_requests_one_bounded_increment():
    config = GamepadJogConfig(speed_mm_s=10.0, control_period_s=0.2)
    np.testing.assert_allclose(gamepad_delta_mm({0: 32767}, config), [2.0, 0.0, 0.0])


def test_deadzone_suppresses_noise():
    assert not np.any(gamepad_delta_mm({0: 3000, 1: -3000, 4: 3000}, GamepadJogConfig()))


def test_rejects_unsafe_physical_speed_before_connection():
    with pytest.raises(ValueError, match="speed-mm-s"):
        validate_config(GamepadJogConfig(speed_mm_s=10.1))
