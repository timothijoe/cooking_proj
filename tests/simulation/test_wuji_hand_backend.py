import numpy as np
import pytest

from twin_sim.wuji_hand_backend import SimWujiHand


def test_backend_uses_sdk_shape_and_starts_enabled():
    hand = SimWujiHand()
    assert hand.read_joint_actual_position().shape == (5, 4)
    assert hand.read_joint_target_position().shape == (5, 4)
    np.testing.assert_array_equal(
        hand.read_joint_enabled(), np.ones((5, 4), dtype=bool)
    )
    hand.close()


def test_target_write_is_atomic_and_steps_simulation():
    hand = SimWujiHand()
    target = hand.read_joint_target_position()
    target[0, 1] += 0.1
    hand.write_joint_target_position(target)
    before = hand.read_joint_actual_position()[0, 1]
    for _ in range(10):
        hand.step(0.01)
    after = hand.read_joint_actual_position()[0, 1]
    assert after > before
    np.testing.assert_array_equal(hand.read_joint_target_position(), target)

    accepted = hand.read_joint_target_position()
    with pytest.raises(ValueError, match=r"shape \(5, 4\)"):
        hand.write_joint_target_position(np.zeros(20))
    np.testing.assert_array_equal(hand.read_joint_target_position(), accepted)
    hand.close()


def test_disabled_backend_rejects_new_commands_and_holds_target():
    hand = SimWujiHand()
    accepted = hand.read_joint_target_position()
    hand.write_joint_enabled(False)
    np.testing.assert_array_equal(
        hand.read_joint_enabled(), np.zeros((5, 4), dtype=bool)
    )
    with pytest.raises(RuntimeError, match="disabled"):
        hand.write_joint_target_position(accepted + 0.01)
    np.testing.assert_array_equal(hand.read_joint_target_position(), accepted)
    hand.step(0.01)
    hand.write_joint_enabled(True)
    hand.write_joint_target_position(accepted)
    hand.close()


def test_enabled_joints_remain_commandable_when_one_joint_is_disabled():
    hand = SimWujiHand()
    enabled = hand.read_joint_enabled()
    enabled[0, 0] = False
    hand.write_joint_enabled(enabled)
    target = hand.read_joint_target_position()
    target[0, 1] += 0.05
    hand.write_joint_target_position(target)
    target[0, 0] += 0.05
    with pytest.raises(RuntimeError, match="disabled"):
        hand.write_joint_target_position(target)
    hand.close()


def test_realtime_controller_context_exposes_supported_subset():
    hand = SimWujiHand()
    with hand.realtime_controller() as controller:
        target = hand.read_joint_target_position()
        controller.set_joint_target_position(target)
        assert controller.get_joint_actual_position().shape == (5, 4)
        effort = controller.get_joint_actual_effort()
        assert effort.shape == (5, 4)
        assert np.isfinite(effort).all()
    with pytest.raises(RuntimeError, match="closed"):
        controller.get_joint_actual_position()
    hand.close()


def test_unsupported_sdk_method_is_explicit():
    hand = SimWujiHand()
    with pytest.raises(NotImplementedError, match="not simulated"):
        hand.read_joint_temperature()
    with pytest.raises(NotImplementedError, match="not simulated"):
        hand.read_input_voltage()
    hand.close()


@pytest.mark.parametrize("invalid", ("false", 1, np.ones((5, 4), dtype=int)))
def test_enabled_state_requires_boolean_values(invalid):
    hand = SimWujiHand()
    with pytest.raises(ValueError, match="boolean"):
        hand.write_joint_enabled(invalid)
    hand.close()
