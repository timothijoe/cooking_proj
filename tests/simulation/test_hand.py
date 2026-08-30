import numpy as np
import pytest

from twin_sim.hand import DEFAULT_OPEN_RAD, LeftHandController
from twin_sim.model import SimulationModel


def test_hand_controller_applies_complete_target():
    sim = SimulationModel.load()
    hand = LeftHandController(sim)
    target = sim.model.actuator_ctrlrange[sim.hand.actuator_ids].mean(axis=1)
    hand.command(target)
    hand.apply()
    np.testing.assert_array_equal(hand.target, target)
    np.testing.assert_array_equal(sim.data.ctrl[sim.hand.actuator_ids], target)


def test_hand_open_restores_natural_pose():
    sim = SimulationModel.load()
    hand = LeftHandController(sim)
    hand.command(sim.model.actuator_ctrlrange[sim.hand.actuator_ids].mean(axis=1))
    hand.open()
    hand.apply()
    np.testing.assert_array_equal(hand.target, DEFAULT_OPEN_RAD)
    np.testing.assert_array_equal(
        sim.data.ctrl[sim.hand.actuator_ids], DEFAULT_OPEN_RAD
    )


@pytest.mark.parametrize(
    "invalid",
    (
        np.zeros(19),
        np.r_[np.nan, np.zeros(19)],
    ),
)
def test_invalid_hand_target_is_atomic(invalid):
    sim = SimulationModel.load()
    hand = LeftHandController(sim)
    before_ctrl = sim.data.ctrl.copy()
    before_target = hand.target.copy()
    with pytest.raises(ValueError):
        hand.command(invalid)
    np.testing.assert_array_equal(sim.data.ctrl, before_ctrl)
    np.testing.assert_array_equal(hand.target, before_target)


def test_out_of_range_hand_target_is_atomic():
    sim = SimulationModel.load()
    hand = LeftHandController(sim)
    invalid = DEFAULT_OPEN_RAD.copy()
    invalid[3] = sim.model.actuator_ctrlrange[sim.hand.actuator_ids[3], 1] + 0.1
    before_ctrl = sim.data.ctrl.copy()
    before_target = hand.target.copy()
    with pytest.raises(ValueError, match="joint 4 target"):
        hand.command(invalid)
    np.testing.assert_array_equal(sim.data.ctrl, before_ctrl)
    np.testing.assert_array_equal(hand.target, before_target)
