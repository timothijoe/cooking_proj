import sys
from types import ModuleType

import numpy as np
import pytest

from tianji_robotics.wuji_sdk.retargeter import OfficialWujiRetargeter


def fake_wuji_sdk_module(command=None):
    module = ModuleType("wuji_sdk")
    module.calls = []

    class HandModel:
        WujiHand = object()

    class Handedness:
        Left = object()

    class RetargetSession:
        @classmethod
        def for_hand(cls, hand_model, *, side):
            module.calls.append((hand_model, side))
            return cls()

        def step(self, keypoints_m):
            if command is not None:
                return command
            return np.arange(20, dtype=np.float32)

    module.HandModel = HandModel
    module.Handedness = Handedness
    module.RetargetSession = RetargetSession
    return module


def test_adapter_uses_left_first_generation_without_device_manager(monkeypatch):
    fake = fake_wuji_sdk_module()
    monkeypatch.setitem(sys.modules, "wuji_sdk", fake)

    adapter = OfficialWujiRetargeter.create_left_first_generation()
    result = adapter.step(np.zeros((21, 3), dtype=np.float32))

    assert result.shape == (20,)
    assert fake.calls == [(fake.HandModel.WujiHand, fake.Handedness.Left)]


def test_adapter_rejects_invalid_sdk_command(monkeypatch):
    monkeypatch.setitem(sys.modules, "wuji_sdk", fake_wuji_sdk_module(np.ones(19)))
    adapter = OfficialWujiRetargeter.create_left_first_generation()

    with pytest.raises(ValueError, match="20"):
        adapter.step(np.zeros((21, 3), dtype=np.float32))


def test_adapter_rejects_values_that_overflow_float64(monkeypatch):
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("platform longdouble is not wider than float64")
    command = np.zeros(20, dtype=np.longdouble)
    command[0] = np.longdouble(str(np.finfo(np.float64).max)) * 2
    monkeypatch.setitem(sys.modules, "wuji_sdk", fake_wuji_sdk_module(command))
    adapter = OfficialWujiRetargeter.create_left_first_generation()

    with pytest.raises(ValueError, match="non-finite"):
        adapter.step(np.zeros((21, 3), dtype=np.float32))
