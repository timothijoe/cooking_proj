import numpy as np
import pytest

from tianji_robotics.hardware.wuji_hand.sdk import SdkWujiHand


class FakeRuntime:
    def __init__(self, fail_command=False):
        self.calls = []
        self.fail_command = fail_command

    def connect(self): self.calls.append("connect")
    def arm(self): self.calls.append("arm")
    def command_position_rad(self, target):
        self.calls.append(("command", target.copy()))
        if self.fail_command: raise OSError("transport")
    def disarm(self): self.calls.append("disarm")
    def close(self): self.calls.append("close")


def test_constructing_sdk_backend_does_not_connect():
    runtime = FakeRuntime()
    SdkWujiHand(runtime)
    assert runtime.calls == []


def test_command_requires_connect_and_arm():
    hand = SdkWujiHand(FakeRuntime())
    with pytest.raises(RuntimeError, match="not connected"):
        hand.command_position_rad(np.zeros(20))
    hand.connect()
    with pytest.raises(RuntimeError, match="not armed"):
        hand.command_position_rad(np.zeros(20))


def test_guarded_lifecycle_and_idempotent_close():
    runtime = FakeRuntime()
    hand = SdkWujiHand(runtime)
    hand.connect(); hand.arm(); hand.command_position_rad(np.zeros(20)); hand.disarm(); hand.close(); hand.close()
    assert [call if isinstance(call, str) else call[0] for call in runtime.calls] == ["connect", "arm", "command", "disarm", "close"]


@pytest.mark.parametrize("target", [np.zeros(19), np.full(20, np.nan)])
def test_command_validates_target_before_runtime(target):
    runtime = FakeRuntime(); hand = SdkWujiHand(runtime); hand.connect(); hand.arm()
    with pytest.raises(ValueError, match="20 finite"):
        hand.command_position_rad(target)
    assert runtime.calls == ["connect", "arm"]


def test_command_failure_disarms_before_propagating():
    runtime = FakeRuntime(fail_command=True); hand = SdkWujiHand(runtime); hand.connect(); hand.arm()
    with pytest.raises(OSError, match="transport"):
        hand.command_position_rad(np.zeros(20))
    assert runtime.calls[-1] == "disarm"
