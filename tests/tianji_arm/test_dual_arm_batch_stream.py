from __future__ import annotations

import numpy as np
import pytest

from tianji_arm.experiments.dual_arm_batch_stream import (
    _configure_dual_joint_impedance,
    build_dual_arm_entry,
    build_hold_frames,
    dual_arms_ready_issue,
    disable_both_arms,
    validate_dual_feedback,
    parse_args,
    send_dual_arm_frame,
)


class FakeRobot:
    def __init__(self, *, send_result: object = True) -> None:
        self.send_result = send_result
        self.calls: list[tuple] = []

    def clear_set(self):
        self.calls.append(("clear_set",))

    def set_joint_cmd_pose(self, *, arm: str, joints: list[float]):
        self.calls.append(("set_joint_cmd_pose", arm, joints))

    def send_cmd(self):
        self.calls.append(("send_cmd",))
        return self.send_result


def test_batch_frame_stages_both_arms_then_sends_once():
    robot = FakeRobot()
    a = np.arange(7, dtype=float)
    b = np.arange(10, 17, dtype=float)

    send_dual_arm_frame(robot, a, b)

    assert robot.calls == [
        ("clear_set",),
        ("set_joint_cmd_pose", "A", a.tolist()),
        ("set_joint_cmd_pose", "B", b.tolist()),
        ("send_cmd",),
    ]


def test_batch_frame_rejects_non_finite_or_wrong_sized_targets():
    with pytest.raises(ValueError, match="seven finite"):
        send_dual_arm_frame(FakeRobot(), np.zeros(6), np.zeros(7))
    with pytest.raises(ValueError, match="seven finite"):
        send_dual_arm_frame(FakeRobot(), np.zeros(7), np.full(7, np.nan))


def test_batch_frame_fails_closed_when_the_controller_rejects_send():
    with pytest.raises(RuntimeError, match="rejected"):
        send_dual_arm_frame(FakeRobot(send_result=False), np.zeros(7), np.zeros(7))


def test_batch_frame_accepts_the_lower_sdk_integer_success_code():
    send_dual_arm_frame(FakeRobot(send_result=1), np.zeros(7), np.zeros(7))


def test_dual_arm_entry_has_exact_endpoints_for_both_arms():
    entry_a, entry_b = build_dual_arm_entry(
        np.zeros(7), np.ones(7), np.full(7, 2.0), np.full(7, 3.0), duration_s=1.0, control_hz=10.0
    )

    assert entry_a.shape == (11, 7)
    assert entry_b.shape == (11, 7)
    assert np.array_equal(entry_a[0], np.zeros(7))
    assert np.array_equal(entry_a[-1], np.ones(7))
    assert np.array_equal(entry_b[0], np.full(7, 2.0))
    assert np.array_equal(entry_b[-1], np.full(7, 3.0))


def test_probe_current_is_an_explicit_execute_mode():
    args = parse_args(["--execute", "--probe-current"])

    assert args.execute
    assert args.probe_current


def test_both_arms_ready_check_defaults_to_enabled_and_can_be_disabled_explicitly():
    assert parse_args([]).require_both_arms_ready is True
    assert parse_args(["--require-both-arms-ready=false"]).require_both_arms_ready is False


def test_both_arms_ready_check_reports_b_arm_error_before_streaming():
    feedback = {
        "states": [{"cur_state": 3, "err_code": 0}, {"cur_state": 3, "err_code": 6}],
        "outputs": [
            {"fb_joint_pos": [0.0] * 7},
            {"fb_joint_pos": [0.0] * 7},
        ],
    }

    assert dual_arms_ready_issue(feedback) == "B-arm controller error: 6"


def test_dual_impedance_configuration_commits_mode_before_kd(monkeypatch):
    class ConfigRobot:
        def __init__(self):
            self.calls: list[tuple] = []

        def clear_set(self): self.calls.append(("clear",))
        def set_state(self, **kwargs): self.calls.append(("state", kwargs["arm"]))
        def set_impedance_type(self, **kwargs): self.calls.append(("type", kwargs["arm"]))
        def set_vel_acc(self, **kwargs): self.calls.append(("vel", kwargs["arm"]))
        def set_joint_kd_params(self, **kwargs): self.calls.append(("kd", kwargs["arm"])); return True
        def send_cmd(self): self.calls.append(("send",)); return True

    robot = ConfigRobot()
    monkeypatch.setattr("tianji_arm.experiments.dual_arm_batch_stream.time.sleep", lambda _: None)

    _configure_dual_joint_impedance(robot)

    assert robot.calls == [
        ("clear",), ("state", "A"), ("type", "A"), ("vel", "A"),
        ("state", "B"), ("type", "B"), ("vel", "B"), ("send",),
        ("clear",), ("kd", "A"), ("kd", "B"), ("send",),
    ]


def test_disable_both_arms_uses_the_concise_sdk_after_the_batch_connection_is_released():
    class DisableRobot:
        def __init__(self): self.calls = []
        def connect(self, ip): self.calls.append(("connect", ip)); return True
        def disable(self, arm): self.calls.append(("disable", arm)); return True
        def release_robot(self): self.calls.append(("release",))

    robot = DisableRobot()
    disable_both_arms("192.168.1.190", concise_factory=lambda: robot)

    assert robot.calls == [
        ("connect", "192.168.1.190"), ("disable", "A"), ("disable", "B"), ("release",)
    ]


def test_feedback_validator_returns_max_error_for_active_error_free_arms():
    feedback = {
        "states": [{"cur_state": 3, "err_code": 0}, {"cur_state": 3, "err_code": 0}],
        "outputs": [
            {"frame_serial": 10, "fb_joint_pos": [0.0] * 7},
            {"frame_serial": 11, "fb_joint_pos": [1.0] * 7},
        ],
    }

    assert validate_dual_feedback(feedback, np.zeros(7), np.zeros(7)) == 1.0


def test_feedback_validator_rejects_inactive_or_stale_arms():
    feedback = {
        "states": [{"cur_state": 0, "err_code": 0}, {"cur_state": 3, "err_code": 0}],
        "outputs": [
            {"frame_serial": 10, "fb_joint_pos": [0.0] * 7},
            {"frame_serial": 11, "fb_joint_pos": [0.0] * 7},
        ],
    }

    with pytest.raises(RuntimeError, match="state"):
        validate_dual_feedback(feedback, np.zeros(7), np.zeros(7))


def test_hold_frames_repeat_the_current_target_for_both_arms():
    left, right = build_hold_frames(np.arange(7), np.arange(10, 17), frames=3)

    assert np.array_equal(left, np.tile(np.arange(7), (3, 1)))
    assert np.array_equal(right, np.tile(np.arange(10, 17), (3, 1)))
