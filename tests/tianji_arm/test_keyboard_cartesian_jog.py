from __future__ import annotations

import numpy as np
import pytest

from tianji_arm.experiments.keyboard_cartesian_jog import (
    JogConfig,
    candidate_pose,
    inside_workspace,
    key_to_delta,
    parse_args,
    run_jog_session,
    validate_config,
    workspace_bounds_from_center,
)
import tianji_arm.experiments.keyboard_cartesian_jog as jog


class FakeDcss:
    pass


class FakeRobot:
    def __init__(self) -> None:
        self.planned_commands: list[tuple[str, object]] = []
        self.disabled = False
        self.released = False
        self.frame_serial = 0

    def connect(self, robot_ip: str) -> bool:
        return True

    def check_error_and_clear(self, dcss: FakeDcss) -> None:
        pass

    def subscribe(self, dcss: FakeDcss) -> dict:
        self.frame_serial += 1
        return {
            "outputs": [{"frame_serial": self.frame_serial, "fb_joint_pos": [0.0] * 7, "traj_state": 0}],
            "states": [{"cur_state": 1, "err_code": 0}],
        }

    def setPln_Cart(self, arm: str, pset: object) -> None:
        self.planned_commands.append((arm, pset))

    def set_vel_acc(self, arm: str, velRatio: int, AccRatio: int) -> None:
        pass

    def clear_set(self) -> None:
        pass

    def set_state(self, arm: str, state: int) -> None:
        self.disabled = state == 0

    def send_cmd(self) -> None:
        pass

    def release_robot(self) -> None:
        self.released = True


class FakeKine:
    def fk(self, joints: list[float]) -> np.ndarray:
        return np.asarray(joints, dtype=float)

    def mat4x4_to_xyzabc(self, matrix: np.ndarray) -> list[float]:
        return [0.0] * 6

    def movLA(self, **kwargs):
        return [[0.0] * 7], object()


class FailingKine(FakeKine):
    def movLA(self, **kwargs):
        return [], None


def test_w_requests_one_positive_x_step():
    assert key_to_delta("w", 2.0) == (2.0, 0.0, 0.0)


def test_f_requests_one_negative_z_step():
    assert key_to_delta("F", 2.0) == (0.0, 0.0, -2.0)


def test_candidate_pose_translates_xyz_and_preserves_orientation():
    current = np.array([100.0, 200.0, 300.0, 10.0, 20.0, 30.0])

    target = candidate_pose(current, (2.0, -2.0, 0.0))

    assert np.array_equal(target, np.array([102.0, 198.0, 300.0, 10.0, 20.0, 30.0]))


def test_execute_requires_both_workspace_bounds():
    with pytest.raises(ValueError, match="workspace-min.*workspace-max"):
        validate_config(JogConfig(execute=True))


def test_execute_accepts_a_workspace_radius_instead_of_explicit_bounds():
    validate_config(JogConfig(execute=True, workspace_radius_mm=50.0))


def test_parse_execute_accepts_workspace_radius():
    config = parse_args(["--execute", "--workspace-around-current-mm", "50"])

    assert config.workspace_radius_mm == 50.0


def test_workspace_radius_builds_a_symmetric_box_from_current_tcp():
    lower, upper = workspace_bounds_from_center(np.array([347.0, 90.0, 18.0]), 50.0)

    assert np.array_equal(lower, np.array([297.0, 40.0, -32.0]))
    assert np.array_equal(upper, np.array([397.0, 140.0, 68.0]))


def test_ten_mm_step_is_allowed_but_larger_step_is_rejected():
    validate_config(JogConfig(step_mm=10.0))

    with pytest.raises(ValueError, match="step-mm must be in .*10"):
        validate_config(JogConfig(step_mm=10.1))


def test_workspace_radius_allows_up_to_150_mm():
    validate_config(JogConfig(execute=True, workspace_radius_mm=150.0))

    with pytest.raises(ValueError, match="workspace-around-current-mm must be in .*150"):
        validate_config(JogConfig(execute=True, workspace_radius_mm=150.1))


def test_workspace_includes_edges_but_rejects_outside_point():
    lower, upper = (0.0, 0.0, 0.0), (10.0, 10.0, 10.0)

    assert inside_workspace(np.array([0.0, 10.0, 5.0]), lower, upper)
    assert not inside_workspace(np.array([10.1, 10.0, 5.0]), lower, upper)


def test_dry_run_plans_one_step_without_sending_robot_command():
    robot, dcss, kine = FakeRobot(), FakeDcss(), FakeKine()

    poses = run_jog_session(
        JogConfig(),
        read_key=iter(["w", "q"]).__next__,
        sdk_factory=lambda: (robot, dcss, kine),
    )

    assert np.array_equal(poses[-1][:3], np.array([5.0, 0.0, 0.0]))
    assert robot.planned_commands == []
    assert robot.disabled
    assert robot.released


def test_execute_outside_workspace_does_not_plan_or_send():
    robot, dcss, kine = FakeRobot(), FakeDcss(), FakeKine()
    config = JogConfig(
        execute=True,
        workspace_min=(0.0, 0.0, 0.0),
        workspace_max=(1.0, 1.0, 1.0),
    )

    with pytest.raises(ValueError, match="outside workspace"):
        run_jog_session(config, read_key=iter(["w"]).__next__, sdk_factory=lambda: (robot, dcss, kine))

    assert robot.planned_commands == []
    assert robot.disabled


def test_execute_planning_failure_does_not_send_command():
    robot, dcss, kine = FakeRobot(), FakeDcss(), FailingKine()
    config = JogConfig(
        execute=True,
        workspace_min=(-5.0, -5.0, -5.0),
        workspace_max=(5.0, 5.0, 5.0),
    )

    with pytest.raises(RuntimeError, match="MOVLA planning failed"):
        run_jog_session(config, read_key=iter(["w"]).__next__, sdk_factory=lambda: (robot, dcss, kine))

    assert robot.planned_commands == []
    assert robot.disabled


def test_execute_sends_one_planned_command_inside_workspace():
    robot, dcss, kine = FakeRobot(), FakeDcss(), FakeKine()
    config = JogConfig(
        execute=True,
        workspace_min=(-5.0, -5.0, -5.0),
        workspace_max=(5.0, 5.0, 5.0),
    )

    run_jog_session(config, read_key=iter(["w", "q"]).__next__, sdk_factory=lambda: (robot, dcss, kine))

    assert len(robot.planned_commands) == 1
    assert robot.planned_commands[0][0] == "A"


def test_parse_execute_requires_workspace_values():
    with pytest.raises(ValueError, match="workspace-min.*workspace-max"):
        parse_args(["--execute"])


def test_space_stops_before_later_motion_key():
    robot, dcss, kine = FakeRobot(), FakeDcss(), FakeKine()

    run_jog_session(
        JogConfig(),
        read_key=iter([" ", "w"]).__next__,
        sdk_factory=lambda: (robot, dcss, kine),
    )

    assert robot.disabled
    assert robot.planned_commands == []


def test_feedback_frame_check_waits_for_controller_refresh(monkeypatch):
    class DelayedFrameRobot:
        def __init__(self) -> None:
            self.frame_serial = 100

        def subscribe(self, dcss: FakeDcss) -> dict:
            return {"outputs": [{"frame_serial": self.frame_serial}]}

    robot = DelayedFrameRobot()
    sleeps: list[float] = []

    def advance_controller_frame(delay_s: float) -> None:
        sleeps.append(delay_s)
        robot.frame_serial += 1

    monkeypatch.setattr(jog.time, "sleep", advance_controller_frame)

    jog._verify_frame_updates(robot, FakeDcss(), arm_index=0)

    assert sleeps == [0.01] * 5


def test_sdk_byte_zero_trajectory_state_is_idle():
    class IdleRobot:
        def subscribe(self, dcss: FakeDcss) -> dict:
            return {"outputs": [{"traj_state": b"\x00"}]}

    assert jog._trajectory_is_idle(IdleRobot(), FakeDcss(), arm_index=0)


def test_ctrl_c_byte_stops_before_later_motion_key():
    robot, dcss, kine = FakeRobot(), FakeDcss(), FakeKine()

    poses = run_jog_session(
        JogConfig(),
        read_key=iter(["\x03", "w"]).__next__,
        sdk_factory=lambda: (robot, dcss, kine),
    )

    assert len(poses) == 1
    assert robot.planned_commands == []
    assert robot.disabled


def test_planning_mode_setup_waits_for_state_transition(monkeypatch):
    robot = FakeRobot()
    sleeps: list[float] = []
    monkeypatch.setattr(jog.time, "sleep", sleeps.append)

    jog._configure_planning_mode(robot, JogConfig(execute=True), FakeDcss(), arm_index=0)

    assert sleeps[:2] == [0.1, 0.2]


def test_terminal_key_reader_uses_cbreak_to_preserve_ctrl_c(monkeypatch):
    class Stream:
        def fileno(self) -> int:
            return 42

        def read(self, count: int) -> str:
            return "w"

    events: list[tuple[str, int]] = []
    monkeypatch.setattr(jog.termios, "tcgetattr", lambda fd: ["saved"])
    monkeypatch.setattr(jog.termios, "tcsetattr", lambda fd, when, attrs: events.append(("restore", fd)))
    monkeypatch.setattr(jog.tty, "setcbreak", lambda fd: events.append(("cbreak", fd)))
    monkeypatch.setattr(jog.tty, "setraw", lambda fd: (_ for _ in ()).throw(AssertionError("raw mode disables Ctrl+C")))

    with jog.raw_terminal_keys(Stream()) as read_key:
        assert read_key() == "w"

    assert events == [("cbreak", 42), ("restore", 42)]
