from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tianji_arm.experiments.arm_command_diagnostic import (
    ArmDiagnosticConfig,
    build_quintic_entry,
    command_single_joint_trial,
    entry_progress_line,
    load_entry_target_deg,
    validate_config,
)


class FakeRobot:
    def __init__(self, *, command_result: bool = True) -> None:
        self.command_result = command_result
        self.commands: list[tuple[str, list[float]]] = []

    def set_joint_position_cmd(self, arm: str, joint: list[float]) -> bool:
        self.commands.append((arm, joint))
        return self.command_result


def test_read_only_configuration_never_requires_a_trial_target():
    validate_config(ArmDiagnosticConfig())


def test_execute_requires_an_explicit_joint_and_delta():
    with pytest.raises(ValueError, match="trial-joint.*trial-delta-deg"):
        validate_config(ArmDiagnosticConfig(execute=True))


def test_execute_rejects_a_trial_larger_than_one_degree():
    with pytest.raises(ValueError, match="must be in .*1"):
        validate_config(
            ArmDiagnosticConfig(execute=True, trial_joint=3, trial_delta_deg=1.01)
        )


def test_trial_command_changes_only_the_requested_a_arm_joint():
    robot = FakeRobot()

    target = command_single_joint_trial(
        robot,
        current_deg=np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]),
        joint_number=3,
        delta_deg=-1.0,
    )

    assert target == [10.0, 20.0, 29.0, 40.0, 50.0, 60.0, 70.0]
    assert robot.commands == [("A", target)]


def test_trial_command_refusal_is_reported_as_an_error():
    with pytest.raises(RuntimeError, match="rejected"):
        command_single_joint_trial(
            FakeRobot(command_result=False),
            current_deg=np.zeros(7),
            joint_number=1,
            delta_deg=1.0,
        )


def test_trial_command_rejects_a_target_outside_the_hardware_limit():
    with pytest.raises(ValueError, match="hardware limit"):
        command_single_joint_trial(
            FakeRobot(),
            current_deg=np.array([169.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            joint_number=1,
            delta_deg=1.0,
        )


def test_entry_mode_requires_a_source_npz_and_explicit_execution():
    with pytest.raises(ValueError, match="entry-npz"):
        validate_config(ArmDiagnosticConfig(execute_entry=True))


def test_load_entry_target_converts_first_left_arm_frame_to_degrees(tmp_path):
    source = tmp_path / "trajectory.npz"
    np.savez(
        source,
        time_s=np.array([0.0, 0.005]),
        left_arm_target_rad=np.array([[0.1] * 7, [0.2] * 7]),
    )

    assert np.allclose(load_entry_target_deg(source), np.rad2deg(np.full(7, 0.1)))


def test_quintic_entry_has_exact_endpoints():
    entry = build_quintic_entry(np.zeros(7), np.ones(7), duration_s=1.0, control_hz=10.0)

    assert entry.shape == (11, 7)
    assert np.array_equal(entry[0], np.zeros(7))
    assert np.array_equal(entry[-1], np.ones(7))


def test_entry_configuration_accepts_the_original_player_speed_ratios():
    validate_config(
        ArmDiagnosticConfig(
            execute=True,
            execute_entry=True,
            entry_npz=Path(__file__),
            vel_ratio=100,
            acc_ratio=100,
        )
    )


def test_entry_progress_reports_frame_schedule_and_wall_clock_lag():
    assert entry_progress_line(index=400, control_hz=200.0, elapsed_s=2.15) == (
        "entry_frame=400 scheduled_s=2.000 elapsed_s=2.150 lag_s=0.150"
    )
