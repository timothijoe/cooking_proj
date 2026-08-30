import subprocess
import sys

import numpy as np
import pytest

from tianji_robotics.data.left_arm_entry import (
    generate_left_arm_entry,
    save_left_arm_entry_npz,
)


def test_generates_200hz_20_second_quintic_entry_with_exact_endpoints():
    start = np.deg2rad(np.array([-90, -90, 90, -90, 0, 0, 0], dtype=float))
    target = np.deg2rad(
        np.array([71.431, -66.107, -49.733, -128.492, 94.122, 45.648, -49.911])
    )

    result = generate_left_arm_entry(start, target)

    assert result.time_s.shape == (4001,)
    assert result.left_arm_target_rad.shape == (4001, 7)
    assert result.time_s[0] == 0.0
    assert result.time_s[-1] == 20.0
    assert np.allclose(np.diff(result.time_s), 0.005)
    assert np.array_equal(result.left_arm_target_rad[0], start)
    assert np.array_equal(result.left_arm_target_rad[-1], target)


@pytest.mark.parametrize("bad", [np.zeros(6), np.full(7, np.nan)])
def test_rejects_nonfinite_or_nonseven_joint_vector(bad):
    with pytest.raises(ValueError, match="seven finite"):
        generate_left_arm_entry(bad, np.zeros(7))


def test_has_near_zero_boundary_velocity_and_acceleration():
    result = generate_left_arm_entry(
        np.zeros(7), np.ones(7), duration_s=2.0, sample_rate_hz=200.0
    )

    velocity = np.gradient(result.left_arm_target_rad, result.time_s, axis=0)
    acceleration = np.gradient(velocity, result.time_s, axis=0)

    assert np.all(np.abs(velocity[[0, -1]]) < 2e-4)
    assert np.all(np.abs(acceleration[[0, -1]]) < 0.05)


def test_save_writes_left_arm_only_entry_metadata(tmp_path):
    source = tmp_path / "source.npz"
    np.savez(source, left_arm_target_rad=np.vstack((np.zeros(7), np.ones(7))))
    output = tmp_path / "left_offline_entry_only.npz"
    trajectory = generate_left_arm_entry(
        np.zeros(7), np.ones(7), duration_s=1.0, sample_rate_hz=10.0
    )

    saved = save_left_arm_entry_npz(
        trajectory,
        source_npz=source,
        destination=output,
        start_deg=np.zeros(7),
        duration_s=1.0,
        sample_rate_hz=10.0,
    )

    with np.load(saved, allow_pickle=False) as data:
        assert set(data.files) == {
            "format_version",
            "time_s",
            "left_arm_target_rad",
            "source_npz_path",
            "entry_start_deg",
            "entry_destination_rad",
            "duration_s",
            "sample_rate_hz",
            "interpolation",
            "peak_velocity_rad_s",
            "peak_acceleration_rad_s2",
        }
        assert "right_arm_target_rad" not in data.files
        assert "right_hand_target_rad" not in data.files
        assert data["interpolation"].item() == "quintic_smoothstep"


@pytest.mark.parametrize("bad_start_deg", [np.zeros(6), np.full(7, np.nan)])
def test_save_rejects_invalid_entry_start_audit_metadata(tmp_path, bad_start_deg):
    trajectory = generate_left_arm_entry(
        np.zeros(7), np.ones(7), duration_s=1.0, sample_rate_hz=10.0
    )

    with pytest.raises(ValueError, match="start_deg must contain seven finite"):
        save_left_arm_entry_npz(
            trajectory,
            source_npz=tmp_path / "source.npz",
            destination=tmp_path / "left_offline_entry_only.npz",
            start_deg=bad_start_deg,
            duration_s=1.0,
            sample_rate_hz=10.0,
        )


def test_save_rejects_entry_start_audit_metadata_inconsistent_with_trajectory(tmp_path):
    trajectory = generate_left_arm_entry(
        np.zeros(7), np.ones(7), duration_s=1.0, sample_rate_hz=10.0
    )

    with pytest.raises(ValueError, match="start_deg must match"):
        save_left_arm_entry_npz(
            trajectory,
            source_npz=tmp_path / "source.npz",
            destination=tmp_path / "left_offline_entry_only.npz",
            start_deg=np.ones(7),
            duration_s=1.0,
            sample_rate_hz=10.0,
        )


def test_cli_writes_offline_entry_file(tmp_path):
    source = tmp_path / "source.npz"
    output = tmp_path / "result_offline_entry_only.npz"
    source_first_row = np.arange(7, dtype=float)
    np.savez(
        source,
        left_arm_target_rad=np.vstack((source_first_row, np.full(7, -3.0))),
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/data/generate_left_arm_entry_trajectory.py",
            "--source-npz",
            str(source),
            "--start-deg",
            "0,0,0,0,0,0,0",
            "--output",
            str(output),
            "--duration-s",
            "1",
            "--sample-rate-hz",
            "10",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert output.exists()
    with np.load(output, allow_pickle=False) as data:
        assert np.array_equal(data["left_arm_target_rad"][-1], source_first_row)


def test_cli_rejects_source_without_left_arm_targets(tmp_path):
    source = tmp_path / "missing_left_arm.npz"
    output = tmp_path / "result_offline_entry_only.npz"
    np.savez(source, right_arm_target_rad=np.zeros((1, 7)))

    result = subprocess.run(
        [
            sys.executable,
            "scripts/data/generate_left_arm_entry_trajectory.py",
            "--source-npz",
            str(source),
            "--start-deg",
            "0,0,0,0,0,0,0",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "left_arm_target_rad" in result.stderr
