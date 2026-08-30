import csv
import warnings

import numpy as np
import pytest

from twin_sim.force_monitor import ForceMonitor
from twin_sim.logging import SimulationSample, write_csv
from twin_sim.robot import RightArmRobot


def test_threshold_warns_once_per_crossing_without_raising():
    monitor = ForceMonitor(alpha=1.0, warning_threshold_n=5.0)

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        assert monitor.update(6.0).over_threshold is True
        assert monitor.update(7.0).over_threshold is True
        assert monitor.update(4.0).over_threshold is False
        assert monitor.update(6.0).over_threshold is True

    assert len(seen) == 2
    assert all(item.category is RuntimeWarning for item in seen)


def test_monitor_reads_force_sensor_vector_norm():
    robot = RightArmRobot()
    monitor = ForceMonitor(
        robot.sim, alpha=1.0, warning_threshold_n=100.0
    )
    sensor_id = robot.sim.require_sensor("right_tool_force")
    address = robot.sim.model.sensor_adr[sensor_id]
    robot.sim.data.sensordata[address : address + 3] = (3.0, 4.0, 0.0)

    sample = monitor.sample()

    assert sample.raw_force_n == pytest.approx(5.0)


def test_csv_contains_force_phase_and_flattened_state(tmp_path):
    path = tmp_path / "run.csv"
    sample = SimulationSample(
        time_s=0.1,
        phase="TEST",
        target_joints_rad=np.zeros(7),
        actual_joints_rad=np.ones(7),
        target_pose=np.eye(4),
        actual_pose=np.eye(4),
        raw_force_n=2.0,
        filtered_force_n=1.5,
        force_over_threshold=False,
        cut_index=3,
    )

    write_csv(path, [sample])

    row = next(csv.DictReader(path.open()))
    assert row["phase"] == "TEST"
    assert row["cut_index"] == "3"
    assert row["filtered_force_n"] == "1.5"
    assert row["force_over_threshold"] == "False"
    assert row["target_q_0"] == "0.0"
    assert row["actual_q_0"] == "1.0"


def test_csv_rejects_missing_parent_before_opening(tmp_path):
    with pytest.raises(ValueError, match="parent directory"):
        write_csv(tmp_path / "missing" / "run.csv", [])


def test_csv_rejects_a_directory_destination_immediately(tmp_path):
    with pytest.raises(IsADirectoryError):
        write_csv(tmp_path, [])
