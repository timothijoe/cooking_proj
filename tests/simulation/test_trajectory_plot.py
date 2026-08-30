import numpy as np
import pytest

from twin_sim.logging import SimulationSample
import twin_sim.trajectory_plot as trajectory_plot


def _sample(x, y, z, *, phase, cut_index, actual_offset=0.001):
    target = np.eye(4)
    actual = np.eye(4)
    target[:3, 3] = (x, y, z)
    actual[:3, 3] = (x + actual_offset, y, z)
    return SimulationSample(
        time_s=float(cut_index),
        phase=phase,
        target_joints_rad=np.zeros(7),
        actual_joints_rad=np.zeros(7),
        target_pose=target,
        actual_pose=actual,
        raw_force_n=0.0,
        filtered_force_n=0.0,
        force_over_threshold=False,
        cut_index=cut_index,
    )


def test_svg_contains_top_side_paths_legend_and_cut_labels(tmp_path):
    path = tmp_path / "trajectory.svg"
    samples = (
        _sample(0.62, 0.0, 0.31, phase="READY", cut_index=0),
        _sample(0.62, 0.0, 0.23, phase="HOLD", cut_index=1),
        _sample(0.59, 0.0, 0.31, phase="SHIFT", cut_index=1),
        _sample(0.59, 0.0, 0.23, phase="HOLD", cut_index=2),
    )

    trajectory_plot.write_trajectory_svg(path, samples)

    svg = path.read_text(encoding="utf-8")
    assert svg.startswith("<svg")
    assert "Top View (X-Y)" in svg
    assert "Side View (Path-Z)" in svg
    assert "Target" in svg
    assert "Actual" in svg
    assert "Cut 1" in svg
    assert "Cut 2" in svg
    assert svg.count("<polyline") >= 4


def test_svg_rejects_missing_parent_before_writing(tmp_path):
    with pytest.raises(ValueError, match="parent directory"):
        trajectory_plot.write_trajectory_svg(
            tmp_path / "missing" / "trajectory.svg",
            (_sample(0.0, 0.0, 0.0, phase="READY", cut_index=0),),
        )
