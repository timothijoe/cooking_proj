from collections import Counter

import numpy as np
import pytest

import twin_sim.tasks as tasks


def test_default_line_chop_runs_five_cuts_and_four_shifts(tmp_path):
    result = tasks.run_line_chop(
        tasks.LineChopConfig(),
        log_path=tmp_path / "line.csv",
        plot_path=tmp_path / "line.svg",
    )

    assert result.completed
    counts = Counter(sample.phase for sample in result.samples)
    assert counts["DESCEND"] == 5 * 100
    assert counts["HOLD"] == 5 * 20
    assert counts["RETRACT"] == 5 * 100
    assert counts["SHIFT"] == 4 * 100
    assert result.samples[-2].phase == "RETRACT"
    assert result.samples[-1].phase == "COMPLETE"


def test_default_cut_points_are_three_centimetres_apart(tmp_path):
    result = tasks.run_line_chop(
        tasks.LineChopConfig(),
        log_path=tmp_path / "spacing.csv",
        plot_path=tmp_path / "spacing.svg",
    )

    deltas = np.diff(result.cut_points_xy, axis=0)
    np.testing.assert_allclose(
        np.linalg.norm(deltas, axis=1),
        0.03,
        rtol=0.0,
        atol=0.001,
    )
    np.testing.assert_allclose(deltas[:, 0], 0.0, atol=0.001)
    assert np.all(deltas[:, 1] < 0.0)


@pytest.mark.parametrize("cuts", [0, -1, 1.5, True])
def test_line_chop_rejects_invalid_cut_count(cuts, tmp_path):
    with pytest.raises(ValueError, match="cuts"):
        tasks.run_line_chop(
            tasks.LineChopConfig(cuts=cuts),
            log_path=tmp_path / "invalid.csv",
            plot_path=tmp_path / "invalid.svg",
        )


def test_line_chop_rejects_path_outside_board_before_motion(tmp_path):
    with pytest.raises(ValueError, match="board"):
        tasks.run_line_chop(
            tasks.LineChopConfig(cuts=20, spacing_m=0.03),
            log_path=tmp_path / "outside.csv",
            plot_path=tmp_path / "outside.svg",
        )
