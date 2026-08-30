from pathlib import Path
from types import SimpleNamespace

import pytest

import twin_sim.cli as cli
from twin_sim.cli import build_parser, main


def test_guarded_chop_cli_defaults_to_plane_scene():
    args = build_parser().parse_args(["guarded-chop", "--headless"])

    assert args.scene == "plane"
    assert args.final_hold == 3.0


def test_recorded_hand_guarded_chop_cli_has_project_relative_default():
    args = build_parser().parse_args(
        ["recorded-hand-guarded-chop", "--headless"]
    )

    assert args.hand_mcap == Path(
        "recordings/wuji/august_02/"
        "session_20260802_174440_936_right_to_left_wuji_hand.mcap"
    )
    assert args.final_hold == 3.0


def test_recorded_hand_guarded_chop_cli_dispatches_independent_task(
    monkeypatch, capsys
):
    captured = {}

    def fake_run(config, *, hand_mcap, viewer):
        captured.update(config=config, hand_mcap=hand_mcap, viewer=viewer)
        return SimpleNamespace(
            success=True,
            completed_cuts=5,
            completed_hand_cycles=5,
            surface_offset_m=.04,
            minimum_distance_m=.044,
            minimum_lateral_spacing_m=.029,
            maximum_lateral_spacing_m=.031,
            selected_clearance_tier_m=.020,
            selected_lateral_spacing_m=.030,
            maximum_depth_mismatch_m=.006,
            minimum_long_pad_clearances_m=(.000, .003, .004, .002),
            maximum_hand_penetration_m=.0004,
            minimum_thumb_clearance_m=.031,
            reason="",
        )

    monkeypatch.setattr(cli, "run_recorded_hand_guarded_chop", fake_run)
    status = main(
        [
            "recorded-hand-guarded-chop",
            "--hand-mcap",
            "recordings/test.mcap",
            "--headless",
            "--final-hold",
            "0",
        ]
    )

    assert status == 0
    assert captured["viewer"] is False
    assert captured["hand_mcap"] == Path("recordings/test.mcap")
    assert captured["config"].final_hold_s == 0.0
    output = capsys.readouterr().out
    assert "hand_cycles=5" in output
    assert "surface_offset_m=0.040" in output
    assert "lateral_spacing_m=0.029..0.031" in output
    assert "clearance_tier_m=0.020" in output
    assert "depth_mismatch_m=0.006" in output
    assert "pad_clearance_m=0.000..0.004" in output


def test_guarded_chop_cli_accepts_object_scene():
    args = build_parser().parse_args(
        ["guarded-chop", "--scene", "object", "--headless"]
    )

    assert args.scene == "object"


def test_guarded_chop_cli_accepts_replay_and_optional_record_path():
    default_recording = build_parser().parse_args(
        ["guarded-chop", "--record"]
    )
    named_recording = build_parser().parse_args(
        [
            "guarded-chop",
            "--replay-rate",
            "2.0",
            "--record",
            "recordings/demo.npz",
        ]
    )

    assert default_recording.record == Path(
        "recordings/guarded_chop_latest.npz"
    )
    assert named_recording.replay_rate == 2.0
    assert named_recording.record.name == "demo.npz"


def test_guarded_chop_cli_rejects_headless_replay():
    with pytest.raises(SystemExit):
        main(["guarded-chop", "--headless", "--replay-rate", "2.0"])


def test_guarded_chop_replay_cli_defaults_to_latest_at_two_x():
    args = build_parser().parse_args(["guarded-chop-replay"])

    assert args.recording == Path("recordings/guarded_chop_latest.npz")
    assert args.rate == 2.0


def test_guarded_chop_replay_cli_dispatches_existing_recording(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        cli,
        "play_guarded_chop_recording",
        lambda path, *, rate: captured.update(path=path, rate=rate),
    )

    status = main(
        [
            "guarded-chop-replay",
            "--recording",
            "recordings/demo.npz",
            "--rate",
            "1.5",
        ]
    )

    assert status == 0
    assert captured == {
        "path": Path("recordings/demo.npz"),
        "rate": 1.5,
    }


@pytest.mark.parametrize("rate", ("0", "-1", "nan", "inf"))
def test_guarded_chop_replay_cli_rejects_invalid_rate(rate):
    with pytest.raises(SystemExit):
        main(["guarded-chop-replay", "--rate", rate])


def test_guarded_chop_cli_prints_coordination_metrics(
    monkeypatch, capsys
):
    captured = {}

    def fake_run(config, *, viewer, replay_rate=None, record_path=None):
        captured.update(
            config=config,
            viewer=viewer,
            replay_rate=replay_rate,
            record_path=record_path,
        )
        return SimpleNamespace(
            success=True,
            completed_cuts=5,
            completed_shifts=4,
            total_shift_m=0.08,
            minimum_distance_m=0.0234,
            reason="",
        )

    monkeypatch.setattr(cli, "run_guarded_chop", fake_run)

    status = cli.main(
        [
            "guarded-chop",
            "--scene",
            "object",
            "--headless",
            "--final-hold",
            "0",
        ]
    )

    assert status == 0
    assert captured["viewer"] is False
    assert captured["config"].scene_mode == "object"
    assert captured["config"].final_hold_s == 0.0
    assert captured["replay_rate"] is None
    assert captured["record_path"] is None
    output = capsys.readouterr().out
    assert "cuts=5" in output
    assert "shifts=4" in output
    assert "total_shift_m=0.080" in output
    assert "min_distance_m=0.023" in output


def test_headless_chop_cli(tmp_path):
    path = tmp_path / "cli.csv"

    assert main(["chop", "--headless", "--log", str(path)]) == 0
    assert path.is_file()


def test_headless_hand_demo_cli(monkeypatch):
    captured = {}

    def fake_run(config, *, viewer):
        captured.update(config=config, viewer=viewer)

    monkeypatch.setattr(cli, "run_hand_demo", fake_run)
    assert main(["hand-demo", "--headless"]) == 0
    assert captured["viewer"] is False


def test_slow_hand_demo_cli_uses_long_motion(monkeypatch):
    captured = {}

    def fake_run(config, *, viewer):
        captured.update(config=config, viewer=viewer)

    monkeypatch.setattr(cli, "run_hand_demo", fake_run)
    assert main(["hand-demo", "--slow"]) == 0
    assert captured["config"].close_duration_s == 4.0
    assert captured["config"].open_duration_s == 4.0
    assert captured["config"].viewer_start_hold_s == 5.0
    assert captured["config"].viewer_end_hold_s == 10.0
    assert captured["viewer"] is True


def test_headless_joint_and_cartesian_cli():
    assert main(["joint", "--joint", "1", "--delta-rad", "0.01", "--headless"]) == 0
    assert main(["cartesian", "--dz-m", "0.005", "--headless"]) == 0


def test_slow_chop_cli_uses_observable_durations(monkeypatch, tmp_path):
    captured = {}

    def fake_run(config, *, log_path, viewer):
        captured.update(config=config, log_path=log_path, viewer=viewer)

    monkeypatch.setattr(cli, "run_chop", fake_run)
    path = tmp_path / "slow.csv"

    assert main(["chop", "--slow", "--log", str(path)]) == 0
    config = captured["config"]
    assert (
        config.approach_duration_s,
        config.descent_duration_s,
        config.hold_duration_s,
        config.retract_duration_s,
    ) == (3.0, 3.0, 1.0, 3.0)
    assert (config.viewer_start_hold_s, config.viewer_end_hold_s) == (5.0, 8.0)
    assert config.full_motion is False
    assert captured["viewer"] is True


def test_full_motion_cli_preserves_complete_sequence(monkeypatch, tmp_path):
    captured = {}

    def fake_run(config, *, log_path, viewer):
        captured["config"] = config

    monkeypatch.setattr(cli, "run_chop", fake_run)

    assert main(
        [
            "chop",
            "--full-motion",
            "--headless",
            "--log",
            str(tmp_path / "full.csv"),
        ]
    ) == 0
    assert captured["config"].full_motion is True


def test_line_chop_cli_maps_cuts_spacing_and_outputs(monkeypatch, tmp_path):
    captured = {}

    def fake_run(config, *, log_path, plot_path, viewer):
        captured.update(
            config=config,
            log_path=log_path,
            plot_path=plot_path,
            viewer=viewer,
        )

    monkeypatch.setattr(cli, "run_line_chop", fake_run)
    log_path = tmp_path / "line.csv"
    plot_path = tmp_path / "line.svg"

    assert main(
        [
            "line-chop",
            "--cuts",
            "5",
            "--spacing-m",
            "0.03",
            "--headless",
            "--log",
            str(log_path),
            "--plot",
            str(plot_path),
        ]
    ) == 0
    assert captured["config"].cuts == 5
    assert captured["config"].spacing_m == pytest.approx(0.03)
    assert captured["log_path"] == log_path
    assert captured["plot_path"] == plot_path
    assert captured["viewer"] is False
