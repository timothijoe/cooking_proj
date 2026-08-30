from pathlib import Path

import pytest

from tianji_robotics import cli


def test_headless_replay_routes_without_viewer(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cli, "_run_wuji_replay", lambda args: calls.append(args) or 0)
    assert cli.main(["sim", "wuji-replay", str(tmp_path / "in.mcap"), "--headless"]) == 0
    assert calls[0].headless is True


def test_hardware_preflight_only_loads_trajectory(monkeypatch, tmp_path):
    path = tmp_path / "trajectory.npz"
    loaded = []
    monkeypatch.setattr(cli, "_preflight_trajectory", lambda candidate: loaded.append(candidate) or 0)
    assert cli.main(["hardware", "wuji-sdk", "preflight", str(path)]) == 0
    assert loaded == [path]


def test_hardware_cli_rejects_arm_option(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["hardware", "wuji-sdk", "preflight", str(tmp_path / "x.npz"), "--arm"])


def test_simulation_parser_rejects_hardware_options(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["sim", "wuji-replay", str(tmp_path / "x.mcap"), "--arm"])


def test_angle_bar_routes_with_default_left_hand(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "_run_wuji_angle_bar", lambda args: calls.append(args) or 0)
    assert cli.main(["sim", "wuji-angle-bar"]) == 0
    assert calls[0].hand == "left"


def test_angle_bar_accepts_right_hand(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "_run_wuji_angle_bar", lambda args: calls.append(args) or 0)
    assert cli.main(["sim", "wuji-angle-bar", "--hand", "right"]) == 0
    assert calls[0].hand == "right"


def test_table_retreat_routes_as_an_independent_simulation(monkeypatch, tmp_path):
    calls=[]
    monkeypatch.setattr(cli,"_run_wuji_table_retreat",lambda args: calls.append(args) or 0)
    assert cli.main(["sim","wuji-table-retreat",str(tmp_path/"in.mcap"),"--headless","--retreat-distance", "0.04"])==0
    assert calls[0].headless is True and calls[0].retreat_distance==0.04


def test_table_retreat_defaults_to_three_loops(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        cli, "_run_wuji_table_retreat", lambda args: calls.append(args) or 0
    )

    cli.main(["sim", "wuji-table-retreat", str(tmp_path / "in.mcap"), "--headless"])

    assert calls[0].loops == 3


def test_table_retreat_accepts_custom_positive_loop_count(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        cli, "_run_wuji_table_retreat", lambda args: calls.append(args) or 0
    )

    cli.main(
        ["sim", "wuji-table-retreat", str(tmp_path / "in.mcap"), "--loops", "5"]
    )

    assert calls[0].loops == 5


@pytest.mark.parametrize("value", ["0", "-1", "1.5"])
def test_table_retreat_rejects_invalid_loop_count(value, tmp_path):
    with pytest.raises(SystemExit):
        cli.main(
            ["sim", "wuji-table-retreat", str(tmp_path / "in.mcap"), "--loops", value]
        )
