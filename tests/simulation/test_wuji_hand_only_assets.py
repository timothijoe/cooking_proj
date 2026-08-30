from pathlib import Path

import mujoco
import pytest


def test_asset_path_is_independent_of_working_directory(monkeypatch, tmp_path):
    from robot_core.paths import asset_root
    from tianji_robotics.simulation.paths import official_wuji_left_mjcf

    expected = asset_root() / "robot_assets/mujoco/wuji_hand_standalone/mjcf/left.xml"
    monkeypatch.chdir(tmp_path)
    assert official_wuji_left_mjcf() == expected


def test_missing_asset_fails_with_resolved_path(monkeypatch, tmp_path):
    from tianji_robotics.simulation import paths

    missing_root = tmp_path / "assets"
    monkeypatch.setattr(paths, "asset_root", lambda: missing_root)
    with pytest.raises(FileNotFoundError, match="wuji_hand_standalone/mjcf/left.xml"):
        paths.official_wuji_left_mjcf()


def test_official_asset_is_a_hand_only_20_dof_model():
    from tianji_robotics.simulation.paths import official_wuji_left_mjcf

    model = mujoco.MjModel.from_xml_path(str(official_wuji_left_mjcf()))
    assert (model.nq, model.nv, model.nu) == (20, 20, 20)
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm_link") >= 0

    names = bytes(model.names).decode(errors="ignore").lower()
    for forbidden in ("marvin", "tianji", "left_link", "right_link", "arm_joint"):
        assert forbidden not in names
