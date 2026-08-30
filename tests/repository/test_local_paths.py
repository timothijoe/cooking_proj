from pathlib import Path

import pytest


def test_local_roots_default_to_repository_local(monkeypatch):
    monkeypatch.delenv("COOKING_LOCAL_ROOT", raising=False)
    from robot_core.paths import local_root

    assert local_root().name == "local"
    assert local_root().parent == Path(__file__).resolve().parents[2]


def test_local_root_can_be_overridden(monkeypatch, tmp_path):
    monkeypatch.setenv("COOKING_LOCAL_ROOT", str(tmp_path))
    from robot_core.paths import asset_root, data_root, vendor_root

    assert asset_root() == tmp_path / "assets"
    assert data_root() == tmp_path / "data"
    assert vendor_root() == tmp_path / "vendor"


def test_missing_asset_has_actionable_error(monkeypatch, tmp_path):
    monkeypatch.setenv("COOKING_LOCAL_ROOT", str(tmp_path))
    from robot_core.paths import require_local

    with pytest.raises(FileNotFoundError, match="local assets"):
        require_local("assets/simulation/model.xml", kind="local assets")
