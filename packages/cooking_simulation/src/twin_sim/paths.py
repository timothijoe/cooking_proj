from pathlib import Path

from robot_core.paths import asset_root, repository_root


def project_root() -> Path:
    return repository_root()


def scene_path() -> Path:
    path = asset_root() / "robot_assets" / "mujoco" / "right_chopping_scene.xml"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing simulation model: {path}; restore local assets before running MuJoCo"
        )
    return path
