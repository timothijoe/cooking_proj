from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def test_required_package_and_entrypoint_directories_exist():
    expected = [
        "packages/robot_core",
        "packages/cooking_simulation",
        "packages/wuji_hand",
        "packages/tianji_arm",
        "packages/cooking_workflows",
        "scripts/setup",
        "scripts/simulation",
        "scripts/hardware",
        "scripts/teleop",
    ]
    assert all((ROOT / path).is_dir() for path in expected)


def test_local_content_is_ignored_and_ros2_is_not_scaffolded():
    result = subprocess.run(
        ["git", "check-ignore", "local/assets/model.stl", "local/vendor/sdk.so", "local/data/demo.npz"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert not (ROOT / "ros2_ws").exists()


def test_environment_contract_is_tracked_as_text():
    assert (ROOT / ".python-version").read_text().strip() == "3.12"
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert "simulation" in pyproject
    assert "hardware" in pyproject
