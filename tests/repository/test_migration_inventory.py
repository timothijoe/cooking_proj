from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def test_all_tianji_and_cross_device_experiments_were_migrated():
    expected = {
        "a_arm_impedance_two_stage.py",
        "arm_command_diagnostic.py",
        "arm_impedance_playback.py",
        "bak.py",
        "dual_arm_batch_stream.py",
        "dual_arm_hand_batch_stream.py",
        "dual_arm_hand_playback.py",
        "gamepad_cartesian_jog.py",
        "gamepad_cartesian_jog_simstyle.py",
        "hand_only_stream.py",
        "keyboard_cartesian_jog.py",
        "plan_sampled_ik_chop.py",
        "real_ik_cart_impedance_lateral.py",
        "real_pln_cart_position_chop.py",
        "real_position_mode_a_arm.py",
        "real_sampled_joint_impedance_chop.py",
        "real_sampled_position_chop.py",
        "shift_right_arm_trajectory.py",
    }
    roots = [
        ROOT / "packages/tianji_arm/src/tianji_arm/experiments",
        ROOT / "packages/cooking_workflows/src/cooking_workflows",
        ROOT / "packages/wuji_hand/src/wuji_hand/tools",
    ]
    actual = {path.name for root in roots for path in root.glob("*.py")}
    assert expected <= actual


def test_ros2_was_deferred():
    assert not (ROOT / "ros2_ws").exists()
    assert not (ROOT / "packages/cooking_simulation/src/twin_sim/ros2_bridge.py").exists()


def test_no_trackable_file_is_larger_than_five_megabytes():
    oversized = []
    # Test versionable files, not ignored virtualenv wheels or local archives.
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
    ).decode().split("\0")
    for name in names:
        path = ROOT / name
        if not name or not path.is_file():
            continue
        if path.stat().st_size > 5 * 1024 * 1024:
            oversized.append(path.relative_to(ROOT))
    assert oversized == []
