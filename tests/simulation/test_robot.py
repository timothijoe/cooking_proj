import threading

import mujoco.viewer
import numpy as np
import pytest

from twin_sim.robot import RIGHT_HOME_RAD, NumericalSafetyError, RightArmRobot


def test_position_target_is_tracked():
    robot = RightArmRobot()
    robot.reset(RIGHT_HOME_RAD)
    target = RIGHT_HOME_RAD + np.array([0.02, 0, 0, 0, 0, 0, 0])
    robot.command(target)
    for _ in range(250):
        robot.step(0.002)
    assert np.linalg.norm(robot.joint_positions - target) < 0.03
    robot.close()


def test_invalid_command_is_rejected():
    robot = RightArmRobot()
    with pytest.raises(ValueError, match="7 finite"):
        robot.command([float("nan")] * 7)
    robot.close()


def test_command_rejects_a_joint_outside_model_range():
    robot = RightArmRobot()
    target = RIGHT_HOME_RAD.copy()
    target[5] = 99.0

    with pytest.raises(ValueError, match="joint 6.*range"):
        robot.command(target)
    robot.close()


def test_validate_targets_rejects_complete_path_before_stepping():
    robot = RightArmRobot()
    invalid = RIGHT_HOME_RAD.copy()
    invalid[5] = 99.0

    with pytest.raises(ValueError, match="sample 1.*joint 6.*range"):
        robot.validate_targets([RIGHT_HOME_RAD, invalid])

    assert robot.sim.data.time == 0.0
    robot.close()


def test_nonfinite_state_stops():
    robot = RightArmRobot()
    robot.sim.data.qpos[robot.sim.right.qpos_ids[0]] = np.nan
    with pytest.raises(NumericalSafetyError):
        robot.step(0.002)
    robot.close()


def test_step_requires_a_positive_integer_number_of_physics_ticks():
    robot = RightArmRobot()
    with pytest.raises(ValueError, match="positive integer multiple"):
        robot.step(0.003)
    robot.close()


def test_step_rejects_a_nonfinite_control_period():
    robot = RightArmRobot()
    with pytest.raises(ValueError, match="positive integer multiple"):
        robot.step(float("nan"))
    robot.close()


def test_step_rejects_a_near_multiple_control_period():
    robot = RightArmRobot()
    with pytest.raises(ValueError, match="positive integer multiple"):
        robot.step(0.002000000001)
    robot.close()


def test_reset_holds_the_left_arm_and_exposes_right_arm_state():
    robot = RightArmRobot()
    left_hold = robot.sim.data.qpos[robot.sim.left.qpos_ids].copy()
    robot.reset(RIGHT_HOME_RAD)

    np.testing.assert_allclose(robot.joint_positions, RIGHT_HOME_RAD)
    np.testing.assert_allclose(robot.joint_velocities, np.zeros(7))
    assert robot.tcp_pose().shape == (7,)
    assert np.isfinite(robot.tcp_pose()).all()

    robot.command(RIGHT_HOME_RAD)
    robot.step(0.002)

    np.testing.assert_allclose(robot.sim.data.ctrl[robot.sim.left.actuator_ids], left_hold)
    robot.close()


def test_viewer_step_is_paced_in_real_time(monkeypatch):
    robot = RightArmRobot()

    class FakeViewer:
        def sync(self):
            pass

        def close(self):
            pass

    sleeps = []
    robot._viewer = FakeViewer()
    monkeypatch.setattr("twin_sim.robot.time.sleep", sleeps.append)

    robot.step(0.002)

    assert sleeps == [0.002]
    robot.close()


def test_viewer_step_can_defer_sync_without_changing_pacing(monkeypatch):
    robot = RightArmRobot()

    class FakeViewer:
        def __init__(self):
            self.sync_calls = 0

        def sync(self):
            self.sync_calls += 1

        def close(self):
            pass

    viewer = FakeViewer()
    sleeps = []
    robot._viewer = viewer
    monkeypatch.setattr("twin_sim.robot.time.sleep", sleeps.append)

    robot.step(0.002, sync_viewer=False)

    assert viewer.sync_calls == 0
    assert sleeps == [0.002]
    robot.close()


def test_close_waits_for_passive_viewer_thread():
    robot = RightArmRobot()

    class FakeViewer:
        def close(self):
            pass

    class FakeThread:
        def __init__(self):
            self.joined = False

        def join(self, timeout=None):
            self.joined = True

    thread = FakeThread()
    robot._viewer = FakeViewer()
    robot._viewer_thread = thread

    robot.close()

    assert thread.joined


def test_viewer_launch_tracks_only_the_thread_created_for_its_target(
    monkeypatch,
):
    class FakeViewer:
        def close(self):
            pass

    class FakeThread:
        def __init__(self, ident, name, target):
            self.ident = ident
            self.name = name
            self._target = target

        def join(self, timeout=None):
            pass

    existing = FakeThread(1, "existing", object())
    launched = FakeThread(
        2, "viewer-worker", mujoco.viewer._launch_internal
    )
    decoy = FakeThread(3, "Thread (_launch_internal)", object())
    snapshots = iter(
        ([existing], [existing, launched, decoy])
    )
    monkeypatch.setattr(threading, "enumerate", lambda: next(snapshots))
    monkeypatch.setattr(
        mujoco.viewer,
        "launch_passive",
        lambda model, data: FakeViewer(),
    )

    robot = RightArmRobot(viewer=True)

    assert robot._viewer_thread is launched
    robot.close()


def test_open_viewer_is_idempotent(monkeypatch):
    class FakeViewer:
        def close(self):
            pass

    launches = []

    def launch(model, data):
        launches.append((model, data))
        return FakeViewer()

    monkeypatch.setattr(mujoco.viewer, "launch_passive", launch)
    robot = RightArmRobot(viewer=False)

    robot.open_viewer()
    robot.open_viewer()

    assert launches == [(robot.sim.model, robot.sim.data)]
    robot.close()


def test_close_bounds_the_passive_viewer_thread_wait():
    robot = RightArmRobot()

    class FakeViewer:
        def close(self):
            pass

    class FakeThread:
        def __init__(self):
            self.join_timeouts = []

        def join(self, timeout):
            self.join_timeouts.append(timeout)

    thread = FakeThread()
    robot._viewer = FakeViewer()
    robot._viewer_thread = thread

    robot.close()

    assert thread.join_timeouts == [1.0]
