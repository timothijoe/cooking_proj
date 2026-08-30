import numpy as np
import pytest

from tianji_robotics.simulation.replay import replay_trajectory
from tianji_robotics.simulation.wuji_hand import MujocoWujiHand
from tianji_robotics.wuji_hand.models import HandTrajectory
from tianji_robotics.wuji_hand.names import HAND_JOINT_NAMES
from twin_sim.hand import DEFAULT_OPEN_RAD


def _trajectory():
    positions = np.vstack((DEFAULT_OPEN_RAD, DEFAULT_OPEN_RAD + 0.01))
    return HandTrajectory(
        timestamps_ns=np.array([0, 10_000_000]),
        positions_rad=positions,
        joint_names=HAND_JOINT_NAMES,
        metadata={"source": "test"},
    )


def test_headless_replay_advances_mujoco_and_accepts_every_target():
    backend = MujocoWujiHand(viewer=False)
    start_time = float(backend.data.time)
    try:
        summary = replay_trajectory(_trajectory(), backend, realtime=False)

        assert float(backend.data.time) > start_time
        np.testing.assert_allclose(
            backend.read_target_position_rad(), DEFAULT_OPEN_RAD + 0.01
        )
        assert summary.frame_count == 2
        assert summary.duration_s == 0.01
    finally:
        backend.close()


class RecordingBackend:
    timestep_s = 0.002
    realtime_paced = False
    range_tolerance_rad = 0.001
    joint_ranges_rad = {name: (-1.0, 2.0) for name in HAND_JOINT_NAMES}

    def __init__(self):
        self.commands = []
        self.steps = []

    def read_position_rad(self):
        return np.zeros(20)

    def command_position_rad(self, target):
        self.commands.append(np.asarray(target).copy())

    def step(self, duration_s):
        self.steps.append(duration_s)

    def close(self):
        pass


def test_replay_subdivides_intervals_and_commands_each_frame():
    backend = RecordingBackend()

    summary = replay_trajectory(_trajectory(), backend, realtime=False)

    assert len(backend.commands) == 2
    assert backend.steps == [0.002] * 5
    assert summary.frame_count == 2


def test_replay_preflights_complete_trajectory_before_first_command():
    backend = RecordingBackend()
    positions = np.vstack((DEFAULT_OPEN_RAD, DEFAULT_OPEN_RAD.copy()))
    positions[1, 4] = 3.0
    trajectory = HandTrajectory(
        timestamps_ns=np.array([0, 10_000_000]),
        positions_rad=positions,
        joint_names=HAND_JOINT_NAMES,
        metadata={},
    )

    with pytest.raises(ValueError, match="frame 1.*left_finger2_joint1"):
        replay_trajectory(trajectory, backend, realtime=False)

    assert backend.commands == []


def test_replay_clips_only_backend_declared_range_tolerance():
    backend = RecordingBackend()
    positions = np.vstack((DEFAULT_OPEN_RAD, DEFAULT_OPEN_RAD.copy()))
    positions[:, 0] = 2.0005
    trajectory = HandTrajectory(
        timestamps_ns=np.array([0, 10_000_000]),
        positions_rad=positions,
        joint_names=HAND_JOINT_NAMES,
        metadata={},
    )

    replay_trajectory(trajectory, backend, realtime=False)

    assert [command[0] for command in backend.commands] == [2.0, 2.0]


def test_replay_rounds_against_cumulative_recording_time():
    backend = RecordingBackend()
    trajectory = HandTrajectory(
        timestamps_ns=np.array([0, 3_000_000, 6_000_000]),
        positions_rad=np.vstack(
            (DEFAULT_OPEN_RAD, DEFAULT_OPEN_RAD, DEFAULT_OPEN_RAD)
        ),
        joint_names=HAND_JOINT_NAMES,
        metadata={},
    )

    replay_trajectory(trajectory, backend, realtime=False)

    assert backend.steps == [0.002] * 3
