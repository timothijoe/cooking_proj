import json

import numpy as np
from mcap.reader import make_reader

from tianji_robotics.data.mcap import (
    JointStateMcapSource,
    detect_hand_mcap_kind,
    write_joint_state_mcap,
)
from tianji_robotics.data.npz import load_trajectory_npz, save_trajectory_npz
from tianji_robotics.wuji_hand.models import HandTrajectory
from tianji_robotics.wuji_hand.names import HAND_JOINT_NAMES


def _trajectory():
    return HandTrajectory(
        timestamps_ns=np.array([1_000, 2_000], dtype=np.int64),
        positions_rad=np.arange(40, dtype=np.float32).reshape(2, 20) / 10,
        joint_names=HAND_JOINT_NAMES,
        metadata={
            "source": "test",
            "nested": {"flags": (True, None), "raw": b"bytes"},
            "labels": frozenset({"a", "b"}),
        },
    )


def test_npz_round_trip_preserves_trajectory_values_and_metadata(tmp_path):
    trajectory = _trajectory()

    path = save_trajectory_npz(trajectory, tmp_path / "trajectory.npz")
    loaded = load_trajectory_npz(path)

    assert path == tmp_path / "trajectory.npz"
    np.testing.assert_array_equal(loaded.timestamps_ns, trajectory.timestamps_ns)
    np.testing.assert_allclose(loaded.positions_rad, trajectory.positions_rad)
    assert loaded.joint_names == HAND_JOINT_NAMES
    assert dict(loaded.metadata) == dict(trajectory.metadata)


def test_joint_state_mcap_contains_each_trajectory_sample(tmp_path):
    trajectory = _trajectory()
    path = write_joint_state_mcap(trajectory, tmp_path / "nested" / "trajectory.mcap")

    with path.open("rb") as stream:
        messages = list(make_reader(stream).iter_messages(topics="/joint_states"))

    assert path == tmp_path / "nested" / "trajectory.mcap"
    assert len(messages) == 2
    payload = json.loads(messages[0][2].data)
    assert payload["name"] == list(HAND_JOINT_NAMES)
    assert payload["position"] == trajectory.positions_rad[0].tolist()
    assert payload["header"] == {
        "seq": 0,
        "timestamp_ns": 1_000,
        "frame_id": "left_palm_link",
    }
    assert payload["velocity"] == []
    assert payload["effort"] == []


def test_joint_state_source_round_trips_canonical_trajectory(tmp_path):
    expected = _trajectory()
    path = write_joint_state_mcap(expected, tmp_path / "left.mcap")

    assert detect_hand_mcap_kind(path) == "joint_states"
    actual = JointStateMcapSource(path).trajectory()

    np.testing.assert_allclose(actual.positions_rad, expected.positions_rad)
    np.testing.assert_array_equal(actual.timestamps_ns, expected.timestamps_ns)
    assert actual.joint_names == HAND_JOINT_NAMES


def test_npz_writer_creates_destination_parent(tmp_path):
    path = save_trajectory_npz(_trajectory(), tmp_path / "nested" / "trajectory.npz")

    assert path.is_file()
