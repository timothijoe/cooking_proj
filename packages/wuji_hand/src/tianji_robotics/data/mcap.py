"""MCAP codecs for offline Wuji hand recordings."""

from collections.abc import Iterator
import json
from pathlib import Path
from typing import Literal

from mcap.reader import make_reader
from mcap.writer import Writer
import numpy as np

from tianji_robotics.wuji_hand.models import HandTrajectory, SkeletonFrame
from tianji_robotics.wuji_hand.names import HAND_JOINT_NAMES


SKELETON_TOPIC = "/right_glove/hand_skeleton"
JOINT_STATES_TOPIC = "/joint_states"
JOINT_STATE_SCHEMA = {
    "title": "sensor_msgs/msg/JointState",
    "type": "object",
    "properties": {
        "header": {"type": "object"},
        "name": {"type": "array", "items": {"type": "string"}},
        "position": {"type": "array", "items": {"type": "number"}},
        "velocity": {"type": "array", "items": {"type": "number"}},
        "effort": {"type": "array", "items": {"type": "number"}},
    },
    "required": ["header", "name", "position"],
}


def detect_hand_mcap_kind(
    path: Path,
) -> Literal["joint_states", "right_glove_skeleton"]:
    path = Path(path)
    with path.open("rb") as stream:
        summary = make_reader(stream).get_summary()
    if summary is None:
        raise ValueError(f"MCAP has no summary: {path}")
    topics = {channel.topic for channel in summary.channels.values()}
    supported = {
        kind
        for topic, kind in (
            (JOINT_STATES_TOPIC, "joint_states"),
            (SKELETON_TOPIC, "right_glove_skeleton"),
        )
        if topic in topics
    }
    if len(supported) != 1:
        raise ValueError(f"MCAP must contain exactly one supported hand topic: {path}")
    return supported.pop()


class JointStateMcapSource:
    """Load canonical left-hand joint samples without running retargeting."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def trajectory(self) -> HandTrajectory:
        timestamps = []
        positions = []
        try:
            with self.path.open("rb") as stream:
                reader = make_reader(stream)
                for _schema, _channel, message in reader.iter_messages(
                    topics=(JOINT_STATES_TOPIC,)
                ):
                    payload = json.loads(message.data)
                    names = tuple(payload["name"])
                    if len(names) != len(set(names)) or set(names) != set(HAND_JOINT_NAMES):
                        raise ValueError("joint names are not canonical")
                    values = np.asarray(payload["position"], dtype=float)
                    if values.shape != (20,) or not np.isfinite(values).all():
                        raise ValueError("joint positions must contain 20 finite values")
                    order = [names.index(name) for name in HAND_JOINT_NAMES]
                    positions.append(values[order])
                    timestamps.append(int(payload["header"]["timestamp_ns"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"malformed joint state in {self.path}") from exc
        if not positions:
            raise ValueError(f"joint-state MCAP contains no samples: {self.path}")
        return HandTrajectory(
            timestamps_ns=np.asarray(timestamps, dtype=np.int64),
            positions_rad=np.asarray(positions, dtype=float),
            joint_names=HAND_JOINT_NAMES,
            metadata={"source_mcap": str(self.path), "source_kind": "joint_states"},
        )


class StudioMcapSkeletonSource:
    """Read right-glove skeleton frames recorded by Wuji Studio."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def frames(self) -> Iterator[SkeletonFrame]:
        try:
            with self.path.open("rb") as stream:
                reader = make_reader(stream)
                for _schema, _channel, message in reader.iter_messages(
                    topics=(SKELETON_TOPIC,)
                ):
                    payload = json.loads(message.data)
                    header = payload["header"]
                    keypoints = [
                        joint["pose"]["position"] for joint in payload["joints"]
                    ]
                    yield SkeletonFrame(
                        timestamp_ns=header["timestamp_us"] * 1_000,
                        frame_id=header["frame_id"],
                        side="right",
                        keypoints_m=keypoints,
                    )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"malformed skeleton frame in {self.path}") from exc


def write_joint_state_mcap(trajectory: HandTrajectory, destination: Path) -> Path:
    """Write a trajectory as JSON messages on the standard joint-state topic."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        writer = Writer(stream)
        writer.start(profile="ros2", library="tianji-robotics")
        schema_id = writer.register_schema(
            "sensor_msgs/msg/JointState",
            "jsonschema",
            json.dumps(JOINT_STATE_SCHEMA, separators=(",", ":")).encode(),
        )
        channel_id = writer.register_channel(
            JOINT_STATES_TOPIC,
            "json",
            schema_id,
            {"handedness": "left"},
        )
        for sequence, (timestamp_ns, positions_rad) in enumerate(
            zip(trajectory.timestamps_ns, trajectory.positions_rad, strict=True)
        ):
            timestamp = int(timestamp_ns)
            payload = json.dumps(
                {
                    "header": {
                        "seq": sequence,
                        "timestamp_ns": timestamp,
                        "frame_id": "left_palm_link",
                    },
                    "name": list(trajectory.joint_names),
                    "position": positions_rad.tolist(),
                    "velocity": [],
                    "effort": [],
                },
                separators=(",", ":"),
            ).encode()
            writer.add_message(
                channel_id,
                log_time=timestamp,
                publish_time=timestamp,
                sequence=sequence,
                data=payload,
            )
        writer.finish()
    return destination
