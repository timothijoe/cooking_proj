"""NPZ serialization for validated hand trajectories."""

from collections.abc import Mapping
import base64
import json
from pathlib import Path

import numpy as np

from tianji_robotics.wuji_hand.models import HandTrajectory


def _metadata_to_json_value(value: object) -> dict[str, object]:
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"type": "scalar", "value": value}
    if isinstance(value, bytes):
        return {"type": "bytes", "value": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Mapping):
        return {
            "type": "mapping",
            "value": [[key, _metadata_to_json_value(item)] for key, item in value.items()],
        }
    if isinstance(value, tuple):
        return {"type": "sequence", "value": [_metadata_to_json_value(item) for item in value]}
    if isinstance(value, frozenset):
        return {"type": "set", "value": [_metadata_to_json_value(item) for item in value]}
    raise ValueError(f"metadata contains unsupported value type: {type(value).__name__}")


def _metadata_from_json_value(value: object) -> object:
    if not isinstance(value, dict) or set(value) != {"type", "value"}:
        raise ValueError("metadata JSON has an invalid value")
    value_type = value["type"]
    raw_value = value["value"]
    if value_type == "scalar":
        return raw_value
    if value_type == "bytes" and isinstance(raw_value, str):
        return base64.b64decode(raw_value.encode("ascii"), validate=True)
    if value_type == "mapping" and isinstance(raw_value, list):
        return {key: _metadata_from_json_value(item) for key, item in raw_value}
    if value_type == "sequence" and isinstance(raw_value, list):
        return tuple(_metadata_from_json_value(item) for item in raw_value)
    if value_type == "set" and isinstance(raw_value, list):
        return frozenset(_metadata_from_json_value(item) for item in raw_value)
    raise ValueError("metadata JSON has an invalid type")


def save_trajectory_npz(trajectory: HandTrajectory, destination: Path) -> Path:
    """Serialize a validated trajectory without relying on pickle."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps(_metadata_to_json_value(trajectory.metadata), separators=(",", ":"))
    np.savez_compressed(
        destination,
        timestamps_ns=trajectory.timestamps_ns,
        positions_rad=trajectory.positions_rad,
        joint_names=np.asarray(trajectory.joint_names),
        metadata_json=np.asarray(metadata),
    )
    return destination


def load_trajectory_npz(path: Path) -> HandTrajectory:
    """Load a trajectory and re-validate it at the domain boundary."""
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata_json = archive["metadata_json"].item()
            metadata = _metadata_from_json_value(json.loads(metadata_json))
            return HandTrajectory(
                timestamps_ns=archive["timestamps_ns"],
                positions_rad=archive["positions_rad"],
                joint_names=tuple(archive["joint_names"].tolist()),
                metadata=metadata,
            )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid trajectory NPZ: {path}") from exc
