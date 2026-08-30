"""Immutable, validated data crossing the Wuji hand domain boundary."""

from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Mapping
from typing import Literal
import warnings

import numpy as np
from numpy.typing import NDArray

from .names import HAND_JOINT_NAMES


def _copied_readonly_array(
    value: object, *, dtype: np.dtype[np.generic] | type[np.generic] | None = None
) -> NDArray[np.generic]:
    """Copy into an ndarray backed by immutable bytes.

    A normal non-writeable ndarray can be made writeable again by callers when it
    owns its memory. A bytes-backed array cannot, which preserves value-object
    immutability while retaining the public ndarray API.
    """
    with np.errstate(over="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        normalized = np.array(value, dtype=dtype, copy=True)
    return np.frombuffer(normalized.tobytes(), dtype=normalized.dtype).reshape(normalized.shape)


def _is_real_numeric_dtype(dtype: np.dtype[np.generic]) -> bool:
    return np.issubdtype(dtype, np.number) and not np.issubdtype(dtype, np.complexfloating)


def _freeze_metadata(value: object) -> object:
    if isinstance(value, np.ndarray):
        raise ValueError("metadata must not contain ndarrays")
    if isinstance(value, np.generic):
        return _freeze_metadata(value.item())
    if value is None:
        return value
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, bytes):
        return bytes(value)
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, nested_value in value.items():
            if not isinstance(key, str):
                raise ValueError("metadata keys must be strings")
            frozen[key] = _freeze_metadata(nested_value)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_metadata(item) for item in value)
    raise ValueError(f"metadata contains unsupported value type: {type(value).__name__}")


@dataclass(frozen=True)
class SkeletonFrame:
    timestamp_ns: int
    frame_id: str
    side: Literal["left", "right"]
    keypoints_m: NDArray[np.floating]

    def __post_init__(self) -> None:
        if (
            isinstance(self.timestamp_ns, bool)
            or not isinstance(self.timestamp_ns, (int, np.integer))
            or self.timestamp_ns < 0
        ):
            raise ValueError("timestamp_ns must be a non-negative integer")
        if not self.frame_id:
            raise ValueError("frame_id must be non-empty")
        if self.side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        keypoints_source = np.asarray(self.keypoints_m)
        if keypoints_source.shape != (21, 3):
            raise ValueError("keypoints_m must have shape (21, 3)")
        if not _is_real_numeric_dtype(keypoints_source.dtype):
            raise ValueError("keypoints_m must contain real numbers")
        if not np.isfinite(keypoints_source).all():
            raise ValueError("keypoints_m must be finite")
        keypoints = _copied_readonly_array(keypoints_source, dtype=np.float64)
        if not np.isfinite(keypoints).all():
            raise ValueError("keypoints_m must be finite after float64 normalization")
        object.__setattr__(self, "keypoints_m", keypoints)


@dataclass(frozen=True)
class HandTrajectory:
    timestamps_ns: NDArray[np.integer]
    positions_rad: NDArray[np.floating]
    joint_names: tuple[str, ...]
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        timestamps_source = np.asarray(self.timestamps_ns)
        positions_source = np.asarray(self.positions_rad)
        joint_names = tuple(self.joint_names)
        if timestamps_source.ndim != 1 or timestamps_source.size == 0:
            raise ValueError("timestamps_ns must be a non-empty one-dimensional array")
        if not np.issubdtype(timestamps_source.dtype, np.integer):
            raise ValueError("timestamps_ns must contain real integers")
        if np.any(timestamps_source < 0):
            raise ValueError("timestamps_ns must be non-negative")
        if (
            np.issubdtype(timestamps_source.dtype, np.unsignedinteger)
            and np.any(timestamps_source > np.iinfo(np.int64).max)
        ):
            raise ValueError("timestamps_ns must fit in int64")
        if positions_source.shape != (timestamps_source.size, 20):
            raise ValueError("positions_rad must have shape (N, 20)")
        if not _is_real_numeric_dtype(positions_source.dtype):
            raise ValueError("positions_rad must contain real numbers")
        if not np.isfinite(positions_source).all():
            raise ValueError("positions_rad must be finite")
        timestamps = _copied_readonly_array(timestamps_source, dtype=np.int64)
        positions = _copied_readonly_array(positions_source, dtype=np.float64)
        if not np.isfinite(positions).all():
            raise ValueError("positions_rad must be finite after float64 normalization")
        if not np.all(timestamps[1:] > timestamps[:-1]):
            raise ValueError("timestamps_ns must be strictly increasing")
        if joint_names != HAND_JOINT_NAMES or len(set(joint_names)) != 20:
            raise ValueError("joint_names must be the 20 unique canonical joint names")
        object.__setattr__(self, "timestamps_ns", timestamps)
        object.__setattr__(self, "positions_rad", positions)
        object.__setattr__(self, "joint_names", joint_names)
        frozen_metadata = _freeze_metadata(self.metadata)
        if not isinstance(frozen_metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        object.__setattr__(self, "metadata", frozen_metadata)
