import numpy as np
import pytest
import warnings

from tianji_robotics.wuji_hand.models import HandTrajectory, SkeletonFrame
from tianji_robotics.wuji_hand.names import HAND_JOINT_NAMES


def test_skeleton_frame_copies_and_validates_keypoints():
    keypoints = np.arange(63, dtype=np.float64).reshape(21, 3)
    frame = SkeletonFrame(1, "right_wrist", "right", keypoints)

    keypoints[0, 0] = -1

    assert frame.timestamp_ns == 1
    assert frame.frame_id == "right_wrist"
    assert frame.side == "right"
    assert frame.keypoints_m[0, 0] == 0
    assert not np.shares_memory(frame.keypoints_m, keypoints)


def test_models_normalize_real_coordinate_arrays_to_float64_and_timestamps_to_int64():
    frame = SkeletonFrame(1, "right_wrist", "right", np.zeros((21, 3), dtype=np.int16))
    trajectory = HandTrajectory(
        np.array([10, 20], dtype=np.int32),
        np.zeros((2, 20), dtype=np.int64),
        HAND_JOINT_NAMES,
        {},
    )

    assert frame.keypoints_m.dtype == np.float64
    assert trajectory.positions_rad.dtype == np.float64
    assert trajectory.timestamps_ns.dtype == np.int64


@pytest.mark.skipif(
    np.finfo(np.longdouble).max <= np.finfo(np.float64).max,
    reason="platform longdouble is not wider than float64",
)
def test_models_reject_longdouble_values_that_overflow_float64_without_warning():
    overflowing = np.longdouble(np.finfo(np.float64).max) * 2
    keypoints = np.zeros((21, 3), dtype=np.longdouble)
    positions = np.zeros((1, 20), dtype=np.longdouble)
    keypoints[0, 0] = overflowing
    positions[0, 0] = overflowing

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match="finite"):
            SkeletonFrame(1, "right_wrist", "right", keypoints)
        with pytest.raises(ValueError, match="finite"):
            HandTrajectory(np.array([10]), positions, HAND_JOINT_NAMES, {})


def test_skeleton_frame_keypoints_cannot_be_made_writeable():
    frame = SkeletonFrame(1, "right_wrist", "right", np.zeros((21, 3)))

    with pytest.raises(ValueError):
        frame.keypoints_m.setflags(write=True)

    assert frame.keypoints_m[0, 0] == 0


@pytest.mark.parametrize(
    ("keypoints", "message"),
    [
        (np.zeros((20, 3)), "shape"),
        (np.full((21, 3), np.nan), "finite"),
    ],
)
def test_skeleton_frame_rejects_invalid_keypoints(keypoints, message):
    with pytest.raises(ValueError, match=message):
        SkeletonFrame(1, "right_wrist", "right", keypoints)


def test_trajectory_requires_strictly_increasing_timestamps():
    with pytest.raises(ValueError, match="strictly increasing"):
        HandTrajectory(np.array([10, 10]), np.zeros((2, 20)), HAND_JOINT_NAMES, {})


def test_trajectory_rejects_negative_timestamps():
    with pytest.raises(ValueError, match="non-negative"):
        HandTrajectory(np.array([-1, 10]), np.zeros((2, 20)), HAND_JOINT_NAMES, {})


def test_trajectory_accepts_zero_to_int64_max_timestamps_without_overflow():
    limits = np.iinfo(np.int64)
    trajectory = HandTrajectory(
        np.array([0, limits.max], dtype=np.int64),
        np.zeros((2, 20)),
        HAND_JOINT_NAMES,
        {},
    )

    np.testing.assert_array_equal(trajectory.timestamps_ns, [0, limits.max])


def test_trajectory_copies_arrays_and_metadata():
    timestamps = np.array([10, 20], dtype=np.int64)
    positions = np.zeros((2, 20), dtype=np.float64)
    metadata = {"source": "fixture"}
    trajectory = HandTrajectory(timestamps, positions, HAND_JOINT_NAMES, metadata)

    timestamps[0] = 0
    positions[0, 0] = 1
    metadata["source"] = "changed"

    assert trajectory.timestamps_ns[0] == 10
    assert trajectory.positions_rad[0, 0] == 0
    assert trajectory.metadata["source"] == "fixture"
    assert not np.shares_memory(trajectory.timestamps_ns, timestamps)
    assert not np.shares_memory(trajectory.positions_rad, positions)


def test_trajectory_arrays_cannot_be_made_writeable():
    trajectory = HandTrajectory(
        np.array([10, 20]), np.zeros((2, 20)), HAND_JOINT_NAMES, {}
    )

    with pytest.raises(ValueError):
        trajectory.timestamps_ns.setflags(write=True)
    with pytest.raises(ValueError):
        trajectory.positions_rad.setflags(write=True)

    np.testing.assert_array_equal(trajectory.timestamps_ns, [10, 20])
    np.testing.assert_array_equal(trajectory.positions_rad, np.zeros((2, 20)))


def test_trajectory_recursively_freezes_metadata_without_nested_aliases():
    metadata = {"nested": {"labels": ["initial"], "modes": {"safe"}}}
    trajectory = HandTrajectory(
        np.array([10]), np.zeros((1, 20)), HAND_JOINT_NAMES, metadata
    )

    metadata["nested"]["labels"].append("changed")
    metadata["nested"]["modes"].add("unsafe")
    metadata["nested"]["new"] = "value"

    assert trajectory.metadata == {"nested": {"labels": ("initial",), "modes": frozenset({"safe"})}}
    with pytest.raises(TypeError):
        trajectory.metadata["new"] = "value"  # type: ignore[index]
    with pytest.raises(TypeError):
        trajectory.metadata["nested"]["new"] = "value"  # type: ignore[index]


def test_trajectory_metadata_copies_binary_values_and_normalizes_numpy_scalars():
    binary = bytearray(b"safe")
    view = memoryview(b"view")
    trajectory = HandTrajectory(
        np.array([10]),
        np.zeros((1, 20)),
        HAND_JOINT_NAMES,
        {"binary": binary, "view": view, "scalar": np.int64(2)},
    )

    binary[0] = ord("u")

    assert trajectory.metadata == {"binary": b"safe", "view": b"view", "scalar": 2}
    assert type(trajectory.metadata["scalar"]) is int
    with pytest.raises(TypeError):
        trajectory.metadata["binary"] = b"changed"  # type: ignore[index]


class MutableMetadataValue:
    pass


@pytest.mark.parametrize("value", [np.array([1]), MutableMetadataValue()])
def test_trajectory_rejects_unsafe_metadata_values(value):
    with pytest.raises(ValueError, match="metadata"):
        HandTrajectory(
            np.array([10]), np.zeros((1, 20)), HAND_JOINT_NAMES, {"unsafe": value}
        )


class IntMetadataSubclass(int):
    pass


def test_trajectory_normalizes_primitive_metadata_subclasses_to_builtin_values():
    value = IntMetadataSubclass(2)
    value.mutable_attribute = []
    trajectory = HandTrajectory(
        np.array([10]), np.zeros((1, 20)), HAND_JOINT_NAMES, {"value": value}
    )

    value.mutable_attribute.append("changed")

    assert trajectory.metadata["value"] == 2
    assert type(trajectory.metadata["value"]) is int


@pytest.mark.parametrize("timestamp", [True, 1.0, "1", -1])
def test_skeleton_frame_rejects_non_integer_or_negative_timestamp(timestamp):
    with pytest.raises(ValueError, match="timestamp_ns"):
        SkeletonFrame(timestamp, "right_wrist", "right", np.zeros((21, 3)))


@pytest.mark.parametrize("timestamp", [1, np.int64(1)])
def test_skeleton_frame_accepts_nonnegative_python_and_numpy_integers(timestamp):
    assert SkeletonFrame(timestamp, "right_wrist", "right", np.zeros((21, 3))).timestamp_ns == timestamp


def test_trajectory_requires_canonical_unique_joint_names():
    names = HAND_JOINT_NAMES[:-1] + (HAND_JOINT_NAMES[-2],)

    with pytest.raises(ValueError, match="canonical"):
        HandTrajectory(np.array([10]), np.zeros((1, 20)), names, {})


@pytest.mark.parametrize(
    ("keypoints", "message"),
    [(np.zeros((21, 3), dtype=np.complex128), "real"),],
)
def test_skeleton_frame_rejects_complex_keypoints(keypoints, message):
    with pytest.raises(ValueError, match=message):
        SkeletonFrame(1, "right_wrist", "right", keypoints)


def test_trajectory_rejects_complex_positions_and_timestamps():
    with pytest.raises(ValueError, match="real"):
        HandTrajectory(
            np.array([10]), np.zeros((1, 20), dtype=np.complex128), HAND_JOINT_NAMES, {}
        )
    with pytest.raises(ValueError, match="real integers"):
        HandTrajectory(
            np.array([10 + 0j]), np.zeros((1, 20)), HAND_JOINT_NAMES, {}
        )
