from dataclasses import dataclass

import numpy as np
import pytest

from tianji_robotics.wuji_hand.models import SkeletonFrame
from tianji_robotics.workflows.wuji_glove_replay import retarget_recording


@dataclass
class FakeSource:
    items: list[SkeletonFrame]

    def frames(self):
        yield from self.items


class RecordingRetargeter:
    def __init__(self):
        self.inputs = []

    def step(self, keypoints_m):
        self.inputs.append(keypoints_m.copy())
        return np.linspace(0.0, 0.19, 20)


def _frames(side="right"):
    return [
        SkeletonFrame(
            timestamp_ns=1_000 + index,
            frame_id="r_wrist",
            side=side,
            keypoints_m=np.arange(63, dtype=float).reshape(21, 3) + index,
        )
        for index in range(2)
    ]


def test_pipeline_mirrors_every_frame_before_retargeting():
    source = FakeSource(_frames())
    retargeter = RecordingRetargeter()

    trajectory = retarget_recording(source, retargeter)

    assert trajectory.positions_rad.shape == (2, 20)
    np.testing.assert_array_equal(
        retargeter.inputs[0][:, 1], -source.items[0].keypoints_m[:, 1]
    )
    np.testing.assert_array_equal(trajectory.timestamps_ns, [1_000, 1_001])
    assert trajectory.metadata["input_transform"] == (
        "mirror_y_right_wrist_to_left_wrist"
    )
    assert trajectory.metadata["target_hand_side"] == "left"


def test_pipeline_rejects_non_right_input():
    with pytest.raises(ValueError, match="right-hand"):
        retarget_recording(FakeSource(_frames(side="left")), RecordingRetargeter())


def test_pipeline_rejects_empty_recording():
    with pytest.raises(ValueError, match="no skeleton frames"):
        retarget_recording(FakeSource([]), RecordingRetargeter())
