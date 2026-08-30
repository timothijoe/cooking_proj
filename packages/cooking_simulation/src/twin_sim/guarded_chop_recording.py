from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
import time
from typing import Callable, Sequence

import mujoco
import numpy as np


_SCHEMA_VERSION = 1


def _immutable_vector(value, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float).copy()
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector")
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class GuardedChopFrame:
    time_s: float
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    phase: str
    cut_index: int
    knife_position: np.ndarray
    guard_position: np.ndarray
    minimum_distance_m: float
    cut_allowed: bool

    def __post_init__(self) -> None:
        time_s = float(self.time_s)
        minimum_distance_m = float(self.minimum_distance_m)
        if not np.isfinite(time_s) or time_s < 0.0:
            raise ValueError("frame time must be finite and non-negative")
        if not np.isfinite(minimum_distance_m):
            raise ValueError("minimum distance must be finite")
        object.__setattr__(self, "time_s", time_s)
        object.__setattr__(self, "minimum_distance_m", minimum_distance_m)
        object.__setattr__(self, "phase", str(self.phase))
        object.__setattr__(self, "cut_index", int(self.cut_index))
        object.__setattr__(self, "cut_allowed", bool(self.cut_allowed))
        for name in (
            "qpos",
            "qvel",
            "ctrl",
            "knife_position",
            "guard_position",
        ):
            object.__setattr__(
                self,
                name,
                _immutable_vector(getattr(self, name), name=name),
            )
        if self.knife_position.shape != (3,) or self.guard_position.shape != (3,):
            raise ValueError("knife and guard positions must contain three values")


@dataclass(frozen=True)
class GuardedChopRecording:
    frames: tuple[GuardedChopFrame, ...]

    def __init__(self, frames: Sequence[GuardedChopFrame]) -> None:
        values = tuple(frames)
        if not values:
            raise ValueError("recording requires at least one frame")
        dimensions = {
            (frame.qpos.size, frame.qvel.size, frame.ctrl.size)
            for frame in values
        }
        if len(dimensions) != 1:
            raise ValueError("recording frame dimensions must match")
        times = np.asarray([frame.time_s for frame in values])
        if np.any(np.diff(times) < 0.0):
            raise ValueError("recording frame times must be nondecreasing")
        object.__setattr__(self, "frames", values)

    @property
    def model_dimensions(self) -> tuple[int, int, int]:
        first = self.frames[0]
        return first.qpos.size, first.qvel.size, first.ctrl.size


def capture_frame(robot, sample) -> GuardedChopFrame:
    phase = getattr(sample.phase, "value", sample.phase)
    return GuardedChopFrame(
        time_s=robot.sim.data.time,
        qpos=robot.sim.data.qpos,
        qvel=robot.sim.data.qvel,
        ctrl=robot.sim.data.ctrl,
        phase=phase,
        cut_index=sample.cut_index,
        knife_position=sample.knife_position,
        guard_position=sample.guard_position,
        minimum_distance_m=sample.knife_guard_distance_m,
        cut_allowed=sample.cut_allowed,
    )


def save_recording(recording: GuardedChopRecording, path: Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(
                temporary,
                schema_version=np.asarray(_SCHEMA_VERSION, dtype=np.int64),
                time_s=np.asarray([frame.time_s for frame in recording.frames]),
                qpos=np.stack([frame.qpos for frame in recording.frames]),
                qvel=np.stack([frame.qvel for frame in recording.frames]),
                ctrl=np.stack([frame.ctrl for frame in recording.frames]),
                phase=np.asarray([frame.phase for frame in recording.frames]),
                cut_index=np.asarray(
                    [frame.cut_index for frame in recording.frames], dtype=np.int64
                ),
                knife_position=np.stack(
                    [frame.knife_position for frame in recording.frames]
                ),
                guard_position=np.stack(
                    [frame.guard_position for frame in recording.frames]
                ),
                minimum_distance_m=np.asarray(
                    [frame.minimum_distance_m for frame in recording.frames]
                ),
                cut_allowed=np.asarray(
                    [frame.cut_allowed for frame in recording.frames], dtype=bool
                ),
            )
        os.replace(temporary_path, target)
        temporary_path = None
        return target
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_recording(path: Path, model) -> GuardedChopRecording:
    with np.load(Path(path), allow_pickle=False) as data:
        version = int(data["schema_version"])
        if version != _SCHEMA_VERSION:
            raise ValueError(f"unsupported recording schema version: {version}")
        qpos = np.asarray(data["qpos"])
        qvel = np.asarray(data["qvel"])
        ctrl = np.asarray(data["ctrl"])
        expected = (int(model.nq), int(model.nv), int(model.nu))
        actual = (
            qpos.shape[1] if qpos.ndim == 2 else -1,
            qvel.shape[1] if qvel.ndim == 2 else -1,
            ctrl.shape[1] if ctrl.ndim == 2 else -1,
        )
        if actual != expected:
            raise ValueError(
                f"recording model dimensions {actual} do not match {expected}"
            )
        count = qpos.shape[0]
        arrays = {
            name: np.asarray(data[name])
            for name in (
                "time_s",
                "phase",
                "cut_index",
                "knife_position",
                "guard_position",
                "minimum_distance_m",
                "cut_allowed",
            )
        }
        if any(value.shape[0] != count for value in arrays.values()):
            raise ValueError("recording arrays have inconsistent frame counts")
        frames = tuple(
            GuardedChopFrame(
                time_s=arrays["time_s"][index],
                qpos=qpos[index],
                qvel=qvel[index],
                ctrl=ctrl[index],
                phase=arrays["phase"][index],
                cut_index=arrays["cut_index"][index],
                knife_position=arrays["knife_position"][index],
                guard_position=arrays["guard_position"][index],
                minimum_distance_m=arrays["minimum_distance_m"][index],
                cut_allowed=arrays["cut_allowed"][index],
            )
            for index in range(count)
        )
    return GuardedChopRecording(frames)


def validate_replay_rate(rate: float) -> float:
    value = float(rate)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("rate must be positive and finite")
    return value


def replay_recording(
    robot,
    recording: GuardedChopRecording,
    *,
    rate: float,
    trace=None,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    playback_rate = validate_replay_rate(rate)
    if recording.model_dimensions != (
        int(robot.sim.model.nq),
        int(robot.sim.model.nv),
        int(robot.sim.model.nu),
    ):
        raise ValueError("recording model dimensions do not match robot model")
    if trace is not None:
        trace.begin_replay(playback_rate)

    times = np.asarray([frame.time_s for frame in recording.frames])
    positive_deltas = np.diff(times)
    positive_deltas = positive_deltas[positive_deltas > 0.0]
    wall_step = (
        float(np.median(positive_deltas)) / playback_rate
        if positive_deltas.size
        else 1.0 / 30.0
    )
    sync_stride = max(1, int(round((1.0 / 30.0) / wall_step)))
    last_index = len(recording.frames) - 1

    for index, frame in enumerate(recording.frames):
        robot.sim.data.time = frame.time_s
        robot.sim.data.qpos[:] = frame.qpos
        robot.sim.data.qvel[:] = frame.qvel
        robot.sim.data.ctrl[:] = frame.ctrl
        mujoco.mj_forward(robot.sim.model, robot.sim.data)
        if trace is not None:
            trace.append(
                actual_knife=frame.knife_position,
                actual_guard=frame.guard_position,
                phase=frame.phase,
                cut_index=frame.cut_index,
                minimum_distance_m=frame.minimum_distance_m,
                cut_allowed=frame.cut_allowed,
            )
        viewer = robot._viewer
        if viewer is not None and (
            index % sync_stride == 0 or index == last_index
        ):
            viewer.sync()
        if index < last_index:
            delay = max(
                0.0,
                (recording.frames[index + 1].time_s - frame.time_s)
                / playback_rate,
            )
            sleep(delay)
