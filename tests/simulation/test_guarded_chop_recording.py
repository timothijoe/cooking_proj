from types import SimpleNamespace

import numpy as np
import pytest

from twin_sim.guarded_chop_recording import (
    GuardedChopFrame,
    GuardedChopRecording,
    capture_frame,
    load_recording,
    replay_recording,
    save_recording,
)


def _robot(nq=4, nv=3, nu=2):
    data = SimpleNamespace(
        time=0.25,
        qpos=np.arange(nq, dtype=float),
        qvel=np.arange(nv, dtype=float) + 10.0,
        ctrl=np.arange(nu, dtype=float) + 20.0,
    )
    model = SimpleNamespace(nq=nq, nv=nv, nu=nu)
    return SimpleNamespace(sim=SimpleNamespace(data=data, model=model))


def _sample(phase="cut_down", cut_index=1):
    return SimpleNamespace(
        phase=SimpleNamespace(value=phase),
        cut_index=cut_index,
        knife_position=np.array((0.7, 0.1, 0.3)),
        guard_position=np.array((0.6, 0.1, 0.35)),
        knife_guard_distance_m=0.04,
        cut_allowed=True,
    )


def _recording(qpos_offset=0.0):
    robot = _robot()
    robot.sim.data.qpos += qpos_offset
    return GuardedChopRecording((capture_frame(robot, _sample()),))


def test_capture_frame_copies_mutable_mujoco_and_sample_arrays():
    robot = _robot()
    sample = _sample()

    frame = capture_frame(robot, sample)
    robot.sim.data.qpos[:] = -1.0
    robot.sim.data.qvel[:] = -1.0
    robot.sim.data.ctrl[:] = -1.0
    sample.knife_position[:] = -1.0
    sample.guard_position[:] = -1.0

    np.testing.assert_array_equal(frame.qpos, np.arange(4, dtype=float))
    np.testing.assert_array_equal(frame.qvel, np.arange(3) + 10.0)
    np.testing.assert_array_equal(frame.ctrl, np.arange(2) + 20.0)
    np.testing.assert_array_equal(frame.knife_position, (0.7, 0.1, 0.3))
    np.testing.assert_array_equal(frame.guard_position, (0.6, 0.1, 0.35))
    assert not frame.qpos.flags.writeable


def test_save_replaces_same_path_and_round_trips(tmp_path):
    path = tmp_path / "nested" / "latest.npz"

    save_recording(_recording(0.0), path)
    save_recording(_recording(5.0), path)
    loaded = load_recording(path, _robot().sim.model)

    assert [item.name for item in path.parent.iterdir()] == ["latest.npz"]
    assert len(loaded.frames) == 1
    np.testing.assert_array_equal(loaded.frames[0].qpos, np.arange(4) + 5.0)
    assert loaded.frames[0].phase == "cut_down"


def test_load_rejects_recording_for_different_model_dimensions(tmp_path):
    path = tmp_path / "recording.npz"
    save_recording(_recording(), path)

    with pytest.raises(ValueError, match="model dimensions"):
        load_recording(path, _robot(nq=5).sim.model)


@pytest.mark.parametrize("frames", ((),))
def test_recording_rejects_empty_frame_sequence(frames):
    with pytest.raises(ValueError, match="at least one frame"):
        GuardedChopRecording(frames)


def test_replay_restores_states_without_stepping_physics(monkeypatch):
    robot = _robot()
    robot.step = lambda *args, **kwargs: pytest.fail("replay called mj_step")
    robot._viewer = SimpleNamespace(sync=lambda: sync_calls.append(True))
    sync_calls = []
    forward_states = []
    sleeps = []
    trace_events = []

    class Trace:
        def begin_replay(self, rate):
            trace_events.append(("begin", rate))

        def append(self, **values):
            trace_events.append(("append", values))

    first = capture_frame(robot, _sample("cut_down", 1))
    robot.sim.data.time = 0.27
    robot.sim.data.qpos += 5.0
    robot.sim.data.qvel += 5.0
    robot.sim.data.ctrl += 5.0
    second = capture_frame(robot, _sample("knife_clear", 2))

    def forward(model, data):
        forward_states.append((data.time, data.qpos.copy()))

    monkeypatch.setattr(
        "twin_sim.guarded_chop_recording.mujoco.mj_forward", forward
    )

    replay_recording(
        robot,
        GuardedChopRecording((first, second)),
        rate=2.0,
        trace=Trace(),
        sleep=sleeps.append,
    )

    assert [state[0] for state in forward_states] == [0.25, 0.27]
    np.testing.assert_array_equal(forward_states[0][1], first.qpos)
    np.testing.assert_array_equal(forward_states[1][1], second.qpos)
    assert sleeps == [pytest.approx(0.01)]
    assert trace_events[0] == ("begin", 2.0)
    assert [event[0] for event in trace_events[1:]] == ["append", "append"]
    assert sync_calls


@pytest.mark.parametrize("rate", (0.0, -1.0, np.nan, np.inf))
def test_replay_rejects_invalid_rate(rate):
    with pytest.raises(ValueError, match="rate must be positive and finite"):
        replay_recording(_robot(), _recording(), rate=rate, sleep=lambda _: None)
