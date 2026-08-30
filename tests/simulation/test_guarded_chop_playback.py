from pathlib import Path
from types import SimpleNamespace

import numpy as np

import twin_sim.guarded_chop_playback as playback


def test_playback_restores_recording_and_replays_without_running_task(
    monkeypatch,
):
    calls = {}
    first = SimpleNamespace(
        time_s=1.25,
        qpos=np.asarray((1.0, 2.0, 3.0)),
        qvel=np.asarray((4.0, 5.0)),
        ctrl=np.asarray((6.0,)),
    )
    recording = SimpleNamespace(frames=(first,))
    data = SimpleNamespace(
        time=0.0,
        qpos=np.zeros(3),
        qvel=np.zeros(2),
        ctrl=np.zeros(1),
    )
    robot = SimpleNamespace(
        sim=SimpleNamespace(model=object(), data=data),
        _viewer=None,
    )

    def open_viewer():
        calls["opened_viewer"] = True
        robot._viewer = object()

    def close():
        calls["closed"] = True

    robot.open_viewer = open_viewer
    robot.close = close
    plan = SimpleNamespace(
        cut_points_xy=np.asarray(
            ((0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (0.0, 3.0), (0.0, 4.0))
        ),
        cuts=tuple(
            SimpleNamespace(
                descent=(SimpleNamespace(joints_rad=np.zeros(7)),)
            )
            for _ in range(5)
        ),
    )

    class FakeTrace:
        def __init__(self, viewer):
            calls["trace_viewer"] = viewer

        def set_plan(self, points):
            calls["plan"] = np.asarray(points)

    monkeypatch.setattr(playback, "RightArmRobot", lambda viewer: robot)
    monkeypatch.setattr(playback, "_preflight_guarded_chop", lambda r, c: plan)
    monkeypatch.setattr(
        playback,
        "_configure_guarded_scene",
        lambda r, scene: calls.update(scene=scene),
    )
    monkeypatch.setattr(
        playback,
        "load_recording",
        lambda path, model: calls.update(load_path=path) or recording,
    )
    monkeypatch.setattr(playback.mujoco, "mj_forward", lambda model, d: None)
    monkeypatch.setattr(
        playback, "_prepare_guarded_chop_viewer", lambda r: None
    )
    monkeypatch.setattr(playback, "GuardedChopTrace", FakeTrace)
    monkeypatch.setattr(
        playback,
        "_blade_center_for_joints",
        lambda r, joints: np.asarray((0.0, 0.0, 0.3)),
    )

    def replay(r, loaded, *, rate, trace):
        calls["rate"] = rate
        calls["first_state_restored"] = (
            r.sim.data.time == first.time_s
            and np.array_equal(r.sim.data.qpos, first.qpos)
            and np.array_equal(r.sim.data.qvel, first.qvel)
            and np.array_equal(r.sim.data.ctrl, first.ctrl)
        )

    monkeypatch.setattr(playback, "replay_recording", replay)

    playback.play_guarded_chop_recording(Path("demo.npz"), rate=2.0)

    assert calls["scene"] == "plane"
    assert calls["load_path"] == Path("demo.npz")
    assert calls["rate"] == 2.0
    assert calls["opened_viewer"] is True
    assert calls["first_state_restored"] is True
    assert calls["closed"] is True
    assert calls["plan"].shape == (5, 3)
    assert not hasattr(playback, "run_guarded_chop")
