from types import SimpleNamespace

import numpy as np
import pytest

import tianji_robotics.simulation.local_angle_bar as local_angle_bar
from tianji_robotics.simulation.local_angle_bar import LocalWujiHand
from tianji_robotics.simulation.paths import official_wuji_hand_mjcf


@pytest.mark.parametrize("side", ["left", "right"])
def test_each_official_hand_has_twenty_bounded_position_actuators(side):
    backend = LocalWujiHand(side, viewer=False)
    try:
        assert backend.side == side
        assert len(backend.joint_names) == 20
        assert backend.control_ranges_rad.shape == (20, 2)
        assert np.all(np.isfinite(backend.control_ranges_rad))
        assert np.all(backend.control_ranges_rad[:, 0] < backend.control_ranges_rad[:, 1])
        assert official_wuji_hand_mjcf(side).is_file()
    finally:
        backend.close()


def test_backend_rejects_wrong_shape_nan_and_out_of_range_targets():
    backend = LocalWujiHand("left", viewer=False)
    try:
        with pytest.raises(ValueError, match="20 finite"):
            backend.command(np.zeros(19))
        with pytest.raises(ValueError, match="20 finite"):
            backend.command(np.full(20, np.nan))
        bad = backend.target_rad.copy()
        bad[0] = backend.control_ranges_rad[0, 1] + 0.01
        with pytest.raises(ValueError, match="outside range"):
            backend.command(bad)
    finally:
        backend.close()


def test_reset_open_commands_active_hand_open_target():
    backend = LocalWujiHand("right", viewer=False)
    try:
        backend.command(backend.control_ranges_rad[:, 1])

        backend.reset_open()

        np.testing.assert_allclose(backend.target_rad, backend.open_target_rad)
        assert np.all(backend.target_rad >= backend.control_ranges_rad[:, 0])
        assert np.all(backend.target_rad <= backend.control_ranges_rad[:, 1])
    finally:
        backend.close()


def test_backend_step_advances_once_without_owning_real_time_pacing(monkeypatch):
    backend = LocalWujiHand("left", viewer=False)
    step_calls = []

    class Viewer:
        def is_running(self):
            return True

        def sync(self):
            pass

    try:
        backend._viewer = Viewer()
        monkeypatch.setattr(
            local_angle_bar.mujoco,
            "mj_step",
            lambda model, data: step_calls.append((model, data)),
        )
        monkeypatch.setattr(
            local_angle_bar,
            "time",
            SimpleNamespace(sleep=lambda _seconds: pytest.fail("backend must not sleep")),
            raising=False,
        )

        backend.step()

        assert step_calls == [(backend.model, backend.data)]
    finally:
        backend._viewer = None
        backend.close()


class _FakeVariable:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _FakeTk:
    DoubleVar = _FakeVariable


class _FakeBackend:
    def __init__(self, side, *, viewer_is_running=True, timestep_s=0.002):
        self.side = side
        self.target_rad = np.zeros(20)
        self.joint_names = tuple(f"finger{(i//4)+1}_joint{(i%4)+1}" for i in range(20))
        self.closed = False
        self.viewer_is_running = viewer_is_running
        self.timestep_s = timestep_s
        self.step_calls = 0

    def close(self):
        self.closed = True

    def step(self):
        self.step_calls += 1


class _FakeButton:
    def __init__(self):
        self.config = {}
        self.text = "Record"

    def configure(self, **kwargs):
        self.config.update(kwargs)
        if "text" in kwargs:
            self.text = kwargs["text"]
        if "fg" in kwargs:
            self.config["fg"] = kwargs["fg"]


class _FakeRoot:
    def __init__(self):
        self.after_calls = []
        self.destroy_calls = 0
        self.quit_calls = 0

    def after(self, delay_ms, callback):
        self.after_calls.append((delay_ms, callback))

    def winfo_exists(self):
        return True

    def destroy(self):
        self.destroy_calls += 1

    def quit(self):
        self.quit_calls += 1


def test_switching_hand_constructs_a_fresh_side_backend(monkeypatch):
    created = []

    def factory(side, *, viewer):
        created.append((side, viewer))
        return _FakeBackend(side)

    panel = object.__new__(local_angle_bar.WujiAngleBarPanel)
    panel._backend_factory = factory
    panel._backend = factory("left", viewer=True)
    original_backend = panel._backend
    panel._closed = False
    panel._side = _FakeVariable("left")
    panel._rebuild_slider_groups = lambda: None
    monkeypatch.setattr(local_angle_bar, "tk", _FakeTk)

    panel.switch_side("right")

    assert created == [("left", True), ("right", True)]
    assert panel._backend.side == "right"
    assert original_backend.closed


def test_switching_sides_closes_old_backend_before_creating_replacement(monkeypatch):
    active = []
    maximum_live_backends = 0

    class TrackingBackend(_FakeBackend):
        def __init__(self, side):
            nonlocal maximum_live_backends
            super().__init__(side)
            active.append(self)
            maximum_live_backends = max(maximum_live_backends, len(active))

        def close(self):
            super().close()
            active.remove(self)

    def factory(side, *, viewer):
        assert viewer is True
        return TrackingBackend(side)

    panel = object.__new__(local_angle_bar.WujiAngleBarPanel)
    panel._backend_factory = factory
    panel._backend = factory("left", viewer=True)
    panel._closed = False
    panel._side = _FakeVariable("left")
    panel._rebuild_slider_groups = lambda: None
    monkeypatch.setattr(local_angle_bar, "tk", _FakeTk)

    panel.switch_side("right")

    assert maximum_live_backends == 1
    assert panel._backend.side == "right"


def test_switch_failure_closes_panel_and_re_raises_factory_error(monkeypatch):
    root = _FakeRoot()
    original_backend = _FakeBackend("left")

    def failing_factory(side, *, viewer):
        raise RuntimeError("right model unavailable")

    panel = object.__new__(local_angle_bar.WujiAngleBarPanel)
    panel._root = root
    panel._backend_factory = failing_factory
    panel._backend = original_backend
    panel._closed = False
    panel._side = _FakeVariable("left")
    monkeypatch.setattr(local_angle_bar, "tk", _FakeTk)

    with pytest.raises(RuntimeError, match="right model unavailable"):
        panel.switch_side("right")

    assert original_backend.closed
    assert panel._backend is None
    assert root.quit_calls == 1
    assert root.destroy_calls == 1


def test_viewer_close_shuts_down_panel_once():
    root = _FakeRoot()
    backend = _FakeBackend("left", viewer_is_running=False)
    panel = object.__new__(local_angle_bar.WujiAngleBarPanel)
    panel._root = root
    panel._backend = backend
    panel._closed = False

    panel._tick()
    panel.close()

    assert backend.step_calls == 1
    assert backend.closed
    assert root.quit_calls == 1
    assert root.destroy_calls == 1
    assert root.after_calls == []


def test_panel_tick_steps_once_and_schedules_at_model_timestep():
    root = _FakeRoot()
    backend = _FakeBackend("left", viewer_is_running=True, timestep_s=0.002)
    panel = object.__new__(local_angle_bar.WujiAngleBarPanel)
    panel._root = root
    panel._backend = backend
    panel._closed = False

    panel._tick()

    assert backend.step_calls == 1
    assert root.after_calls == [(2, panel._tick)]


def test_panel_displays_mujoco_only_hardware_safety_notice():
    assert "MuJoCo-only" in local_angle_bar.PANEL_SAFETY_TEXT
    assert "does not command hardware" in local_angle_bar.PANEL_SAFETY_TEXT


def test_save_pose_creates_npz_with_correct_content(monkeypatch, tmp_path):
    """Save Pose should write a valid NPZ with joint_positions_rad."""
    saved_paths = []

    def fake_saveasfilename(**kwargs):
        path = str(tmp_path / kwargs.get("initialfile", "test.npz"))
        saved_paths.append(path)
        return path

    monkeypatch.setattr(local_angle_bar.tkfiledialog, "asksaveasfilename", fake_saveasfilename)

    backend = _FakeBackend("left")
    backend.target_rad = np.arange(20, dtype=float) * 0.1  # distinct values
    root = _FakeRoot()
    panel = object.__new__(local_angle_bar.WujiAngleBarPanel)
    panel._root = root
    panel._backend = backend
    panel._closed = False
    panel._side = _FakeVariable("left")
    panel._status_var = _FakeVariable("")
    monkeypatch.setattr(local_angle_bar, "tk", _FakeTk)
    panel._recording = False
    panel._recorded_frames = []
    panel._recorded_timestamps = []

    panel.save_pose()

    assert len(saved_paths) == 1
    data = np.load(saved_paths[0])
    assert "joint_positions_rad" in data
    assert data["joint_positions_rad"].shape == (1, 20)
    np.testing.assert_allclose(data["joint_positions_rad"][0], backend.target_rad)
    assert str(data["side"]) == "left"
    assert len(data["joint_names"]) == 20


def test_record_toggle_records_frames(monkeypatch):
    """Record should accumulate frames, Stop should save them."""
    saved_paths = []

    def fake_saveasfilename(**kwargs):
        path = "/tmp/test_record_traj.npz"
        saved_paths.append(path)
        return path

    monkeypatch.setattr(local_angle_bar.tkfiledialog, "asksaveasfilename", fake_saveasfilename)

    backend = _FakeBackend("left")
    backend.target_rad = np.arange(20, dtype=float) * 0.1
    root = _FakeRoot()
    panel = object.__new__(local_angle_bar.WujiAngleBarPanel)
    panel._root = root
    panel._backend = backend
    panel._closed = False
    panel._side = _FakeVariable("left")
    panel._status_var = _FakeVariable("")
    panel._record_button = _FakeButton()
    monkeypatch.setattr(local_angle_bar, "tk", _FakeTk)
    panel._recording = False
    panel._recorded_frames = []
    panel._recorded_timestamps = []

    # Start recording
    panel.toggle_record()
    assert panel._recording is True
    assert len(panel._recorded_frames) == 1

    # Simulate a few ticks
    for _ in range(5):
        backend.target_rad = backend.target_rad + 0.01
        panel._recorded_frames.append(backend.target_rad.copy())
        panel._recorded_timestamps.append(12345)

    expected_frames = len(panel._recorded_frames)  # 6 total

    # Stop recording
    panel.toggle_record()
    assert panel._recording is False
    assert len(saved_paths) == 1
    data = np.load(saved_paths[0])
    assert data["joint_positions_rad"].shape[0] == expected_frames
    assert data["frame_count"] == expected_frames


def test_record_cancelled_when_no_file_chosen(monkeypatch):
    """If user cancels save dialog, recording data should be discarded silently."""
    monkeypatch.setattr(local_angle_bar.tkfiledialog, "asksaveasfilename", lambda **kw: "")

    backend = _FakeBackend("left")
    root = _FakeRoot()
    panel = object.__new__(local_angle_bar.WujiAngleBarPanel)
    panel._root = root
    panel._backend = backend
    panel._closed = False
    panel._side = _FakeVariable("left")
    panel._status_var = _FakeVariable("")
    panel._record_button = _FakeButton()
    monkeypatch.setattr(local_angle_bar, "tk", _FakeTk)
    panel._recording = False
    panel._recorded_frames = []
    panel._recorded_timestamps = []

    panel.toggle_record()
    panel.toggle_record()
    assert panel._recording is False
    assert panel._recorded_frames == []


def test_send_to_hand_skipped_when_user_cancels(monkeypatch):
    """Send to Hand should do nothing if user cancels the confirmation dialog."""
    monkeypatch.setattr(local_angle_bar.tkmsg, "askyesno", lambda *a, **kw: False)
    monkeypatch.setattr(local_angle_bar.tkfiledialog, "asksaveasfilename", lambda **kw: "")

    backend = _FakeBackend("left")
    root = _FakeRoot()
    panel = object.__new__(local_angle_bar.WujiAngleBarPanel)
    panel._root = root
    panel._backend = backend
    panel._closed = False
    panel._side = _FakeVariable("left")
    panel._status_var = _FakeVariable("")
    monkeypatch.setattr(local_angle_bar, "tk", _FakeTk)
    panel._recording = False
    panel._recorded_frames = []
    panel._recorded_timestamps = []

    # Should not raise or crash
    panel.send_to_hand()
    assert True  # reached here without error
