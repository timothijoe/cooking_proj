import numpy as np
import pytest

from twin_sim.guarded_chop_visualization import GuardedChopTrace


class FakeGeom:
    def __init__(self):
        self.type = 0
        self.size = np.zeros(3)
        self.pos = np.zeros(3)
        self.mat = np.zeros((3, 3))
        self.rgba = np.zeros(4)
        self.category = 0
        self.segid = 0
        self.objtype = 0
        self.objid = 0
        self.dataid = 0
        self.emission = 0.0
        self.specular = 0.0
        self.shininess = 0.0
        self.reflectance = 0.0


class FakeScene:
    def __init__(self, maxgeom=64):
        self.ngeom = 0
        self.maxgeom = maxgeom
        self.geoms = [FakeGeom() for _ in range(self.maxgeom)]


class FakeLock:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


class FakeViewer:
    def __init__(self, maxgeom=64):
        self.user_scn = FakeScene(maxgeom)
        self.texts = None
        self.text_call_count = 0

    @staticmethod
    def lock():
        return FakeLock()

    def set_texts(self, texts):
        self.texts = texts
        self.text_call_count += 1


def test_trace_draws_three_trails_and_five_compact_cut_marks():
    viewer = FakeViewer()
    trace = GuardedChopTrace(viewer, marker_stride=1)
    trace.set_plan(
        [(0.7, y, 0.33) for y in (0.04, 0.02, 0.0, -0.02, -0.04)]
    )

    trace.append(
        actual_knife=(0.7, 0.04, 0.35),
        actual_guard=(0.6, 0.10, 0.36),
        phase="hand_open",
        cut_index=1,
        minimum_distance_m=0.06,
        cut_allowed=False,
    )
    trace.append(
        actual_knife=(0.7, 0.02, 0.34),
        actual_guard=(0.6, 0.08, 0.36),
        phase="hand_open",
        cut_index=1,
        minimum_distance_m=0.05,
        cut_allowed=False,
    )

    assert len(trace.cut_points) == 5
    assert not hasattr(trace, "guard_target")
    assert viewer.user_scn.ngeom == 11
    colors = [geom.rgba for geom in viewer.user_scn.geoms[:11]]
    assert (
        sum(
            np.allclose(color, trace.planned_knife_color)
            for color in colors
        )
        == 9
    )
    assert (
        sum(
            np.allclose(color, trace.actual_knife_color)
            for color in colors
        )
        == 1
    )
    assert (
        sum(
            np.allclose(color, trace.actual_guard_color)
            for color in colors
        )
        == 1
    )
    cut_marks = viewer.user_scn.geoms[:5]
    assert all(mark.size[0] <= 0.0015 for mark in cut_marks)
    assert all(mark.size[2] <= 0.006 for mark in cut_marks)
    assert viewer.texts[2] == "Guarded chop"
    assert "phase=hand_open" in viewer.texts[3]


def test_plan_uses_every_limited_scene_slot_for_cut_marks_first():
    viewer = FakeViewer()
    viewer.user_scn.maxgeom = 4
    trace = GuardedChopTrace(viewer)

    trace.set_plan(
        [(0.7, y, 0.33) for y in (0.04, 0.02, 0.0, -0.02, -0.04)]
    )

    assert viewer.user_scn.ngeom == 4
    for geom in viewer.user_scn.geoms[:4]:
        assert np.isclose(geom.size[0], trace.cut_mark_radius_m)
        assert np.isclose(geom.size[2], trace.cut_mark_half_length_m)


def test_append_rejects_removed_planned_knife_argument():
    trace = GuardedChopTrace()

    with pytest.raises(TypeError, match="planned_knife"):
        trace.append(
            planned_knife=(0.7, 0.04, 0.36),
            actual_knife=(0.7, 0.04, 0.35),
            actual_guard=(0.6, 0.10, 0.36),
            phase="hand_open",
            cut_index=1,
            minimum_distance_m=0.06,
            cut_allowed=False,
        )


def test_abort_reason_replaces_live_phase():
    viewer = FakeViewer()
    trace = GuardedChopTrace(viewer)

    trace.set_abort("clearance lost")

    assert "phase=aborted" in viewer.texts[3]
    assert "clearance lost" in viewer.texts[3]


def test_begin_replay_clears_actual_trails_and_labels_overlay():
    viewer = FakeViewer()
    trace = GuardedChopTrace(viewer, marker_stride=1)
    trace.set_plan(
        [(0.7, y, 0.33) for y in (0.04, 0.02, 0.0, -0.02, -0.04)]
    )
    trace.append(
        actual_knife=(0.7, 0.04, 0.35),
        actual_guard=(0.6, 0.10, 0.36),
        phase="cut_down",
        cut_index=1,
        minimum_distance_m=0.04,
        cut_allowed=True,
    )
    trace.append(
        actual_knife=(0.7, 0.02, 0.34),
        actual_guard=(0.6, 0.08, 0.36),
        phase="cut_down",
        cut_index=1,
        minimum_distance_m=0.04,
        cut_allowed=True,
    )

    trace.begin_replay(2.0)

    assert len(trace.cut_points) == 5
    assert list(trace.actual_knife) == []
    assert list(trace.actual_guard) == []
    assert viewer.user_scn.ngeom == 9
    assert "replay 2.0x" in viewer.texts[3]


@pytest.mark.parametrize("phase", ("low_guard_shift", "knife_lift_shift"))
def test_overlay_displays_low_guard_sequence_phases(phase):
    viewer = FakeViewer()
    trace = GuardedChopTrace(viewer)

    trace.append(
        actual_knife=(0.7, 0.04, 0.36),
        actual_guard=(0.6, 0.10, 0.36),
        phase=phase,
        cut_index=2,
        minimum_distance_m=0.04,
        cut_allowed=False,
    )

    assert f"phase={phase}" in viewer.texts[3]


def test_overlay_keeps_cumulative_minimum_distance():
    viewer = FakeViewer()
    trace = GuardedChopTrace(viewer, marker_stride=1)
    common = dict(
        actual_knife=(0.0, 0.0, 0.0),
        actual_guard=(0.0, 0.0, 0.0),
        phase="cut_down",
        cut_index=1,
        cut_allowed=True,
    )
    trace.append(minimum_distance_m=0.025, **common)
    trace.append(minimum_distance_m=0.040, **common)

    assert "distance=0.025" in viewer.texts[3]


def test_trace_bounds_viewer_work_as_sample_history_grows():
    viewer = FakeViewer()
    trace = GuardedChopTrace(
        viewer,
        max_points=4,
        marker_stride=1,
    )
    trace.set_plan(
        [(0.7, y, 0.33) for y in (0.04, 0.02, 0.0, -0.02, -0.04)]
    )

    for index in range(100):
        trace.append(
            actual_knife=(0.7, 0.001 * index, 0.35),
            actual_guard=(0.6, 0.001 * index, 0.36),
            phase="hand_shift",
            cut_index=1,
            minimum_distance_m=0.04,
            cut_allowed=False,
        )

    planned_geoms = 9
    max_actual_geoms = 2 * (4 - 1)
    assert viewer.user_scn.ngeom <= planned_geoms + max_actual_geoms
    assert viewer.text_call_count == 1


def test_default_trace_keeps_a_compact_visible_tail():
    viewer = FakeViewer(maxgeom=512)
    trace = GuardedChopTrace(viewer)
    trace.set_plan(
        [(0.7, y, 0.33) for y in (0.04, 0.02, 0.0, -0.02, -0.04)]
    )

    for index in range(1000):
        trace.append(
            actual_knife=(0.7, 0.0001 * index, 0.35),
            actual_guard=(0.6, 0.0001 * index, 0.36),
            phase="hand_shift",
            cut_index=1,
            minimum_distance_m=0.04,
            cut_allowed=False,
        )

    assert trace.actual_knife.maxlen == 16
    assert viewer.user_scn.ngeom <= 9 + 2 * (16 - 1)
