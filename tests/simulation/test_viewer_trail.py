import numpy as np

import twin_sim.viewer_trail as viewer_trail


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
    def __init__(self, maxgeom=20):
        self.ngeom = 0
        self.maxgeom = maxgeom
        self.geoms = [FakeGeom() for _ in range(maxgeom)]


class FakeLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return None


class FakeViewer:
    def __init__(self):
        self.user_scn = FakeScene()

    @staticmethod
    def lock():
        return FakeLock()


def test_trail_draws_planned_cyan_and_actual_orange_markers():
    viewer = FakeViewer()
    trail = viewer_trail.ViewerTrajectoryTrail(viewer, sample_stride=2)
    planned = [
        np.array((float(index), 0.0, 0.3))
        for index in range(5)
    ]

    trail.draw_target_plan(planned)
    for index, point in enumerate(planned):
        trail.record_actual(point + np.array((0.0, 0.001, 0.0)), index)

    scene = viewer.user_scn
    assert scene.ngeom == 6
    np.testing.assert_allclose(scene.geoms[0].pos, planned[0])
    np.testing.assert_allclose(scene.geoms[3].pos, planned[0] + (0.0, 0.001, 0.0))
    assert not np.allclose(scene.geoms[0].rgba, scene.geoms[3].rgba)
    assert scene.geoms[0].size[0] > scene.geoms[3].size[0]


def test_trail_is_noop_without_viewer():
    trail = viewer_trail.ViewerTrajectoryTrail(None, sample_stride=2)

    trail.draw_target_plan([np.zeros(3)])
    trail.record_actual(np.ones(3), 0)
