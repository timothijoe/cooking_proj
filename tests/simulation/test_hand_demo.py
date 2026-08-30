import numpy as np

import twin_sim.tasks.hand_demo as hand_demo
from twin_sim.tasks.hand_demo import HandDemoConfig, run_hand_demo


def test_headless_hand_demo_returns_to_open_pose():
    result = run_hand_demo(
        HandDemoConfig(control_dt_s=0.01, close_duration_s=0.02, open_duration_s=0.02),
        viewer=False,
    )
    assert result.samples == 5
    np.testing.assert_allclose(result.final_target_rad, result.open_target_rad)


def test_viewer_presentation_focuses_left_hand_and_holds(monkeypatch):
    sleeps = []

    class Camera:
        azimuth = 0.0
        elevation = 0.0
        distance = 0.0
        lookat = np.zeros(3)

    class Viewer:
        cam = Camera()

        @staticmethod
        def is_running():
            return True

        @staticmethod
        def sync():
            pass

    class Robot:
        _viewer = Viewer()

    monkeypatch.setattr(hand_demo.time, "sleep", sleeps.append)
    config = HandDemoConfig(viewer_start_hold_s=2.0, viewer_end_hold_s=3.0)
    hand_demo._prepare_viewer(Robot(), config)
    hand_demo._finish_viewer(Robot(), config)
    assert Robot._viewer.cam.lookat[1] > 0.0
    assert sleeps == [2.0, 3.0]
