from collections import deque
from typing import Sequence

import numpy as np

from twin_sim.viewer_trail import _init_sphere_geom, _position


class PickPlaceTrace:
    """Bounded pick-place traces with optional passive-Viewer markers."""

    planned_color = np.asarray((0.1, 0.35, 1.0, 0.9), dtype=float)
    actual_palm_color = np.asarray((0.0, 0.9, 1.0, 0.9), dtype=float)
    actual_cube_color = np.asarray((1.0, 0.4, 0.05, 0.95), dtype=float)

    def __init__(
        self,
        viewer=None,
        *,
        max_points: int = 3000,
        marker_stride: int = 10,
    ) -> None:
        for name, value in (
            ("max_points", max_points),
            ("marker_stride", marker_stride),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        self._viewer = viewer
        self._marker_stride = int(marker_stride)
        self._sample_index = 0
        self.planned_palm: deque[np.ndarray] = deque(maxlen=int(max_points))
        self.actual_palm: deque[np.ndarray] = deque(maxlen=int(max_points))
        self.actual_cube: deque[np.ndarray] = deque(maxlen=int(max_points))
        self.phase = ""
        self.contact_count = 0
        self.grasp_ready = False
        self.abort_reason = ""

    def append(
        self,
        *,
        planned_palm: Sequence[float],
        actual_palm: Sequence[float],
        actual_cube: Sequence[float],
        phase: str = "",
        contact_count: int = 0,
        grasp_ready: bool = False,
        abort_reason: str = "",
    ) -> None:
        planned = _position(np.asarray(planned_palm, dtype=float))
        actual = _position(np.asarray(actual_palm, dtype=float))
        cube = _position(np.asarray(actual_cube, dtype=float))
        self.planned_palm.append(planned)
        self.actual_palm.append(actual)
        self.actual_cube.append(cube)
        self.phase = str(phase)
        self.contact_count = int(contact_count)
        self.grasp_ready = bool(grasp_ready)
        self.abort_reason = str(abort_reason)
        self._update_overlay()
        if self._sample_index % self._marker_stride == 0:
            self._draw(
                (
                    (planned, self.planned_color, 0.006),
                    (actual, self.actual_palm_color, 0.0045),
                    (cube, self.actual_cube_color, 0.004),
                )
            )
        self._sample_index += 1

    def set_abort(self, reason: str) -> None:
        self.phase = "aborted"
        self.abort_reason = str(reason)
        self._update_overlay()

    def _update_overlay(self) -> None:
        viewer = self._viewer
        if viewer is None or not hasattr(viewer, "set_texts"):
            return
        status = (
            f"phase={self.phase}  contacts={self.contact_count}  "
            f"grasp_ready={self.grasp_ready}"
        )
        if self.abort_reason:
            status += f"  abort={self.abort_reason}"
        viewer.set_texts((None, None, "Pick-place", status))

    def _draw(
        self,
        markers: tuple[tuple[np.ndarray, np.ndarray, float], ...],
    ) -> None:
        viewer = self._viewer
        if viewer is None or getattr(viewer, "user_scn", None) is None:
            return
        with viewer.lock():
            scene = viewer.user_scn
            for position, color, size_m in markers:
                if scene.ngeom >= scene.maxgeom:
                    return
                _init_sphere_geom(
                    scene.geoms[scene.ngeom],
                    position,
                    size_m,
                    color,
                )
                scene.ngeom += 1
