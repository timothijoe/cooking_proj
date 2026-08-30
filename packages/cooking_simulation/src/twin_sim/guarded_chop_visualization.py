from collections import deque
from typing import Sequence

import mujoco
import numpy as np

from twin_sim.viewer_trail import _init_sphere_geom, _position


class GuardedChopTrace:
    planned_knife_color = np.asarray((0.1, 0.35, 1.0, 0.9))
    actual_knife_color = np.asarray((0.0, 0.9, 1.0, 0.9))
    actual_guard_color = np.asarray((0.75, 0.1, 0.9, 0.95))
    trail_radius_m = 0.0015
    cut_mark_radius_m = 0.0012
    cut_mark_half_length_m = 0.005

    def __init__(
        self,
        viewer=None,
        *,
        max_points: int = 16,
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
        self._max_trail_geoms = 2 * (int(max_points) - 1)
        self._marker_stride = int(marker_stride)
        self._sample_index = 0
        self.planned_knife = deque(maxlen=int(max_points))
        self.actual_knife = deque(maxlen=int(max_points))
        self.actual_guard = deque(maxlen=int(max_points))
        self.cut_points: tuple[np.ndarray, ...] = ()
        self._last_drawn_actual_knife = None
        self._last_drawn_actual_guard = None
        self._trail_geom_start = None
        self._trail_geom_count = 0
        self._next_trail_geom = 0
        self._last_overlay_status = None
        self.phase = ""
        self.cut_index = 0
        self.minimum_distance_m = float("inf")
        self.cut_allowed = False
        self.abort_reason = ""
        self._replay_rate = None

    def set_plan(
        self, cut_points: Sequence[Sequence[float]]
    ) -> None:
        points = tuple(
            _position(np.asarray(point, dtype=float))
            for point in cut_points
        )
        if len(points) != 5:
            raise ValueError("guarded chop plan must contain five cut points")
        self.cut_points = points
        self.planned_knife.clear()
        self.planned_knife.extend(point.copy() for point in points)
        vertical = np.array((0.0, 0.0, self.cut_mark_half_length_m))
        segments = [
            (
                point - vertical,
                point + vertical,
                self.planned_knife_color,
                self.cut_mark_radius_m,
            )
            for point in points
        ]
        segments.extend(
            (start, end, self.planned_knife_color, self.trail_radius_m)
            for start, end in zip(points[:-1], points[1:], strict=True)
        )
        self._draw_segments(segments)

    def append(
        self,
        *,
        actual_knife: Sequence[float],
        actual_guard: Sequence[float],
        phase: str,
        cut_index: int,
        minimum_distance_m: float,
        cut_allowed: bool,
    ) -> None:
        knife_point, guard_point = (
            _position(np.asarray(value, dtype=float))
            for value in (actual_knife, actual_guard)
        )
        self.actual_knife.append(knife_point)
        self.actual_guard.append(guard_point)
        self.phase = str(phase)
        self.cut_index = int(cut_index)
        self.minimum_distance_m = min(
            self.minimum_distance_m, float(minimum_distance_m)
        )
        self.cut_allowed = bool(cut_allowed)
        if self._sample_index % self._marker_stride == 0:
            self._update_overlay()
            segments = []
            if self._last_drawn_actual_knife is not None:
                segments.append(
                    (
                        self._last_drawn_actual_knife,
                        knife_point,
                        self.actual_knife_color,
                        self.trail_radius_m,
                    )
                )
            if self._last_drawn_actual_guard is not None:
                segments.append(
                    (
                        self._last_drawn_actual_guard,
                        guard_point,
                        self.actual_guard_color,
                        self.trail_radius_m,
                    )
                )
            self._draw_trail_segments(segments)
            self._last_drawn_actual_knife = knife_point.copy()
            self._last_drawn_actual_guard = guard_point.copy()
        self._sample_index += 1

    def set_abort(self, reason: str) -> None:
        self.phase = "aborted"
        self.cut_allowed = False
        self.abort_reason = str(reason)
        self._update_overlay()

    def begin_replay(self, rate: float) -> None:
        playback_rate = float(rate)
        if not np.isfinite(playback_rate) or playback_rate <= 0.0:
            raise ValueError("rate must be positive and finite")
        viewer = self._viewer
        if (
            viewer is not None
            and getattr(viewer, "user_scn", None) is not None
            and self._trail_geom_start is not None
        ):
            with viewer.lock():
                viewer.user_scn.ngeom = self._trail_geom_start
        self.actual_knife.clear()
        self.actual_guard.clear()
        self._sample_index = 0
        self._last_drawn_actual_knife = None
        self._last_drawn_actual_guard = None
        self._trail_geom_count = 0
        self._next_trail_geom = 0
        self._last_overlay_status = None
        self.minimum_distance_m = float("inf")
        self.cut_allowed = False
        self.abort_reason = ""
        self.phase = ""
        self.cut_index = 0
        self._replay_rate = playback_rate
        self._update_overlay()

    def _update_overlay(self) -> None:
        viewer = self._viewer
        if viewer is None or not hasattr(viewer, "set_texts"):
            return
        replay = (
            f"replay {self._replay_rate:.1f}x  "
            if self._replay_rate is not None
            else ""
        )
        status = replay + (
            f"cut={self.cut_index}/5  phase={self.phase}  "
            f"distance={self.minimum_distance_m:.3f} m  "
            f"cut_allowed={self.cut_allowed}"
        )
        if self.abort_reason:
            status += f"  abort={self.abort_reason}"
        if status == self._last_overlay_status:
            return
        viewer.set_texts((None, None, "Guarded chop", status))
        self._last_overlay_status = status

    def _draw_segments(self, segments) -> None:
        viewer = self._viewer
        if viewer is None or getattr(viewer, "user_scn", None) is None:
            return
        with viewer.lock():
            scene = viewer.user_scn
            for start, end, color, radius_m in segments:
                if np.array_equal(start, end):
                    continue
                if scene.ngeom >= scene.maxgeom:
                    return
                _init_capsule_between(
                    scene.geoms[scene.ngeom],
                    start,
                    end,
                    radius_m,
                    color,
                )
                scene.ngeom += 1

    def _draw_trail_segments(self, segments) -> None:
        viewer = self._viewer
        if viewer is None or getattr(viewer, "user_scn", None) is None:
            return
        with viewer.lock():
            scene = viewer.user_scn
            if self._trail_geom_start is None:
                self._trail_geom_start = scene.ngeom
            capacity = min(
                self._max_trail_geoms,
                scene.maxgeom - self._trail_geom_start,
            )
            if capacity <= 0:
                return
            for start, end, color, radius_m in segments:
                if np.array_equal(start, end):
                    continue
                if self._trail_geom_count < capacity:
                    geom_index = (
                        self._trail_geom_start + self._trail_geom_count
                    )
                    self._trail_geom_count += 1
                    scene.ngeom += 1
                else:
                    geom_index = (
                        self._trail_geom_start + self._next_trail_geom
                    )
                    self._next_trail_geom = (
                        self._next_trail_geom + 1
                    ) % capacity
                _init_capsule_between(
                    scene.geoms[geom_index],
                    start,
                    end,
                    radius_m,
                    color,
                )


def _init_capsule_between(
    geom,
    start: np.ndarray,
    end: np.ndarray,
    radius_m: float,
    rgba: np.ndarray,
) -> None:
    midpoint = (start + end) * 0.5
    _init_sphere_geom(geom, midpoint, radius_m, rgba)
    try:
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            radius_m,
            start,
            end,
        )
    except TypeError:
        direction = end - start
        length = float(np.linalg.norm(direction))
        z_axis = direction / length
        helper = (
            np.array((1.0, 0.0, 0.0))
            if abs(z_axis[0]) < 0.9
            else np.array((0.0, 1.0, 0.0))
        )
        x_axis = np.cross(helper, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        geom.size[:] = (radius_m, radius_m, length * 0.5)
        geom.pos[:] = midpoint
        rotation = np.column_stack((x_axis, y_axis, z_axis))
        try:
            geom.mat[:] = rotation.ravel()
        except ValueError:
            geom.mat.flat = rotation.ravel()
