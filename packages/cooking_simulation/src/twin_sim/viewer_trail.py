from typing import Iterable

import mujoco
import numpy as np


class ViewerTrajectoryTrail:
    """Render planned and actual TCP paths in a passive MuJoCo viewer."""

    target_color = np.array((0.0, 0.85, 1.0, 0.92), dtype=float)
    actual_color = np.array((1.0, 0.35, 0.0, 0.95), dtype=float)
    target_size_m = 0.009
    actual_size_m = 0.006

    def __init__(self, viewer, *, sample_stride: int = 10) -> None:
        if (
            isinstance(sample_stride, bool)
            or not isinstance(sample_stride, (int, np.integer))
            or sample_stride <= 0
        ):
            raise ValueError("sample_stride must be a positive integer")
        self._viewer = viewer
        self._sample_stride = int(sample_stride)

    def draw_target_plan(self, positions: Iterable[np.ndarray]) -> None:
        points = [
            _position(point)
            for index, point in enumerate(positions)
            if index % self._sample_stride == 0
        ]
        self._append_many(points, self.target_color, self.target_size_m)

    def record_actual(self, position: np.ndarray, step_index: int) -> None:
        if step_index % self._sample_stride:
            return
        self._append_many(
            (_position(position),),
            self.actual_color,
            self.actual_size_m,
        )

    def _append_many(
        self,
        positions: Iterable[np.ndarray],
        rgba: np.ndarray,
        size_m: float,
    ) -> None:
        viewer = self._viewer
        if viewer is None or getattr(viewer, "user_scn", None) is None:
            return
        with viewer.lock():
            scene = viewer.user_scn
            for position in positions:
                if scene.ngeom >= scene.maxgeom:
                    return
                geom = scene.geoms[scene.ngeom]
                _init_sphere_geom(geom, position, size_m, rgba)
                scene.ngeom += 1


def _position(value: np.ndarray) -> np.ndarray:
    try:
        position = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("trail position must contain 3 finite values") from error
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("trail position must contain 3 finite values")
    return position.copy()


def _init_sphere_geom(
    geom,
    position: np.ndarray,
    size_m: float,
    rgba: np.ndarray,
) -> None:
    size = np.full(3, size_m, dtype=float)
    rotation = np.eye(3)
    try:
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            size,
            position,
            rotation,
            rgba,
        )
    except TypeError:
        geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
        geom.size[:] = size
        geom.pos[:] = position
        try:
            geom.mat[:] = rotation.ravel()
        except ValueError:
            geom.mat.flat = rotation.ravel()
        geom.rgba[:] = rgba
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    for attribute in ("segid", "objtype", "objid", "dataid"):
        if hasattr(geom, attribute):
            setattr(geom, attribute, -1)
    for attribute in ("emission", "specular", "shininess", "reflectance"):
        if hasattr(geom, attribute):
            setattr(geom, attribute, 0.0)
