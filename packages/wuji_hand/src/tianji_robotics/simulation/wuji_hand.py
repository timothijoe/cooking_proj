"""Official hand-only MuJoCo backend for the first-generation left Wuji Hand."""

import time

import mujoco
import numpy as np

from tianji_robotics.simulation.paths import official_wuji_left_mjcf
from tianji_robotics.wuji_hand.names import HAND_JOINT_NAMES


OFFICIAL_JOINT_NAMES = tuple(name.removeprefix("left_") for name in HAND_JOINT_NAMES)


class MujocoWujiHand:
    """Control only the official 20-DOF Wuji Hand model; no arm is loaded."""

    def __init__(self, *, viewer: bool = False) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(official_wuji_left_mjcf()))
        self.data = mujoco.MjData(self.model)
        self._joint_ids = self._ids(mujoco.mjtObj.mjOBJ_JOINT)
        self._actuator_ids = self._ids(mujoco.mjtObj.mjOBJ_ACTUATOR)
        self._qpos_ids = self.model.jnt_qposadr[self._joint_ids].copy()
        self.timestep_s = float(self.model.opt.timestep)
        self.realtime_paced = bool(viewer)
        self.range_tolerance_rad = 0.0
        self._closed = False
        self._viewer = None
        if viewer:
            from mujoco import viewer as mujoco_viewer

            self._viewer = mujoco_viewer.launch_passive(self.model, self.data)
            self._viewer.cam.azimuth = 180.0
            self._viewer.cam.elevation = -20.0
            self._viewer.cam.distance = 0.5
            self._viewer.cam.lookat[:] = (0.0, 0.0, 0.05)
            self._viewer.sync()

    def _ids(self, object_type: mujoco.mjtObj) -> np.ndarray:
        ids = np.asarray(
            [mujoco.mj_name2id(self.model, object_type, name) for name in OFFICIAL_JOINT_NAMES],
            dtype=np.int32,
        )
        if np.any(ids < 0) or len(set(ids.tolist())) != 20:
            raise RuntimeError("official Wuji model does not expose the canonical 20 joints")
        return ids

    @property
    def joint_ranges_rad(self) -> dict[str, tuple[float, float]]:
        joint = self.model.jnt_range[self._joint_ids]
        ctrl = self.model.actuator_ctrlrange[self._actuator_ids]
        return {
            name: (max(float(j[0]), float(c[0])), min(float(j[1]), float(c[1])))
            for name, j, c in zip(HAND_JOINT_NAMES, joint, ctrl, strict=True)
        }

    def read_position_rad(self) -> np.ndarray:
        self._require_open()
        return self.data.qpos[self._qpos_ids].copy()

    def read_target_position_rad(self) -> np.ndarray:
        self._require_open()
        return self.data.ctrl[self._actuator_ids].copy()

    def command_position_rad(self, target: np.ndarray) -> None:
        self._require_open()
        values = np.asarray(target, dtype=float)
        if values.shape != (20,) or not np.isfinite(values).all():
            raise ValueError("Wuji hand target must contain 20 finite radians")
        for value, (lower, upper), name in zip(values, self.joint_ranges_rad.values(), HAND_JOINT_NAMES, strict=True):
            if value < lower or value > upper:
                raise ValueError(f"Wuji hand target for {name} is outside range [{lower}, {upper}]")
        self.data.ctrl[self._actuator_ids] = values

    def step(self, duration_s: float) -> None:
        self._require_open()
        if not np.isclose(float(duration_s), self.timestep_s):
            raise ValueError("hand-only backend step must equal the MuJoCo timestep")
        mujoco.mj_step(self.model, self.data)
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()
            time.sleep(self.timestep_s)

    def close(self) -> None:
        if self._closed:
            return
        if self._viewer is not None:
            self._viewer.close()
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("hand-only Wuji backend is closed")
