"""Derived official Wuji Hand scene with a table and movable palm."""

from dataclasses import dataclass
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from tianji_robotics.simulation.paths import official_wuji_left_mjcf
from tianji_robotics.simulation.wuji_hand import OFFICIAL_JOINT_NAMES
from tianji_robotics.wuji_hand.names import HAND_JOINT_NAMES


@dataclass(frozen=True)
class ContactDiagnostics:
    fingertip_heights_m: np.ndarray
    thumb_height_m: float
    penetration_m: float


class TabletopWujiHand:
    def __init__(self, *, viewer: bool = False, table_height_m: float = 0.0) -> None:
        self.table_height_m = float(table_height_m)
        if not np.isfinite(self.table_height_m):
            raise ValueError("table height must be finite")
        xml = self._derived_xml(self.table_height_m)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self._joint_ids = self._ids(mujoco.mjtObj.mjOBJ_JOINT, OFFICIAL_JOINT_NAMES)
        self._actuator_ids = self._ids(mujoco.mjtObj.mjOBJ_ACTUATOR, OFFICIAL_JOINT_NAMES)
        self._qpos_ids = self.model.jnt_qposadr[self._joint_ids].copy()
        self._tip_site_ids = self._ids(mujoco.mjtObj.mjOBJ_SITE, tuple(f"finger{i}_contact" for i in range(2, 6)))
        self._thumb_site_id = int(self._ids(mujoco.mjtObj.mjOBJ_SITE, ("thumb_clearance",))[0])
        self._palmar_site_id = int(self._ids(mujoco.mjtObj.mjOBJ_SITE, ("palmar_reference",))[0])
        self._dorsal_site_id = int(self._ids(mujoco.mjtObj.mjOBJ_SITE, ("dorsal_reference",))[0])
        self._long_finger_root_body_ids = self._ids(
            mujoco.mjtObj.mjOBJ_BODY,
            tuple(f"finger{i}_link1" for i in range(2, 6)),
        )
        self._table_geom_id = int(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table")
        )
        palm_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "palm_link")
        self._mocap_id = int(self.model.body_mocapid[palm_id])
        self.timestep_s = float(self.model.opt.timestep)
        self.realtime_paced = bool(viewer)
        self.range_tolerance_rad = 0.0
        self._closed = False
        self._viewer = None
        if viewer:
            from mujoco import viewer as mujoco_viewer
            self._viewer = mujoco_viewer.launch_passive(self.model, self.data)
            self._viewer.cam.azimuth = 90.0
            self._viewer.cam.elevation = -8.0
            self._viewer.cam.distance = 0.34
            self._viewer.cam.lookat[:] = (-0.09, 0.0, 0.04)
            self._viewer.sync()

    @staticmethod
    def _derived_xml(table_height_m: float) -> str:
        source = official_wuji_left_mjcf()
        root = ET.parse(source).getroot()
        compiler = root.find("compiler")
        if compiler is None:
            raise RuntimeError("official Wuji model has no compiler element")
        compiler.set("meshdir", str((source.parent / compiler.get("meshdir", "")).resolve()))
        world = root.find("worldbody")
        palm = world.find("body[@name='palm_link']") if world is not None else None
        if world is None or palm is None:
            raise RuntimeError("official Wuji model has no palm_link")
        palm.set("mocap", "true")
        ET.SubElement(palm, "site", name="palmar_reference", pos="0 0 0.015", size="0.002", rgba="0.9 0.2 0.2 1")
        ET.SubElement(palm, "site", name="dorsal_reference", pos="0 0 -0.015", size="0.002", rgba="0.2 0.2 0.9 1")
        ET.SubElement(world, "geom", name="table", type="plane", pos=f"0 0 {table_height_m:.17g}", size="0.5 0.5 0.02", rgba="0.55 0.55 0.58 1", contype="1", conaffinity="1")
        for finger in range(1, 6):
            body = palm.find(f".//body[@name='finger{finger}_link4']")
            tip = body.find(f"geom[@mesh='finger{finger}_tip_link']") if body is not None else None
            if body is None or tip is None:
                raise RuntimeError(f"official Wuji model has no finger{finger} tip")
            ET.SubElement(body, "site", name="thumb_clearance" if finger == 1 else f"finger{finger}_contact", pos=tip.get("pos", "0 0 0"), size="0.002", rgba="1 0.2 0.2 1" if finger == 1 else "0.1 0.9 0.2 1")
        return ET.tostring(root, encoding="unicode")

    def _ids(self, kind: mujoco.mjtObj, names: tuple[str, ...]) -> np.ndarray:
        ids = np.asarray([mujoco.mj_name2id(self.model, kind, name) for name in names], dtype=np.int32)
        if np.any(ids < 0):
            raise RuntimeError(f"derived Wuji scene is missing names: {names}")
        return ids

    @property
    def joint_ranges_rad(self) -> dict[str, tuple[float, float]]:
        joint = self.model.jnt_range[self._joint_ids]
        ctrl = self.model.actuator_ctrlrange[self._actuator_ids]
        return {name: (max(float(j[0]), float(c[0])), min(float(j[1]), float(c[1]))) for name, j, c in zip(HAND_JOINT_NAMES, joint, ctrl, strict=True)}

    def command_pose(self, joints_rad, palm_position_m, palm_quaternion_wxyz) -> None:
        joints = self._validated_joints(joints_rad)
        position, quaternion = self._validated_palm_pose(palm_position_m, palm_quaternion_wxyz)
        self.data.ctrl[self._actuator_ids] = joints
        self.data.mocap_pos[self._mocap_id] = position
        self.data.mocap_quat[self._mocap_id] = quaternion
        mujoco.mj_forward(self.model, self.data)

    def set_kinematic_pose(self, joints_rad, palm_position_m, palm_quaternion_wxyz) -> None:
        self.command_pose(joints_rad, palm_position_m, palm_quaternion_wxyz)
        self.data.qpos[self._qpos_ids] = self._validated_joints(joints_rad)
        mujoco.mj_forward(self.model, self.data)

    def _validated_joints(self, values) -> np.ndarray:
        joints = np.asarray(values, dtype=float)
        if joints.shape != (20,) or not np.isfinite(joints).all():
            raise ValueError("joint pose must contain 20 finite radians")
        return joints.copy()

    @staticmethod
    def _validated_palm_pose(position, quaternion) -> tuple[np.ndarray, np.ndarray]:
        pos = np.asarray(position, dtype=float)
        quat = np.asarray(quaternion, dtype=float)
        norm = float(np.linalg.norm(quat)) if quat.shape == (4,) else 0.0
        if pos.shape != (3,) or not np.isfinite(pos).all() or quat.shape != (4,) or not np.isfinite(quat).all() or norm <= 1e-12:
            raise ValueError("palm pose must contain a finite position and nonzero quaternion")
        return pos.copy(), quat / norm

    def read_target_position_rad(self) -> np.ndarray:
        return self.data.ctrl[self._actuator_ids].copy()

    def read_palm_position_m(self) -> np.ndarray:
        return self.data.mocap_pos[self._mocap_id].copy()

    def fingertip_positions_m(self) -> np.ndarray:
        return self.data.site_xpos[self._tip_site_ids].copy()

    def thumb_position_m(self) -> np.ndarray:
        return self.data.site_xpos[self._thumb_site_id].copy()

    def palmar_reference_position_m(self) -> np.ndarray:
        return self.data.site_xpos[self._palmar_site_id].copy()

    def dorsal_reference_position_m(self) -> np.ndarray:
        return self.data.site_xpos[self._dorsal_site_id].copy()

    def long_finger_root_positions_m(self) -> np.ndarray:
        return self.data.xpos[self._long_finger_root_body_ids].copy()

    def fingertip_position_jacobian(self) -> np.ndarray:
        """Return stacked world-position Jacobians in canonical joint order."""
        blocks = []
        for site_id in self._tip_site_ids:
            jacobian = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacobian, None, int(site_id))
            blocks.append(jacobian[:, self.model.jnt_dofadr[self._joint_ids]])
        return np.vstack(blocks)

    def contact_diagnostics(self) -> ContactDiagnostics:
        heights = self.fingertip_positions_m()[:, 2] - self.table_height_m
        thumb = float(self.thumb_position_m()[2] - self.table_height_m)
        return ContactDiagnostics(heights.copy(), thumb, max(0.0, float(-heights.min())))

    def minimum_hand_table_clearance_m(self) -> float:
        distances = [
            float(self.data.contact[index].dist)
            for index in range(self.data.ncon)
            if self._table_geom_id
            in (self.data.contact[index].geom1, self.data.contact[index].geom2)
        ]
        return min(distances, default=float("inf"))

    def maximum_table_penetration_m(self) -> float:
        return max(0.0, -self.minimum_hand_table_clearance_m())

    def long_fingertip_table_distances_m(self) -> np.ndarray:
        """Return the shallowest contact distance for each long fingertip body."""
        distances = np.full(4, np.nan)
        tip_body_ids = {
            int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"finger{i}_link4")): i - 2
            for i in range(2, 6)
        }
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if self._table_geom_id not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == self._table_geom_id else contact.geom1
            finger = tip_body_ids.get(int(self.model.geom_bodyid[other]))
            if finger is None:
                continue
            distance = float(contact.dist)
            if not np.isfinite(distances[finger]) or distance < distances[finger]:
                distances[finger] = distance
        return distances

    def step(self, duration_s: float) -> None:
        if not np.isclose(float(duration_s), self.timestep_s):
            raise ValueError("tabletop backend step must equal the MuJoCo timestep")
        mujoco.mj_step(self.model, self.data)
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync(); time.sleep(self.timestep_s)

    @property
    def has_viewer(self) -> bool:
        return self._viewer is not None

    def viewer_is_running(self) -> bool:
        return self._viewer is not None and self._viewer.is_running()

    def wait_until_viewer_closes(self) -> None:
        while self.viewer_is_running():
            time.sleep(0.05)

    def close(self) -> None:
        if self._closed: return
        if self.viewer_is_running(): self._viewer.close()
        self._closed = True
