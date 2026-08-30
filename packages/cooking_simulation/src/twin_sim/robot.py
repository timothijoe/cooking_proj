from pathlib import Path
import threading
import time
from typing import Sequence

import mujoco
import numpy as np

from twin_sim.model import SimulationModel
from twin_sim.hand import DEFAULT_OPEN_RAD, LeftHandController
from twin_sim.kinematics import Kinematics


RIGHT_HOME_RAD = np.array(
    (0.4, -1.3, 0.08, -1.606525, 0.057176, 0.79256, 1.5), dtype=float
)
_VIEWER_CLOSE_TIMEOUT_S = 1.0


class NumericalSafetyError(RuntimeError):
    pass


class RightArmRobot:
    def __init__(self, model_path: Path | None = None, viewer: bool = False):
        self.sim = SimulationModel.load(model_path)
        self._viewer = None
        self._viewer_thread = None
        self.hand = LeftHandController(self.sim)
        self.right_kinematics = Kinematics(
            self.sim, self.sim.right, "right_tool_tip_site"
        )
        self._left_kinematics = None
        self._tcp_site_id = self.sim.require_site("right_tool_tip_site")
        self._left_palm_site_id = None
        self._left_target = np.zeros(7)
        self._right_target = RIGHT_HOME_RAD.copy()
        self._left_joint_limits = self.sim.model.jnt_range[
            self.sim.left.joint_ids
        ].copy()
        self._joint_limits = self.sim.model.jnt_range[
            self.sim.right.joint_ids
        ].copy()
        self.reset()

        if viewer:
            self.open_viewer()

    def open_viewer(self) -> None:
        if self._viewer is not None:
            return
        from mujoco import viewer as mujoco_viewer

        threads_before = set(threading.enumerate())
        self._viewer = mujoco_viewer.launch_passive(
            self.sim.model, self.sim.data
        )
        launch_target = getattr(mujoco_viewer, "_launch_internal", None)
        launched_threads = set(threading.enumerate()) - threads_before
        matching_threads = [
            thread
            for thread in launched_threads
            if launch_target is not None
            and getattr(thread, "_target", None) is launch_target
        ]
        if len(matching_threads) == 1:
            self._viewer_thread = matching_threads[0]

    def reset(self, joints_rad: Sequence[float] = RIGHT_HOME_RAD) -> None:
        joints = self._validated_target(joints_rad)
        mujoco.mj_resetData(self.sim.model, self.sim.data)
        self._left_target = self.sim.data.qpos[self.sim.left.qpos_ids].copy()
        self.sim.data.qpos[self.sim.right.qpos_ids] = joints
        self.sim.data.qvel[self.sim.right.dof_ids] = 0.0
        self.sim.data.qpos[self.sim.hand.qpos_ids] = DEFAULT_OPEN_RAD
        self.sim.data.qvel[self.sim.hand.dof_ids] = 0.0
        self._right_target = joints.copy()
        self.sim.data.ctrl[self.sim.left.actuator_ids] = self._left_target
        self.sim.data.ctrl[self.sim.right.actuator_ids] = self._right_target
        self.hand.open()
        self.hand.apply()
        mujoco.mj_forward(self.sim.model, self.sim.data)
        self._require_finite_state()
        self._sync_viewer()

    def command(self, joints_rad: Sequence[float]) -> None:
        self._right_target = self._validated_target(joints_rad)

    def command_left(self, joints_rad: Sequence[float]) -> None:
        self._left_target = self._validated_arm_target(
            joints_rad, self._left_joint_limits, "left"
        )

    def validate_targets(self, targets: Sequence[Sequence[float]]) -> None:
        for index, target in enumerate(targets):
            try:
                self._validated_target(target)
            except ValueError as error:
                raise ValueError(f"invalid target at sample {index}: {error}") from error

    def step(
        self,
        control_dt_s: float,
        *,
        sync_viewer: bool = True,
    ) -> None:
        substeps = self._substeps(control_dt_s)
        self._require_finite_state()
        self.sim.data.ctrl[self.sim.left.actuator_ids] = self._left_target
        self.sim.data.ctrl[self.sim.right.actuator_ids] = self._right_target
        self.hand.apply()
        for _ in range(substeps):
            mujoco.mj_step(self.sim.model, self.sim.data)
        self._require_finite_state()
        if sync_viewer:
            self._sync_viewer()
        if self._viewer is not None:
            time.sleep(float(control_dt_s))

    @property
    def joint_positions(self) -> np.ndarray:
        return self.sim.data.qpos[self.sim.right.qpos_ids].copy()

    @property
    def joint_velocities(self) -> np.ndarray:
        return self.sim.data.qvel[self.sim.right.dof_ids].copy()

    @property
    def left_joint_positions(self) -> np.ndarray:
        return self.sim.data.qpos[self.sim.left.qpos_ids].copy()

    @property
    def left_joint_velocities(self) -> np.ndarray:
        return self.sim.data.qvel[self.sim.left.dof_ids].copy()

    @property
    def left_kinematics(self) -> Kinematics:
        if self._left_kinematics is None:
            self._left_kinematics = Kinematics(
                self.sim, self.sim.left, "left_palm_tcp_site"
            )
        return self._left_kinematics

    def tcp_pose(self) -> np.ndarray:
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(
            quaternion, self.sim.data.site_xmat[self._tcp_site_id]
        )
        return np.concatenate(
            (self.sim.data.site_xpos[self._tcp_site_id], quaternion)
        )

    def left_palm_pose(self) -> np.ndarray:
        if self._left_palm_site_id is None:
            self._left_palm_site_id = self.sim.require_site(
                "left_palm_tcp_site"
            )
        return self._site_pose_matrix(self._left_palm_site_id)

    def _site_pose_matrix(self, site_id: int) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, :3] = self.sim.data.site_xmat[site_id].reshape(3, 3)
        pose[:3, 3] = self.sim.data.site_xpos[site_id]
        return pose

    def close(self) -> None:
        viewer, self._viewer = self._viewer, None
        viewer_thread, self._viewer_thread = self._viewer_thread, None
        try:
            if viewer is not None:
                viewer.close()
        finally:
            if viewer_thread is not None:
                viewer_thread.join(timeout=_VIEWER_CLOSE_TIMEOUT_S)

    @staticmethod
    def _validated_joints(joints_rad: Sequence[float]) -> np.ndarray:
        try:
            joints = np.asarray(joints_rad, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("joints_rad must contain 7 finite values") from error
        if joints.shape != (7,) or not np.isfinite(joints).all():
            raise ValueError("joints_rad must contain 7 finite values")
        return joints.copy()

    def _substeps(self, control_dt_s: float) -> int:
        try:
            duration = float(control_dt_s)
        except (TypeError, ValueError) as error:
            raise ValueError("control_dt_s must be a positive integer multiple of timestep") from error
        timestep = self.sim.model.opt.timestep
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("control_dt_s must be a positive integer multiple of timestep")
        ratio = duration / timestep
        substeps = round(ratio)
        if substeps < 1 or ratio != substeps:
            raise ValueError("control_dt_s must be a positive integer multiple of timestep")
        return substeps

    def _validated_target(self, joints_rad: Sequence[float]) -> np.ndarray:
        return self._validated_arm_target(
            joints_rad, self._joint_limits, "right"
        )

    def _validated_arm_target(
        self,
        joints_rad: Sequence[float],
        joint_limits: np.ndarray,
        label: str,
    ) -> np.ndarray:
        try:
            joints = np.asarray(joints_rad, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{label} joints_rad must contain 7 finite values"
            ) from error
        if joints.shape != (7,) or not np.isfinite(joints).all():
            raise ValueError(f"{label} joints_rad must contain 7 finite values")
        outside = np.flatnonzero(
            (joints < joint_limits[:, 0])
            | (joints > joint_limits[:, 1])
        )
        if outside.size:
            index = int(outside[0])
            lower, upper = joint_limits[index]
            raise ValueError(
                f"{label} joint {index + 1} target {joints[index]:.6g} is outside "
                f"range [{lower:.6g}, {upper:.6g}]"
            )
        return joints.copy()

    def _require_finite_state(self) -> None:
        if not all(
            np.isfinite(values).all()
            for values in (self.sim.data.qpos, self.sim.data.qvel, self.sim.data.ctrl)
        ):
            raise NumericalSafetyError("MuJoCo state contains non-finite values")

    def _sync_viewer(self) -> None:
        if self._viewer is not None:
            self._viewer.sync()
