from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Sequence

import mujoco
import numpy as np

from twin_sim.model import ArmIndices, SimulationModel


@dataclass(frozen=True)
class IkResult:
    joints_rad: np.ndarray
    success: bool
    iterations: int
    residual: float


class PathIkError(RuntimeError):
    pass


class Kinematics:
    def __init__(
        self,
        simulation: SimulationModel,
        arm: ArmIndices | None = None,
        tcp_site: str = "right_tool_tip_site",
    ):
        self._model = simulation.model
        self._data = simulation.data
        self._arm = simulation.right if arm is None else arm
        self._tcp_site_id = simulation.require_site(tcp_site)
        self._lower_limits = self._model.jnt_range[
            self._arm.joint_ids, 0
        ].copy()
        self._upper_limits = self._model.jnt_range[
            self._arm.joint_ids, 1
        ].copy()

    def fk(self, joints_rad: Sequence[float]) -> np.ndarray:
        joints = self._validated_joints(joints_rad)
        with self._configuration(joints):
            pose = np.eye(4)
            pose[:3, :3] = self._data.site_xmat[self._tcp_site_id].reshape(3, 3)
            pose[:3, 3] = self._data.site_xpos[self._tcp_site_id]
            return pose

    def jacobian(self, joints_rad: Sequence[float]) -> np.ndarray:
        joints = self._validated_joints(joints_rad)
        with self._configuration(joints):
            jacobian_position = np.zeros((3, self._model.nv))
            jacobian_rotation = np.zeros((3, self._model.nv))
            mujoco.mj_jacSite(
                self._model,
                self._data,
                jacobian_position,
                jacobian_rotation,
                self._tcp_site_id,
            )
            return np.vstack(
                (
                    jacobian_position[:, self._arm.dof_ids],
                    jacobian_rotation[:, self._arm.dof_ids],
                )
            )

    def ik(
        self,
        target: np.ndarray,
        reference: Sequence[float],
        *,
        max_iterations: int = 200,
        tolerance: float = 1e-5,
        damping: float = 1e-3,
    ) -> IkResult:
        target_pose = self._validated_pose(target)
        joints = np.clip(
            self._validated_joints(reference), self._lower_limits, self._upper_limits
        )
        if (
            isinstance(max_iterations, bool)
            or not isinstance(max_iterations, (int, np.integer))
            or max_iterations < 0
        ):
            raise ValueError("max_iterations must be a non-negative integer")
        max_iterations = int(max_iterations)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be positive and finite")
        if not np.isfinite(damping) or damping <= 0.0:
            raise ValueError("damping must be positive and finite")

        for iteration in range(max_iterations + 1):
            current_pose = self.fk(joints)
            error = self._pose_error(target_pose, current_pose)
            residual = float(np.linalg.norm(error))
            if residual <= tolerance:
                return IkResult(joints.copy(), True, iteration, residual)
            if iteration == max_iterations:
                break

            jacobian = self.jacobian(joints)
            damped = jacobian @ jacobian.T + damping**2 * np.eye(6)
            joint_update = jacobian.T @ np.linalg.solve(damped, error)
            joints = np.clip(
                joints + joint_update, self._lower_limits, self._upper_limits
            )

        return IkResult(joints.copy(), False, max_iterations, residual)

    def solve_path(
        self,
        poses: Sequence[np.ndarray],
        seed: Sequence[float],
        *,
        max_joint_step_rad: float = 0.15,
    ) -> np.ndarray:
        if not np.isfinite(max_joint_step_rad) or max_joint_step_rad <= 0.0:
            raise ValueError("max_joint_step_rad must be positive and finite")

        previous = self._validated_joints(seed)
        solutions = []
        for index, pose in enumerate(poses):
            try:
                result = self.ik(pose, previous)
            except ValueError as error:
                raise PathIkError(f"invalid pose at sample {index}: {error}") from error
            if not result.success:
                raise PathIkError(
                    f"inverse kinematics failed at sample {index} "
                    f"(residual {result.residual:.6g})"
                )
            joint_step = float(np.max(np.abs(result.joints_rad - previous)))
            if joint_step > max_joint_step_rad:
                raise PathIkError(
                    f"joint-step violation at sample {index}: "
                    f"{joint_step:.6g} rad exceeds {max_joint_step_rad:.6g} rad"
                )
            solutions.append(result.joints_rad.copy())
            previous = result.joints_rad

        if not solutions:
            return np.empty((0, self._arm.joint_ids.size))
        return np.asarray(solutions)

    @contextmanager
    def _configuration(self, joints: np.ndarray) -> Iterator[None]:
        saved_data = mujoco.MjData(self._model)
        mujoco.mj_copyData(saved_data, self._model, self._data)
        try:
            self._data.qpos[self._arm.qpos_ids] = joints
            mujoco.mj_forward(self._model, self._data)
            yield
        finally:
            mujoco.mj_copyData(self._data, self._model, saved_data)

    @staticmethod
    def _validated_joints(joints_rad: Sequence[float]) -> np.ndarray:
        try:
            joints = np.asarray(joints_rad, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("joints_rad must contain 7 finite values") from error
        if joints.shape != (7,) or not np.isfinite(joints).all():
            raise ValueError("joints_rad must contain 7 finite values")
        return joints.copy()

    @staticmethod
    def _validated_pose(pose: np.ndarray) -> np.ndarray:
        try:
            target = np.asarray(pose, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("target must be a finite rigid 4x4 pose") from error
        if target.shape != (4, 4) or not np.isfinite(target).all():
            raise ValueError("target must be a finite rigid 4x4 pose")
        rotation = target[:3, :3]
        if (
            not np.allclose(target[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1e-9)
            or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1.0, rtol=0.0, atol=1e-6)
        ):
            raise ValueError("target must be a finite rigid 4x4 pose")
        return target.copy()

    @staticmethod
    def _pose_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
        target_quaternion = np.empty(4)
        current_quaternion = np.empty(4)
        rotation_error = np.empty(3)
        mujoco.mju_mat2Quat(target_quaternion, target[:3, :3].reshape(9))
        mujoco.mju_mat2Quat(current_quaternion, current[:3, :3].reshape(9))
        mujoco.mju_subQuat(rotation_error, target_quaternion, current_quaternion)
        rotation_error_world = current[:3, :3] @ rotation_error
        return np.concatenate(
            (target[:3, 3] - current[:3, 3], rotation_error_world)
        )
