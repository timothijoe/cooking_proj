from dataclasses import dataclass
from typing import Sequence

import mujoco
import numpy as np

from twin_sim.kinematics import Kinematics

DEFAULT_MAX_VELOCITY_RAD_S = 2.0
DEFAULT_MAX_ACCELERATION_RAD_S2 = 10.0

@dataclass(frozen=True)
class TrajectoryPoint:
    time_s: float
    joints_rad: np.ndarray
    velocity_rad_s: np.ndarray
    target_pose: np.ndarray | None


def minimum_jerk(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    phase = np.asarray(s, dtype=float)
    if not np.isfinite(phase).all() or np.any((phase < 0.0) | (phase > 1.0)):
        raise ValueError("s must contain finite values in [0, 1]")
    position = 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
    derivative = 30.0 * phase**2 - 60.0 * phase**3 + 30.0 * phase**4
    return position, derivative


def joint_trajectory(
    start: Sequence[float],
    goal: Sequence[float],
    duration_s: float,
    control_dt_s: float,
) -> list[TrajectoryPoint]:
    start_joints = _joints(start, "start")
    goal_joints = _joints(goal, "goal")
    times = _sample_times(duration_s, control_dt_s)
    blend, blend_derivative = minimum_jerk(times / duration_s)
    delta = goal_joints - start_joints
    joints = start_joints + blend[:, None] * delta
    velocities = blend_derivative[:, None] * delta / duration_s
    joints[0] = start_joints
    joints[-1] = goal_joints
    velocities[[0, -1]] = 0.0
    _validate_motion(joints, control_dt_s)
    return [
        TrajectoryPoint(float(time_s), q.copy(), qd.copy(), None)
        for time_s, q, qd in zip(times, joints, velocities, strict=True)
    ]


def cartesian_trajectory(
    kinematics: Kinematics,
    start_pose: np.ndarray,
    goal_pose: np.ndarray,
    seed: Sequence[float],
    duration_s: float,
    control_dt_s: float,
) -> list[TrajectoryPoint]:
    times = _sample_times(duration_s, control_dt_s)
    start = _pose(start_pose, "start_pose")
    goal = _pose(goal_pose, "goal_pose")
    blend, _ = minimum_jerk(times / duration_s)
    start_quaternion = _matrix_to_quaternion(start[:3, :3])
    goal_quaternion = _matrix_to_quaternion(goal[:3, :3])
    poses: list[np.ndarray] = []
    for index, value in enumerate(blend):
        pose = np.eye(4)
        pose[:3, 3] = start[:3, 3] + value * (goal[:3, 3] - start[:3, 3])
        pose[:3, :3] = _quaternion_to_matrix(
            _slerp(start_quaternion, goal_quaternion, float(value))
        )
        poses.append(pose)
    poses[0] = start.copy()
    poses[-1] = goal.copy()

    joints = kinematics.solve_path(poses, seed)
    if len(joints) == 2:
        velocities = np.zeros_like(joints)
    else:
        velocities = np.gradient(joints, control_dt_s, axis=0, edge_order=2)
    velocities[[0, -1]] = 0.0
    _validate_motion(joints, control_dt_s)
    return [
        TrajectoryPoint(float(time_s), q.copy(), qd.copy(), pose.copy())
        for time_s, q, qd, pose in zip(
            times, joints, velocities, poses, strict=True
        )
    ]


def _sample_times(duration_s: float, control_dt_s: float) -> np.ndarray:
    if (
        not np.isfinite(duration_s)
        or not np.isfinite(control_dt_s)
        or duration_s <= 0.0
        or control_dt_s <= 0.0
    ):
        raise ValueError("duration_s and control_dt_s must be positive and finite")
    ratio = duration_s / control_dt_s
    steps = round(ratio)
    if ratio != steps:
        raise ValueError("duration_s must be an integer multiple of control_dt_s")
    return np.linspace(0.0, duration_s, steps + 1)


def _validate_motion(joints: np.ndarray, control_dt_s: float) -> None:
    segment_velocity = np.diff(joints, axis=0) / control_dt_s
    if segment_velocity.size and np.max(np.abs(segment_velocity)) > DEFAULT_MAX_VELOCITY_RAD_S:
        raise ValueError(
            f"trajectory exceeds velocity limit {DEFAULT_MAX_VELOCITY_RAD_S:g} rad/s"
        )
    endpoint_velocity = np.zeros((1, joints.shape[1]))
    velocity_with_endpoints = np.vstack(
        (endpoint_velocity, segment_velocity, endpoint_velocity)
    )
    segment_acceleration = np.diff(velocity_with_endpoints, axis=0) / control_dt_s
    if (
        segment_acceleration.size
        and np.max(np.abs(segment_acceleration)) > DEFAULT_MAX_ACCELERATION_RAD_S2
    ):
        raise ValueError(
            "trajectory exceeds acceleration limit "
            f"{DEFAULT_MAX_ACCELERATION_RAD_S2:g} rad/s^2"
        )


def _joints(values: Sequence[float], name: str) -> np.ndarray:
    try:
        joints = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain 7 finite values") from error
    if joints.shape != (7,) or not np.isfinite(joints).all():
        raise ValueError(f"{name} must contain 7 finite values")
    return joints.copy()


def _pose(value: np.ndarray, name: str) -> np.ndarray:
    try:
        pose = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite rigid 4x4 pose") from error
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"{name} must be a finite rigid 4x4 pose")
    rotation = pose[:3, :3]
    if (
        not np.allclose(pose[3], (0, 0, 0, 1), rtol=0.0, atol=1e-9)
        or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-6)
        or not np.isclose(np.linalg.det(rotation), 1.0, rtol=0.0, atol=1e-6)
    ):
        raise ValueError(f"{name} must be a finite rigid 4x4 pose")
    return pose.copy()


def _matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, rotation.reshape(9))
    return quaternion


def _quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quaternion)
    return rotation.reshape(3, 3)


def _slerp(start: np.ndarray, goal: np.ndarray, fraction: float) -> np.ndarray:
    end = goal.copy()
    dot = float(np.dot(start, end))
    if dot < 0.0:
        end = -end
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = start + fraction * (end - start)
        return result / np.linalg.norm(result)
    angle = np.arccos(dot)
    scale = np.sin(angle)
    return (
        np.sin((1.0 - fraction) * angle) / scale * start
        + np.sin(fraction * angle) / scale * end
    )
