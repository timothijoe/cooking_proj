"""A small wujihandpy-shaped API backed only by the MuJoCo model."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

import numpy as np

from twin_sim.robot import RightArmRobot


_SDK_SHAPE = (5, 4)


class SimRealtimeController(AbstractContextManager["SimRealtimeController"]):
    def __init__(self, hand: "SimWujiHand"):
        self._hand = hand
        self._closed = False

    def __enter__(self) -> "SimRealtimeController":
        self._require_open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._closed = True

    def set_joint_target_position(self, target: Any) -> None:
        self._require_open()
        self._hand.write_joint_target_position(target)

    def get_joint_actual_position(self) -> np.ndarray:
        self._require_open()
        return self._hand.read_joint_actual_position()

    def get_joint_actual_effort(self) -> np.ndarray:
        """Return uncalibrated MuJoCo actuator force, not physical torque."""
        self._require_open()
        self._hand._require_open()
        forces = self._hand.robot.sim.data.actuator_force[
            self._hand.robot.sim.hand.actuator_ids
        ]
        return np.asarray(forces, dtype=float).reshape(_SDK_SHAPE).copy()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("realtime controller is closed")


class SimWujiHand:
    """Simulation-only subset of :class:`wujihandpy.Hand`.

    Arrays follow the official finger-major layout: five fingers by four
    joints. This class never discovers USB devices or writes physical motors.
    """

    def __init__(
        self, robot: RightArmRobot | None = None, *, viewer: bool = False
    ):
        self.robot = robot if robot is not None else RightArmRobot(viewer=viewer)
        self._owns_robot = robot is None
        self._enabled = np.ones(_SDK_SHAPE, dtype=bool)
        self._closed = False

    def read_joint_actual_position(self) -> np.ndarray:
        self._require_open()
        values = self.robot.sim.data.qpos[self.robot.sim.hand.qpos_ids]
        return np.asarray(values, dtype=float).reshape(_SDK_SHAPE).copy()

    def read_joint_target_position(self) -> np.ndarray:
        self._require_open()
        return self.robot.hand.target.reshape(_SDK_SHAPE).copy()

    def write_joint_target_position(self, target: Any) -> None:
        self._require_open()
        values = self._validated_matrix(target, "joint target")
        current = self.read_joint_target_position()
        if np.any((values != current) & ~self._enabled):
            raise RuntimeError("command changes one or more disabled simulated joints")
        self.robot.hand.command(values.reshape(20))

    def write_joint_enabled(self, enabled: Any) -> None:
        self._require_open()
        values = np.asarray(enabled)
        if values.shape == ():
            if not np.issubdtype(values.dtype, np.bool_):
                raise ValueError("joint enabled must contain boolean values")
            self._enabled.fill(bool(values))
            return
        if values.shape != _SDK_SHAPE:
            raise ValueError("joint enabled must be a bool or have shape (5, 4)")
        if not np.issubdtype(values.dtype, np.bool_):
            raise ValueError("joint enabled must contain boolean values")
        self._enabled = values.copy()

    def read_joint_enabled(self) -> np.ndarray:
        self._require_open()
        return self._enabled.copy()

    def step(self, control_dt_s: float) -> None:
        self._require_open()
        self.robot.step(control_dt_s)

    def realtime_controller(self, *args: object, **kwargs: object) -> SimRealtimeController:
        self._require_open()
        if args or kwargs:
            raise NotImplementedError(
                "realtime controller options are not simulated"
            )
        return SimRealtimeController(self)

    def read_joint_temperature(self) -> np.ndarray:
        raise NotImplementedError("joint temperature is not simulated")

    def read_input_voltage(self) -> float:
        raise NotImplementedError("input voltage is not simulated")

    def close(self) -> None:
        if not self._closed and self._owns_robot:
            self.robot.close()
        self._closed = True

    @staticmethod
    def _validated_matrix(values: Any, label: str) -> np.ndarray:
        try:
            matrix = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} must contain finite values with shape (5, 4)") from error
        if matrix.shape != _SDK_SHAPE or not np.isfinite(matrix).all():
            raise ValueError(f"{label} must contain finite values with shape (5, 4)")
        return matrix.copy()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("simulated hand is closed")
