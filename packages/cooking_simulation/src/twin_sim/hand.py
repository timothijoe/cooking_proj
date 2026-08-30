from typing import Sequence

import numpy as np

from twin_sim.model import SimulationModel


DEFAULT_OPEN_RAD = np.asarray(
    (0.15, 0.0, 0.05, 0.05)
    + (0.05, 0.0, 0.05, 0.05) * 4,
    dtype=float,
)


class LeftHandController:
    def __init__(self, sim: SimulationModel):
        self.sim = sim
        self._control_ranges = sim.model.actuator_ctrlrange[
            sim.hand.actuator_ids
        ].copy()
        self._target = self._validated_target(DEFAULT_OPEN_RAD)

    @property
    def target(self) -> np.ndarray:
        return self._target.copy()

    def command(self, joints_rad: Sequence[float]) -> None:
        self._target = self._validated_target(joints_rad)

    def apply(self) -> None:
        self.sim.data.ctrl[self.sim.hand.actuator_ids] = self._target

    def open(self) -> None:
        self._target = DEFAULT_OPEN_RAD.copy()

    def _validated_target(self, joints_rad: Sequence[float]) -> np.ndarray:
        try:
            joints = np.asarray(joints_rad, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("hand target must contain 20 finite values") from error
        if joints.shape != (20,) or not np.isfinite(joints).all():
            raise ValueError("hand target must contain 20 finite values")
        outside = np.flatnonzero(
            (joints < self._control_ranges[:, 0])
            | (joints > self._control_ranges[:, 1])
        )
        if outside.size:
            index = int(outside[0])
            lower, upper = self._control_ranges[index]
            raise ValueError(
                f"joint {index + 1} target {joints[index]:.6g} is outside "
                f"range [{lower:.6g}, {upper:.6g}]"
            )
        return joints.copy()
