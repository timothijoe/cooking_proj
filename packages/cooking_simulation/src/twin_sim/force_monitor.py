from dataclasses import dataclass
import warnings

import numpy as np

from twin_sim.model import SimulationModel


@dataclass(frozen=True)
class ForceSample:
    raw_force_n: float
    filtered_force_n: float
    over_threshold: bool


class ForceMonitor:
    def __init__(
        self,
        simulation: SimulationModel | None = None,
        *,
        alpha: float,
        warning_threshold_n: float,
    ):
        if not np.isfinite(alpha) or not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be finite and in (0, 1]")
        if not np.isfinite(warning_threshold_n) or warning_threshold_n < 0.0:
            raise ValueError("warning_threshold_n must be non-negative and finite")
        self._simulation = simulation
        self._alpha = float(alpha)
        self._threshold = float(warning_threshold_n)
        self._filtered = 0.0
        self._was_over_threshold = False
        self._sensor_address: int | None = None
        self._sensor_dimension: int | None = None
        if simulation is not None:
            sensor_id = simulation.require_sensor("right_tool_force")
            self._sensor_address = int(simulation.model.sensor_adr[sensor_id])
            self._sensor_dimension = int(simulation.model.sensor_dim[sensor_id])
            if self._sensor_dimension != 3:
                raise ValueError("right_tool_force sensor must have dimension 3")

    def sample(self) -> ForceSample:
        if (
            self._simulation is None
            or self._sensor_address is None
            or self._sensor_dimension is None
        ):
            raise RuntimeError("sample() requires a SimulationModel")
        vector = self._simulation.data.sensordata[
            self._sensor_address : self._sensor_address + self._sensor_dimension
        ]
        return self.update(float(np.linalg.norm(vector)))

    def update(self, raw_force_n: float) -> ForceSample:
        raw = float(raw_force_n)
        if not np.isfinite(raw) or raw < 0.0:
            raise ValueError("raw_force_n must be non-negative and finite")
        self._filtered = self._alpha * raw + (1.0 - self._alpha) * self._filtered
        over_threshold = self._filtered > self._threshold
        if over_threshold and not self._was_over_threshold:
            warnings.warn(
                (
                    f"contact force {self._filtered:.3f} N exceeds warning "
                    f"threshold {self._threshold:.3f} N"
                ),
                RuntimeWarning,
                stacklevel=2,
            )
        self._was_over_threshold = over_threshold
        return ForceSample(raw, self._filtered, over_threshold)

