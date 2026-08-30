"""Backend-neutral Wuji hand capability contracts."""

from collections.abc import Iterator, Mapping
from typing import Protocol, runtime_checkable

import numpy as np

from .models import SkeletonFrame


@runtime_checkable
class SkeletonSource(Protocol):
    def frames(self) -> Iterator[SkeletonFrame]: ...


@runtime_checkable
class Retargeter(Protocol):
    def step(self, keypoints_m: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class WujiHandBackend(Protocol):
    @property
    def joint_ranges_rad(self) -> Mapping[str, tuple[float, float]]: ...

    @property
    def range_tolerance_rad(self) -> float: ...

    def read_position_rad(self) -> np.ndarray: ...

    def command_position_rad(self, target: np.ndarray) -> None: ...

    def step(self, duration_s: float) -> None: ...

    def close(self) -> None: ...
