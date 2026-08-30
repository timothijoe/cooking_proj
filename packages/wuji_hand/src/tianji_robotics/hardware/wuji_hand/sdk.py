"""Injected lifecycle boundary for a future official Wuji SDK runtime."""

from typing import Any

import numpy as np


class SdkWujiHand:
    """Enforce explicit connection and arming without importing vendor code."""

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime
        self._connected = False
        self._armed = False
        self._closed = False

    def connect(self) -> None:
        if self._closed:
            raise RuntimeError("hand is closed")
        if not self._connected:
            self._runtime.connect()
            self._connected = True

    def arm(self) -> None:
        if not self._connected:
            raise RuntimeError("hand is not connected")
        if not self._armed:
            self._runtime.arm()
            self._armed = True

    def command_position_rad(self, target: np.ndarray) -> None:
        if not self._connected:
            raise RuntimeError("hand is not connected")
        if not self._armed:
            raise RuntimeError("hand is not armed")
        values = np.asarray(target, dtype=float)
        if values.shape != (20,) or not np.isfinite(values).all():
            raise ValueError("Wuji hand target must contain 20 finite radians")
        try:
            self._runtime.command_position_rad(values.copy())
        except BaseException:
            self.disarm()
            raise

    def disarm(self) -> None:
        if self._armed:
            self._runtime.disarm()
            self._armed = False

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.disarm()
        finally:
            if self._connected:
                self._runtime.close()
            self._connected = False
            self._closed = True
