"""Safety preflight, planning, and explicit opt-in execution for index MCP tests.

The module only operates on an injected hand interface; it never connects to a
device itself. Planning is read-only, while execution requires explicit opt-in.
"""

from dataclasses import dataclass
import time

import numpy as np


RIGHT_HANDEDNESS, INDEX_FINGER, MCP_JOINT = 0, 1, 0
MAX_TEMPERATURE_C, AMPLITUDE_RAD, LIMIT_MARGIN_RAD = 40.0, 0.05, 0.01
_JOINT_MATRIX_SHAPE = (5, 4)


@dataclass(frozen=True)
class IndexMcpTestPlan:
    current_rad: float
    targets_rad: tuple[float, float, float]
    max_temperature_c: float


def build_index_mcp_test_plan(hand) -> IndexMcpTestPlan:
    """Validate a hand state and plan a small index-MCP return motion."""
    if int(hand.read_handedness()) != RIGHT_HANDEDNESS:
        raise ValueError("connected device is not a right hand")

    errors = np.asarray(hand.read_joint_error_code())
    if errors.shape != _JOINT_MATRIX_SHAPE or np.any(errors != 0):
        raise ValueError("joint error codes must all be zero")

    temperatures = np.asarray(hand.read_joint_temperature(), dtype=float)
    if temperatures.shape != _JOINT_MATRIX_SHAPE or not np.isfinite(temperatures).all():
        raise ValueError("joint temperature matrix is invalid")
    maximum = float(np.max(temperatures))
    if maximum > MAX_TEMPERATURE_C:
        raise ValueError(f"joint temperature {maximum:.1f}°C exceeds 40.0°C")

    positions = np.asarray(hand.read_joint_actual_position(), dtype=float)
    lower = np.asarray(hand.read_joint_lower_limit(), dtype=float)
    upper = np.asarray(hand.read_joint_upper_limit(), dtype=float)
    if any(
        values.shape != _JOINT_MATRIX_SHAPE or not np.isfinite(values).all()
        for values in (positions, lower, upper)
    ):
        raise ValueError("index MCP state or limits are invalid")

    current = float(positions[INDEX_FINGER, MCP_JOINT])
    low = float(lower[INDEX_FINGER, MCP_JOINT])
    high = float(upper[INDEX_FINGER, MCP_JOINT])
    targets = (
        current - AMPLITUDE_RAD,
        current + AMPLITUDE_RAD,
        current,
    )
    if low >= high or any(
        target < low + LIMIT_MARGIN_RAD or target > high - LIMIT_MARGIN_RAD
        for target in targets
    ):
        raise ValueError("index MCP target lacks hardware-limit margin")

    return IndexMcpTestPlan(current, targets, maximum)


def run_index_mcp_test(hand, plan, *, execute: bool, dwell_s: float, sleep=time.sleep) -> bool:
    """Execute an explicitly approved index-MCP plan on the injected hand."""
    if not execute:
        return False
    if not np.isfinite(dwell_s) or dwell_s < 0:
        raise ValueError("dwell_s must be finite and non-negative")

    enabled = False
    try:
        hand.write_joint_enabled(True)
        enabled = True
        joint = hand.finger(INDEX_FINGER).joint(MCP_JOINT)
        for target in plan.targets_rad:
            joint.write_joint_target_position(target)
            sleep(dwell_s)
        return True
    finally:
        if enabled:
            hand.write_joint_enabled(False)
