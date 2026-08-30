from dataclasses import dataclass

import numpy as np


CAT_PAW_RAD = np.asarray(
    (0.95, -0.02, 0.95, 1.05)
    + (0.30, 0.00, 1.10, 1.10) * 4,
    dtype=float,
)


CAT_PAW_OPEN_RAD = np.asarray(
    (0.80, -0.02, 0.55, 0.55)
    + (0.20, 0.00, 0.55, 0.55) * 4,
    dtype=float,
)


GUARD_RELAXED_RAD = CAT_PAW_RAD.copy()
GUARD_RETRACTED_RAD = np.asarray(
    (
        0.95,
        -0.02,
        0.95,
        1.05,
        0.3375,
        -0.225,
        1.3617993878,
        1.1371815399,
        0.35,
        -0.3,
        1.4490658504,
        1.1495753865,
        0.32,
        -0.12,
        1.2396263402,
        1.1198301546,
        0.319,
        -0.114,
        1.2326450232,
        1.1188386469,
    ),
    dtype=float,
)
GUARD_RELAXED_RAD.flags.writeable = False
GUARD_RETRACTED_RAD.flags.writeable = False


@dataclass(frozen=True)
class SafetyObservation:
    phase: str
    knife_height_m: float
    safe_knife_height_m: float
    knife_guard_distance_m: float
    left_target_stationary: bool
    right_target_stationary: bool
    left_speed_rad_s: float
    right_speed_rad_s: float
    hand_speed_rad_s: float
    guard_cube_penetration_m: float
    guard_cube_normal_force_n: float
    finite_state: bool


@dataclass(frozen=True)
class SafetyDecision:
    allowed: bool
    reason: str = ""


class SafetyCoordinator:
    def __init__(
        self,
        minimum_distance_m: float = 0.02,
        maximum_guard_penetration_m: float = 0.003,
        maximum_guard_force_n: float = 35.0,
    ):
        if (
            not np.isfinite(minimum_distance_m)
            or minimum_distance_m <= 0.0
        ):
            raise ValueError(
                "minimum_distance_m must be positive and finite"
            )
        if (
            not np.isfinite(maximum_guard_penetration_m)
            or maximum_guard_penetration_m <= 0.0
            or not np.isfinite(maximum_guard_force_n)
            or maximum_guard_force_n <= 0.0
        ):
            raise ValueError("guard contact limits must be positive and finite")
        self.minimum_distance_m = float(minimum_distance_m)
        self.maximum_guard_penetration_m = float(maximum_guard_penetration_m)
        self.maximum_guard_force_n = float(maximum_guard_force_n)

    def evaluate(self, value: SafetyObservation) -> SafetyDecision:
        scalars = (
            value.knife_height_m,
            value.safe_knife_height_m,
            value.knife_guard_distance_m,
            value.left_speed_rad_s,
            value.right_speed_rad_s,
            value.hand_speed_rad_s,
            value.guard_cube_penetration_m,
            value.guard_cube_normal_force_n,
        )
        if not value.finite_state or not np.isfinite(scalars).all():
            return SafetyDecision(False, "non-finite simulation state")
        if value.knife_guard_distance_m < self.minimum_distance_m:
            return SafetyDecision(
                False, "knife-guard distance below limit"
            )
        if (
            value.guard_cube_penetration_m
            > self.maximum_guard_penetration_m
            or value.guard_cube_normal_force_n > self.maximum_guard_force_n
        ):
            return SafetyDecision(False, "left guard contact exceeds limit")

        phase = str(getattr(value.phase, "value", value.phase)).upper()
        if phase == "CUT_DOWN" and (
            not value.left_target_stationary
            or value.left_speed_rad_s > 0.05
            or value.hand_speed_rad_s > 0.05
        ):
            return SafetyDecision(False, "left guard moved during cut")
        if phase == "KNIFE_CLEAR" and (
            not value.left_target_stationary
            or value.left_speed_rad_s > 0.05
            or value.hand_speed_rad_s > 0.05
        ):
            return SafetyDecision(
                False, "guard moved before knife clearance"
            )
        if phase in {
            "LOW_GUARD_OPEN",
            "LOW_GUARD_SHIFT",
            "LOW_GUARD_CLOSE",
            "GUARD_SETTLE",
            "FINGER_RETRACT",
            "ARM_RESET_RELAX",
        } and (
            not value.right_target_stationary
            or value.right_speed_rad_s > 0.05
        ):
            return SafetyDecision(
                False, "knife moved during low guard shift"
            )
        if phase == "KNIFE_LIFT_SHIFT" and (
            not value.left_target_stationary
            or value.left_speed_rad_s > 0.05
            or value.hand_speed_rad_s > 0.05
        ):
            return SafetyDecision(
                False, "guard moved during knife lift shift"
            )
        return SafetyDecision(True)
