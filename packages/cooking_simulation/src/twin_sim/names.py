from dataclasses import dataclass


@dataclass(frozen=True)
class ArmNames:
    joints: tuple[str, ...]
    actuators: tuple[str, ...]


LEFT_ARM = ArmNames(
    joints=tuple(f"left_joint{i}" for i in range(1, 8)),
    actuators=tuple(f"act_left_joint{i}" for i in range(1, 8)),
)

RIGHT_ARM = ArmNames(
    joints=tuple(f"right_joint{i}" for i in range(1, 8)),
    actuators=tuple(f"act_right_joint{i}" for i in range(1, 8)),
)
