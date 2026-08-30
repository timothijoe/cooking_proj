"""Canonical joint names for a first-generation left Wuji hand."""

HAND_JOINT_NAMES = tuple(
    f"left_finger{finger}_joint{joint}"
    for finger in range(1, 6)
    for joint in range(1, 5)
)
