HAND_JOINTS = tuple(
    f"left_finger{finger}_joint{joint}"
    for finger in range(1, 6)
    for joint in range(1, 5)
)

HAND_ACTUATORS = tuple(f"{joint}_actuator" for joint in HAND_JOINTS)
