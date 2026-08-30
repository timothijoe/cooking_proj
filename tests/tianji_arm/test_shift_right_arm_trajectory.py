from __future__ import annotations

import numpy as np


def test_shift_trajectory_preserves_non_right_arrays_and_shifts_tcp_x_by_100mm():
    from tianji_arm.experiments.shift_right_arm_trajectory import shift_right_arm_trajectory

    class FakeKinematics:
        def fk(self, joints):
            # Encode the first three joints as XYZ in millimetres.
            result = np.eye(4)
            result[:3, 3] = np.asarray(joints[:3], dtype=float)
            return result

        def solve_ik(self, target_matrix, reference_joints):
            target = np.asarray(target_matrix, dtype=float)
            # The result makes both target position and reference observable.
            return np.array([target[0, 3], target[1, 3], target[2, 3], *reference_joints[3:]], dtype=float)

    original_right = np.array([[1.0, 2.0, 3.0, 4, 5, 6, 7], [8.0, 9.0, 10.0, 11, 12, 13, 14]])
    source = {
        "time_s": np.array([0.0, 0.005]),
        "left_arm_target_rad": np.array([[0.1] * 7, [0.2] * 7]),
        "right_arm_target_rad": np.deg2rad(original_right),
        "right_hand_target_rad": np.array([[0.3] * 20, [0.4] * 20]),
        "metadata": np.array("keep-me"),
    }

    transformed, report = shift_right_arm_trajectory(source, FakeKinematics(), shift_mm=100.0)

    np.testing.assert_array_equal(transformed["left_arm_target_rad"], source["left_arm_target_rad"])
    np.testing.assert_array_equal(transformed["right_hand_target_rad"], source["right_hand_target_rad"])
    assert transformed["metadata"] == source["metadata"]
    actual_right_deg = np.rad2deg(transformed["right_arm_target_rad"])
    np.testing.assert_allclose(actual_right_deg[:, 0], original_right[:, 0] + 100.0)
    np.testing.assert_allclose(actual_right_deg[:, 1:3], original_right[:, 1:3])
    # The fake solver carries the previous null-space joints forward; this
    # confirms the conversion uses a continuous reference rather than resets.
    np.testing.assert_allclose(actual_right_deg[:, 3:], np.tile(original_right[0, 3:], (2, 1)))
    assert report.max_tcp_position_error_mm == 0.0


def test_shift_trajectory_stops_when_a_frame_has_no_ik_solution():
    from tianji_arm.experiments.shift_right_arm_trajectory import TrajectoryShiftError, shift_right_arm_trajectory

    class NoSolutionKinematics:
        def fk(self, joints):
            return np.eye(4)

        def solve_ik(self, target_matrix, reference_joints):
            return None

    source = {
        "time_s": np.array([0.0, 0.005]),
        "left_arm_target_rad": np.zeros((2, 7)),
        "right_arm_target_rad": np.zeros((2, 7)),
    }

    with np.testing.assert_raises_regex(TrajectoryShiftError, "frame 0"):
        shift_right_arm_trajectory(source, NoSolutionKinematics(), shift_mm=100.0)


def test_shift_trajectory_applies_forward_x_and_right_z_offsets_together():
    from tianji_arm.experiments.shift_right_arm_trajectory import shift_right_arm_trajectory

    class FakeKinematics:
        def fk(self, joints):
            matrix = np.eye(4)
            matrix[:3, 3] = np.asarray(joints[:3], dtype=float)
            return matrix

        def solve_ik(self, target_matrix, reference_joints):
            target = np.asarray(target_matrix, dtype=float)
            return np.array([*target[:3, 3], *reference_joints[3:]], dtype=float)

    source = {
        "time_s": np.array([0.0, 0.005]),
        "right_arm_target_rad": np.deg2rad([[1.0, 2.0, 3.0, 4, 5, 6, 7], [8.0, 9.0, 10.0, 11, 12, 13, 14]]),
    }
    transformed, report = shift_right_arm_trajectory(
        source, FakeKinematics(), shift_base_mm=np.array([30.0, 0.0, 100.0])
    )

    shifted_deg = np.rad2deg(transformed["right_arm_target_rad"])
    np.testing.assert_allclose(shifted_deg[:, :3], [[31.0, 2.0, 103.0], [38.0, 9.0, 110.0]])
    np.testing.assert_allclose(transformed["right_arm_tcp_shift_base_mm"], [30.0, 0.0, 100.0])
    np.testing.assert_allclose(report.shift_base_mm, [30.0, 0.0, 100.0])
