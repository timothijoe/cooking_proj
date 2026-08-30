import numpy as np

from tianji_robotics.wuji_hand.transforms import mirror_right_to_left


def test_mirror_flips_only_y_and_copies_input():
    source = np.arange(63, dtype=np.float32).reshape(21, 3)
    result = mirror_right_to_left(source)

    np.testing.assert_array_equal(result[:, 0], source[:, 0])
    np.testing.assert_array_equal(result[:, 1], -source[:, 1])
    np.testing.assert_array_equal(result[:, 2], source[:, 2])
    assert not np.shares_memory(result, source)
