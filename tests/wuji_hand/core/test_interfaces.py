import inspect

import numpy as np

from tianji_robotics.wuji_hand.interfaces import Retargeter, WujiHandBackend


def test_backend_protocols_expose_plain_ndarray_contracts():
    assert inspect.signature(Retargeter.step).parameters["keypoints_m"].annotation is np.ndarray
    assert inspect.signature(Retargeter.step).return_annotation is np.ndarray
    assert inspect.signature(WujiHandBackend.read_position_rad).return_annotation is np.ndarray
    assert inspect.signature(WujiHandBackend.command_position_rad).parameters["target"].annotation is np.ndarray
