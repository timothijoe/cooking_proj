"""Coordinate transforms for recorded Wuji hand skeletons."""

import numpy as np
from numpy.typing import NDArray


def mirror_right_to_left(keypoints_m: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return a copied skeleton reflected across the wrist-frame Y axis."""
    mirrored = np.array(keypoints_m, copy=True)
    if mirrored.shape != (21, 3):
        raise ValueError("keypoints_m must have shape (21, 3)")
    if not np.issubdtype(mirrored.dtype, np.number) or not np.isfinite(mirrored).all():
        raise ValueError("keypoints_m must be finite")
    mirrored[:, 1] *= -1
    return mirrored
