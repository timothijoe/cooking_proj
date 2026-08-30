#!/usr/bin/env python3
"""Real robot planned Cartesian position-mode chopping entrypoint.

This is the position-mode wrapper for the left-arm chopping debug script. It
forces the implementation to use the SDK MOVLA + setPln_Cart path, matching
``test/showcase_pln_cart_positionMode.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from robot_core.paths import local_root
ROOT = local_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tianji_arm.experiments import real_ik_cart_impedance_lateral as impl


def main(argv: list[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    return impl.main([*args, "--command-mode", "pln-cart"])


if __name__ == "__main__":
    raise SystemExit(main())
