#!/usr/bin/env python3
"""Run sampled IK chopping with SDK joint impedance tracking."""

from __future__ import annotations

import sys
from pathlib import Path

from robot_core.paths import local_root
ROOT = local_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tianji_arm.experiments import real_ik_cart_impedance_lateral as impl


def main(argv: list[str] | None = None) -> int:
    """Force the shared debug runner into sampled joint impedance mode."""
    args = list(argv) if argv is not None else sys.argv[1:]
    if "--command-mode" not in args:
        args.extend(["--command-mode", "joint-impedance"])
    return impl.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
