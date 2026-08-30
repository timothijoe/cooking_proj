"""Resolve large machine-local resources without committing them to Git."""

from __future__ import annotations

import os
from pathlib import Path


def repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def local_root() -> Path:
    override = os.environ.get("COOKING_LOCAL_ROOT")
    return Path(override).expanduser().resolve() if override else repository_root() / "local"


def asset_root() -> Path:
    return local_root() / "assets"


def vendor_root() -> Path:
    return local_root() / "vendor"


def data_root() -> Path:
    return local_root() / "data"


def require_local(relative_path: str | Path, *, kind: str = "local resource") -> Path:
    path = local_root() / relative_path
    if not path.exists():
        raise FileNotFoundError(
            f"missing {kind}: {path}; restore it under local/ or set COOKING_LOCAL_ROOT"
        )
    return path
