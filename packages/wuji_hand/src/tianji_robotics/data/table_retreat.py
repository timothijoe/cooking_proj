"""Pickle-free storage for corrected tabletop Wuji gestures."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from tianji_robotics.workflows.wuji_table_retreat import CorrectedHandTrajectory, CorrectionReport


def save_corrected_trajectory_npz(value: CorrectedHandTrajectory, path: Path) -> Path:
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,timestamps_ns=value.timestamps_ns,positions_rad=value.positions_rad,palm_positions_m=value.palm_positions_m,palm_quaternions_wxyz=value.palm_quaternions_wxyz,phases=np.asarray(value.phases),source_frame=np.asarray(value.source_frame,dtype=np.int64))
    return path


def load_corrected_trajectory_npz(path: Path) -> CorrectedHandTrajectory:
    path=Path(path)
    try:
        with np.load(path,allow_pickle=False) as archive:
            value=CorrectedHandTrajectory(archive["timestamps_ns"],archive["positions_rad"],archive["palm_positions_m"],archive["palm_quaternions_wxyz"],tuple(archive["phases"].tolist()),int(archive["source_frame"].item()))
    except (KeyError,TypeError,ValueError) as exc:
        raise ValueError(f"invalid corrected trajectory NPZ: {path}") from exc
    count=len(value.timestamps_ns)
    if value.positions_rad.shape!=(count,20) or value.palm_positions_m.shape!=(count,3) or value.palm_quaternions_wxyz.shape!=(count,4) or len(value.phases)!=count or not np.isfinite(value.positions_rad).all() or not np.isfinite(value.palm_positions_m).all() or not np.isfinite(value.palm_quaternions_wxyz).all():
        raise ValueError(f"invalid corrected trajectory NPZ: {path}")
    return value


def write_correction_report_json(report: CorrectionReport, path: Path) -> Path:
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(asdict(report),separators=(",",":"),ensure_ascii=False)+"\n")
    return path
