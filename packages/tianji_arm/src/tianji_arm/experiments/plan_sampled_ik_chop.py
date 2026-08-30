#!/usr/bin/env python3
"""Offline FK/IK planner for MuJoCo-style sampled chopping points.

This script does not connect to or command the robot. It uses the SDK kinematics
model only: FK converts an initial joint pose to the starting TCP pose, then the
MuJoCo-style sampled chopping formula generates TCP targets, and IK converts
each target back to a 7-DOF joint target.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from robot_core.paths import local_root
ROOT = local_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_INIT_JOINTS: tuple[float, float, float, float, float, float, float] = (
    109.81,
    -62.66,
    -95.69,
    -93.79,
    63.32,
    -2.76,
    12.42,
)


@dataclass(frozen=True)
class PlanConfig:
    arm: str = "A"
    sdk_root: Path = ROOT
    kine_config: Path = ROOT / "test" / "ccs_m6_40.MvKDCfg"
    init_joints: tuple[float, float, float, float, float, float, float] = DEFAULT_INIT_JOINTS
    control_hz: float = 250.0
    dz_mm: float = -40.0
    hold_s: float = 2.0
    cycles: int = 30
    lateral: bool = True
    lateral_mm: float = 40.0
    chop_axis: str = "y"
    lateral_axis: str = "x"
    lateral_phase: str = "separate"
    stride: int = 1
    output_csv: Path | None = None


def parse_joints(value: str) -> tuple[float, float, float, float, float, float, float]:
    try:
        joints = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("--init-joints must be seven comma-separated floats") from exc
    if len(joints) != 7:
        raise ValueError("--init-joints must contain 7 values")
    if not all(math.isfinite(joint) for joint in joints):
        raise ValueError("--init-joints must contain 7 finite values")
    return joints  # type: ignore[return-value]


def parse_args(argv: list[str] | None = None) -> PlanConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("A", "B"), default="A")
    parser.add_argument("--sdk-root", type=Path, default=ROOT)
    parser.add_argument("--kine-config", type=Path, default=ROOT / "test" / "ccs_m6_40.MvKDCfg")
    parser.add_argument("--init-joints", type=parse_joints, default=DEFAULT_INIT_JOINTS)
    parser.add_argument("--control-hz", type=float, default=250.0)
    parser.add_argument("--dz-mm", type=float, default=-40.0)
    parser.add_argument("--hold-s", type=float, default=2.0)
    parser.add_argument("--cycles", type=int, default=30)
    parser.add_argument("--lateral", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lateral-mm", type=float, default=40.0)
    parser.add_argument("--chop-axis", choices=("x", "y", "z"), default="y", help="Cartesian axis used for the down-up chopping stroke")
    parser.add_argument("--lateral-axis", choices=("x", "y", "z"), default="x", help="Cartesian axis used for lateral shift between cuts")
    parser.add_argument("--lateral-phase", choices=("separate", "retract"), default="separate", help="When to apply lateral shift: after retracting, or during retract like the MuJoCo demo")
    parser.add_argument("--stride", type=int, default=1, help="Print/write every Nth point, always including the final point")
    parser.add_argument("--output-csv", type=Path, default=None)
    config = PlanConfig(**vars(parser.parse_args(argv)))
    validate_config(config)
    return config


def validate_config(config: PlanConfig) -> None:
    if config.arm not in ("A", "B"):
        raise ValueError("arm must be 'A' or 'B'")
    if config.control_hz <= 0.0:
        raise ValueError("control_hz must be positive")
    if config.hold_s <= 0.0:
        raise ValueError("hold_s must be positive")
    if config.cycles <= 0:
        raise ValueError("cycles must be positive")
    axis_index(config.chop_axis)
    axis_index(config.lateral_axis)
    if config.chop_axis == config.lateral_axis and config.lateral:
        raise ValueError("chop-axis and lateral-axis must differ when lateral motion is enabled")
    if config.lateral_phase not in ("separate", "retract"):
        raise ValueError("lateral-phase must be 'separate' or 'retract'")
    if config.stride <= 0:
        raise ValueError("stride must be positive")


def axis_index(axis: str) -> int:
    axis_map = {"x": 0, "y": 1, "z": 2}
    try:
        return axis_map[str(axis).lower()]
    except KeyError as exc:
        raise ValueError("axis must be one of 'x', 'y', or 'z'") from exc


def chop_delta_mm(config: PlanConfig) -> float:
    dz = float(config.dz_mm)
    if str(config.chop_axis).lower() == "y":
        return abs(dz)
    return dz


def cut_cycle_progress(cycle_step: int, steps_per_cycle: int, lateral_phase: str = "separate") -> tuple[float, float, float]:
    if lateral_phase == "retract":
        descent_steps = max(1, int(steps_per_cycle) // 2)
        retract_steps = max(1, int(steps_per_cycle) - descent_steps)
        if int(cycle_step) < descent_steps:
            denominator = max(descent_steps - 1, 1)
            return float(cycle_step) / float(denominator), 0.0, 0.0

        retract_step = min(int(cycle_step) - descent_steps, retract_steps - 1)
        denominator = max(retract_steps - 1, 1)
        retract_progress = float(retract_step) / float(denominator)
        return 1.0 - retract_progress, retract_progress, retract_progress

    vertical_steps = max(2, int(round(float(steps_per_cycle) * 2.0 / 3.0)))
    shift_steps = max(1, int(steps_per_cycle) - vertical_steps)
    descent_steps = max(1, vertical_steps // 2)
    retract_steps = max(1, vertical_steps - descent_steps)
    if int(cycle_step) < descent_steps:
        denominator = max(descent_steps - 1, 1)
        return float(cycle_step) / float(denominator), 0.0, 0.0
    if int(cycle_step) < vertical_steps:
        retract_step = min(int(cycle_step) - descent_steps, retract_steps - 1)
        denominator = max(retract_steps - 1, 1)
        retract_progress = float(retract_step) / float(denominator)
        return 1.0 - retract_progress, retract_progress, 0.0

    if shift_steps == 1:
        return 0.0, 1.0, 1.0
    shift_step = min(int(cycle_step) - vertical_steps, shift_steps - 1)
    shift_progress = float(shift_step) / float(shift_steps - 1)
    return 0.0, 1.0, shift_progress


def build_target_rows(start_xyzabc: np.ndarray, config: PlanConfig) -> list[dict]:
    start = np.asarray(start_xyzabc, dtype=float).reshape(6)
    steps_per_cycle = max(3, int(round(float(config.hold_s) * float(config.control_hz))))
    cycle_count = max(1, int(config.cycles))
    chop_delta = chop_delta_mm(config)
    lateral_step = abs(float(config.lateral_mm)) if config.lateral else 0.0
    chop_idx = axis_index(config.chop_axis)
    lateral_idx = axis_index(config.lateral_axis)

    rows: list[dict] = []
    for step_index in range(steps_per_cycle * cycle_count):
        cycle_index = min(step_index // steps_per_cycle, cycle_count - 1)
        cycle_step = step_index - cycle_index * steps_per_cycle
        z_progress, _retract_progress, shift_progress = cut_cycle_progress(
            cycle_step, steps_per_cycle, config.lateral_phase
        )
        target = start.copy()
        target[chop_idx] = start[chop_idx] + chop_delta * z_progress
        target[lateral_idx] = start[lateral_idx] + lateral_step * (float(cycle_index) + shift_progress)
        row = {
            "step": int(step_index),
            "cycle": int(cycle_index),
            "cycle_step": int(cycle_step),
            "z_progress": float(z_progress),
            "retract_progress": float(_retract_progress),
            "x": float(target[0]),
            "y": float(target[1]),
            "z": float(target[2]),
            "a": float(target[3]),
            "b": float(target[4]),
            "c": float(target[5]),
        }
        rows.append(row)
    return rows


def plan_rows(config: PlanConfig) -> list[dict]:
    _ensure_sdk_path(config.sdk_root)
    from SDK_PYTHON.fx_kine import FX_InvKineSolvePara, Marvin_Kine

    arm_index = 0 if config.arm == "A" else 1
    kine = Marvin_Kine()
    kine.log_switch(0)
    kine_config = kine.load_config(arm_type=arm_index, config_path=str(config.kine_config))
    if not kine.initial_kine(
        robot_type=kine_config["TYPE"][arm_index],
        dh=kine_config["DH"][arm_index],
        pnva=kine_config["PNVA"][arm_index],
        j67=kine_config["BD"][arm_index],
    ):
        raise RuntimeError("initial_kine failed")

    start_joints = [float(value) for value in config.init_joints]
    start_xyzabc = np.asarray(kine.mat4x4_to_xyzabc(kine.fk(start_joints)), dtype=float).reshape(6)
    rows = build_target_rows(start_xyzabc, config)

    ref_joints = list(start_joints)
    for row in rows:
        target_xyzabc = [row[name] for name in ("x", "y", "z", "a", "b", "c")]
        target_matrix = kine.xyzabc_to_mat4x4(target_xyzabc)
        ik_para = FX_InvKineSolvePara()
        ik_para.set_input_ik_target_tcp(kine.mat4x4_to_mat1x16(target_matrix))
        ik_para.set_input_ik_ref_joint(ref_joints)
        ik_para.set_input_ik_zsp_type(0)
        ik = kine.ik(ik_para)
        ik_success, target_joints = _read_ik_result(ik, ref_joints)
        if ik_success:
            ref_joints = target_joints
        row["ik_success"] = bool(ik_success)
        for index in range(7):
            row[f"q{index}"] = target_joints[index] if index < len(target_joints) else 0.0
    return rows


def _read_ik_result(ik_result: object, fallback_joints: Sequence[float]) -> tuple[bool, list[float]]:
    """Read SDK IK output while tolerating boolean failure returns."""
    fallback = [float(value) for value in list(fallback_joints)[:7]]
    if not ik_result:
        return False, fallback
    if not hasattr(ik_result, "get_output_result_num") or not hasattr(ik_result, "get_output_ret_joint"):
        return False, fallback
    success = bool(ik_result.get_output_result_num() >= 1)
    if not success:
        return False, fallback
    joints = [float(value) for value in list(ik_result.get_output_ret_joint())[:7]]
    if len(joints) != 7:
        return False, fallback
    return True, joints


def iter_output_rows(rows: Sequence[dict], stride: int) -> list[dict]:
    step = max(1, int(stride))
    selected: list[dict] = []
    for index, row in enumerate(rows):
        if index % step == 0 or index == len(rows) - 1:
            selected.append(dict(row))
    return selected


def print_rows(rows: Sequence[dict]) -> None:
    fields = _fieldnames()
    print(",".join(fields))
    for row in rows:
        print(",".join(_format_value(row.get(field, "")) for field in fields))


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    fields = _fieldnames()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fieldnames() -> list[str]:
    return [
        "step",
        "cycle",
        "cycle_step",
        "z_progress",
        "retract_progress",
        "x",
        "y",
        "z",
        "a",
        "b",
        "c",
        "ik_success",
        "q0",
        "q1",
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
    ]


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _ensure_sdk_path(sdk_root: Path) -> None:
    root = Path(sdk_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def main(argv: list[str] | None = None) -> int:
    try:
        config = parse_args(argv)
        rows = plan_rows(config)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output_rows = iter_output_rows(rows, config.stride)
    print_rows(output_rows)
    if config.output_csv is not None:
        write_csv(config.output_csv, output_rows)
        print(f"output_csv: {config.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
