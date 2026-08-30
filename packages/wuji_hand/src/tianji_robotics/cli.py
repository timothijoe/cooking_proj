"""Top-level command line boundary for simulation and real hardware."""

import argparse
from dataclasses import replace
from pathlib import Path
import sys


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tianji-robot")
    domains = parser.add_subparsers(dest="domain", required=True)

    sim = domains.add_parser("sim", help="offline and simulated execution")
    sim_commands = sim.add_subparsers(dest="sim_command", required=True)
    replay = sim_commands.add_parser("wuji-replay", help="replay a Wuji Studio recording")
    replay.add_argument("source", type=Path)
    replay.add_argument("--headless", action="store_true")
    replay.add_argument("--npz", type=Path)
    replay.add_argument("--joint-state-mcap", type=Path)
    replay.set_defaults(handler=_run_wuji_replay)

    retreat = sim_commands.add_parser("wuji-table-retreat", help="derive a four-finger tabletop retreat")
    retreat.add_argument("source", type=Path)
    retreat.add_argument("--headless", action="store_true")
    retreat.add_argument("--retreat-distance", type=float, default=0.03)
    retreat.add_argument("--place-duration", type=float, default=1.0)
    retreat.add_argument("--retreat-duration", type=float, default=2.0)
    retreat.add_argument("--hold-duration", type=float, default=2.0)
    retreat.add_argument("--table-height", type=float, default=0.0)
    retreat.add_argument("--source-frame", type=int)
    retreat.add_argument("--loops", type=_positive_int, default=3)
    retreat.add_argument("--npz", type=Path)
    retreat.add_argument("--report", type=Path)
    retreat.set_defaults(handler=_run_wuji_table_retreat)

    angle_bar = sim_commands.add_parser(
        "wuji-angle-bar",
        help="locally control one simulated Wuji hand with angle bars",
    )
    angle_bar.add_argument("--hand", choices=("left", "right"), default="left")
    angle_bar.set_defaults(handler=_run_wuji_angle_bar)

    hardware = domains.add_parser("hardware", help="guarded real-device tools")
    vendors = hardware.add_subparsers(dest="hardware_vendor", required=True)
    wuji = vendors.add_parser("wuji-sdk", help="Wuji SDK boundary")
    wuji_commands = wuji.add_subparsers(dest="wuji_command", required=True)
    preflight = wuji_commands.add_parser("preflight", help="validate an offline NPZ only")
    preflight.add_argument("trajectory", type=Path)
    preflight.set_defaults(handler=lambda args: _preflight_trajectory(args.trajectory))
    return parser


def _run_wuji_replay(args: argparse.Namespace) -> int:
    try:
        from tianji_robotics.data.mcap import StudioMcapSkeletonSource, write_joint_state_mcap
        from tianji_robotics.data.npz import save_trajectory_npz
        from tianji_robotics.simulation.replay import replay_trajectory
        from tianji_robotics.simulation.wuji_hand import MujocoWujiHand
        from tianji_robotics.workflows.wuji_glove_replay import retarget_recording
        from tianji_robotics.wuji_sdk.retargeter import OfficialWujiRetargeter

        trajectory = retarget_recording(
            StudioMcapSkeletonSource(args.source),
            OfficialWujiRetargeter.create_left_first_generation(),
        )
        if args.npz:
            save_trajectory_npz(trajectory, args.npz)
        if args.joint_state_mcap:
            write_joint_state_mcap(trajectory, args.joint_state_mcap)
        backend = MujocoWujiHand(viewer=not args.headless)
        try:
            summary = replay_trajectory(trajectory, backend, realtime=not args.headless)
        finally:
            backend.close()
        print(f"replayed {summary.frame_count} frames ({summary.duration_s:.3f}s)")
        return 0
    except ModuleNotFoundError as exc:
        if exc.name in {"mcap", "wuji_sdk"}:
            print(
                "missing Wuji offline dependency; install with "
                "'pip install -e .[wuji-offline]'",
                file=sys.stderr,
            )
            return 2
        raise


def _preflight_trajectory(path: Path) -> int:
    from tianji_robotics.data.npz import load_trajectory_npz

    trajectory = load_trajectory_npz(path)
    duration_s = float(trajectory.timestamps_ns[-1] - trajectory.timestamps_ns[0]) / 1e9
    print(f"trajectory valid: {trajectory.positions_rad.shape[0]} frames, {duration_s:.3f}s; no device accessed")
    return 0


def _run_wuji_table_retreat(args: argparse.Namespace) -> int:
    try:
        from tianji_robotics.data.mcap import (
            JointStateMcapSource,
            StudioMcapSkeletonSource,
            detect_hand_mcap_kind,
        )
        from tianji_robotics.data.table_retreat import save_corrected_trajectory_npz, write_correction_report_json
        from tianji_robotics.simulation.tabletop_wuji_hand import TabletopWujiHand
        from tianji_robotics.workflows.wuji_glove_replay import retarget_recording
        from tianji_robotics.workflows.wuji_table_retreat import TableRetreatConfig, build_looped_table_retreat, build_recorded_table_retreat, replay_table_retreat

        source_kind = detect_hand_mcap_kind(args.source)
        if source_kind == "joint_states":
            trajectory = JointStateMcapSource(args.source).trajectory()
        else:
            from tianji_robotics.wuji_sdk.retargeter import OfficialWujiRetargeter
            trajectory = retarget_recording(
                StudioMcapSkeletonSource(args.source),
                OfficialWujiRetargeter.create_left_first_generation(),
            )
        planner=TabletopWujiHand(viewer=False,table_height_m=args.table_height)
        try:
            corrected,report=build_recorded_table_retreat(trajectory,planner,TableRetreatConfig(retreat_distance_m=args.retreat_distance,place_duration_s=args.place_duration,retreat_duration_s=args.retreat_duration,hold_duration_s=args.hold_duration,source_frame=args.source_frame),source_kind=source_kind)
            looped=build_looped_table_retreat(corrected,planner,loops=args.loops)
        finally: planner.close()
        if args.npz: save_corrected_trajectory_npz(corrected,args.npz)
        backend=TabletopWujiHand(viewer=not args.headless,table_height_m=args.table_height)
        try:
            summary=replay_table_retreat(looped,backend)
            if not args.headless and not summary.stopped_early:
                backend.wait_until_viewer_closes()
        finally: backend.close()
        report=replace(report,requested_loop_count=args.loops,executed_loop_count=summary.loop_count,scheduled_playback_duration_s=summary.scheduled_duration_s,stopped_early=summary.stopped_early)
        if args.report: write_correction_report_json(report,args.report)
        print(f"table retreat: source_kind={report.source_kind} frames={len(corrected.timestamps_ns)} loops={report.executed_loop_count}/{report.requested_loop_count} scheduled_s={report.scheduled_playback_duration_s:.3f} motion={report.motion_start_frame}:{report.motion_end_frame} retreat_m={report.actual_retreat_m:.3f} palm_down={report.palm_down_verified} thumb_clearance_m={report.minimum_thumb_clearance_m:.4f} penetration_m={report.maximum_hand_penetration_m:.6f} max_joint_correction_rad={report.maximum_joint_correction_rad:.4f}")
        return 0
    except ModuleNotFoundError as exc:
        if exc.name in {"mcap","wuji_sdk"}:
            print("missing Wuji offline dependency; install with 'pip install -e .[wuji-offline]'",file=sys.stderr)
            return 2
        raise


def _run_wuji_angle_bar(args: argparse.Namespace) -> int:
    """Launch the local MuJoCo-only Wuji angle-bar panel lazily."""
    from tianji_robotics.simulation.local_angle_bar import run_local_angle_bar

    return run_local_angle_bar(args.hand)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.handler(args))
