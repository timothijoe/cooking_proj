import argparse
from pathlib import Path
import time

import numpy as np

from twin_sim.guarded_chop_playback import play_guarded_chop_recording
from twin_sim.guarded_chop_recording import validate_replay_rate
from twin_sim.kinematics import Kinematics
from twin_sim.pick_place_visualization import PickPlaceTrace
from twin_sim.robot import RIGHT_HOME_RAD, RightArmRobot
from twin_sim.tasks.chop import ChopConfig, run_chop
from twin_sim.tasks.line_chop import LineChopConfig, run_line_chop
from twin_sim.tasks.hand_demo import HandDemoConfig, run_hand_demo
from twin_sim.tasks.guarded_chop import GuardedChopConfig, run_guarded_chop
from twin_sim.tasks.recorded_hand_guarded_chop import (
    RecordedHandGuardedChopConfig,
    run_recorded_hand_guarded_chop,
)
from twin_sim.tasks.pick_place import PickPlaceConfig, PickPlaceTask
from twin_sim.trajectory import cartesian_trajectory, joint_trajectory
from twin_sim.gamepad_teleop import TeleopConfig, run_gamepad_teleop


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "gamepad-teleop":
        run_gamepad_teleop(TeleopConfig(
            device=args.device, speed_mm_s=args.speed_mm_s,
            workspace_radius_mm=args.workspace_radius_mm, deadzone=args.deadzone,
            arm=args.arm,
        ))
        return 0
    if args.command == "hand-demo":
        run_hand_demo(
            HandDemoConfig(
                close_duration_s=4.0 if args.slow else 2.0,
                open_duration_s=4.0 if args.slow else 2.0,
                viewer_start_hold_s=5.0 if args.slow else 0.0,
                viewer_end_hold_s=10.0 if args.slow else 0.0,
            ),
            viewer=not args.headless,
        )
        return 0
    if args.command == "chop":
        config = (
            ChopConfig(
                control_dt_s=args.control_dt,
                orient_duration_s=3.0,
                approach_duration_s=3.0,
                descent_duration_s=3.0,
                hold_duration_s=1.0,
                retract_duration_s=3.0,
                viewer_start_hold_s=5.0,
                viewer_end_hold_s=8.0,
                full_motion=args.full_motion,
            )
            if args.slow
            else ChopConfig(
                control_dt_s=args.control_dt,
                full_motion=args.full_motion,
            )
        )
        run_chop(
            config,
            log_path=args.log,
            viewer=not args.headless,
        )
        return 0
    if args.command == "line-chop":
        chop_config = (
            ChopConfig(
                control_dt_s=args.control_dt,
                descent_duration_s=3.0,
                hold_duration_s=1.0,
                retract_duration_s=3.0,
                viewer_start_hold_s=5.0,
                viewer_end_hold_s=8.0,
            )
            if args.slow
            else ChopConfig(control_dt_s=args.control_dt)
        )
        run_line_chop(
            LineChopConfig(
                chop=chop_config,
                cuts=args.cuts,
                spacing_m=args.spacing_m,
                shift_duration_s=3.0 if args.slow else 1.0,
            ),
            log_path=args.log,
            plot_path=args.plot,
            viewer=not args.headless,
        )
        return 0
    if args.command == "pick-place":
        final_hold_s = (
            args.final_hold
            if args.final_hold is not None
            else (5.0 if not args.headless else 0.0)
        )
        robot = RightArmRobot(viewer=not args.headless)
        try:
            trace = PickPlaceTrace(robot._viewer)
            result = PickPlaceTask(
                robot,
                PickPlaceConfig(final_hold_s=final_hold_s),
                trace=trace,
                realtime=args.slow and args.headless,
            ).run()
            cube = np.asarray(
                [sample.cube_position for sample in result.samples],
                dtype=float,
            )
            target = robot.sim.data.site_xpos[
                robot.sim.require_site("pick_target_site")
            ]
            lift = float(cube[:, 2].max() - cube[0, 2]) if len(cube) else 0.0
            transfer = (
                float(np.linalg.norm(cube[-1, :2] - cube[0, :2]))
                if len(cube)
                else 0.0
            )
            target_error = (
                float(np.linalg.norm(cube[-1, :2] - target[:2]))
                if len(cube)
                else float("nan")
            )
            max_force = max(
                (
                    sample.grasp.max_hand_actuator_force
                    for sample in result.samples
                ),
                default=0.0,
            )
            print(
                f"success={result.success} phase={result.final_phase.value} "
                f"lift_m={lift:.3f} transfer_m={transfer:.3f} "
                f"target_error_m={target_error:.3f} "
                f"max_hand_force={max_force:.3f} "
                f"reason={result.reason or '-'}"
            )
            return 0 if result.success else 1
        finally:
            robot.close()
    if args.command == "guarded-chop":
        if args.headless and args.replay_rate is not None:
            parser.error("guarded-chop replay requires a Viewer")
        result = run_guarded_chop(
            GuardedChopConfig(
                scene_mode=args.scene,
                final_hold_s=args.final_hold,
            ),
            viewer=not args.headless,
            replay_rate=args.replay_rate,
            record_path=args.record,
        )
        print(
            f"success={result.success} cuts={result.completed_cuts} "
            f"shifts={result.completed_shifts} "
            f"total_shift_m={result.total_shift_m:.3f} "
            f"min_distance_m={result.minimum_distance_m:.3f} "
            f"reason={result.reason or '-'}"
        )
        return 0 if result.success else 1
    if args.command == "recorded-hand-guarded-chop":
        result = run_recorded_hand_guarded_chop(
            RecordedHandGuardedChopConfig(final_hold_s=args.final_hold),
            hand_mcap=args.hand_mcap,
            viewer=not args.headless,
        )
        print(
            f"success={result.success} cuts={result.completed_cuts} "
            f"hand_cycles={result.completed_hand_cycles} "
            f"surface_offset_m={result.surface_offset_m:.3f} "
            f"min_distance_m={result.minimum_distance_m:.3f} "
            f"lateral_spacing_m={result.minimum_lateral_spacing_m:.3f}.."
            f"{result.maximum_lateral_spacing_m:.3f} "
            f"clearance_tier_m={result.selected_clearance_tier_m:.3f} "
            f"selected_spacing_m={result.selected_lateral_spacing_m:.3f} "
            f"depth_mismatch_m={result.maximum_depth_mismatch_m:.3f} "
            f"pad_clearance_m={float(np.min(result.minimum_long_pad_clearances_m)):.3f}.."
            f"{float(np.max(result.minimum_long_pad_clearances_m)):.3f} "
            f"max_penetration_m={result.maximum_hand_penetration_m:.6f} "
            f"thumb_clearance_m={result.minimum_thumb_clearance_m:.3f} "
            f"reason={result.reason or '-'}"
        )
        return 0 if result.success else 1
    if args.command == "guarded-chop-replay":
        try:
            rate = validate_replay_rate(args.rate)
        except ValueError as error:
            parser.error(str(error))
        play_guarded_chop_recording(args.recording, rate=rate)
        return 0
    if args.command == "view":
        robot = RightArmRobot(viewer=True)
        try:
            while robot._viewer is not None and robot._viewer.is_running():
                robot.step(0.01)
                time.sleep(0.01)
        finally:
            robot.close()
        return 0

    robot = RightArmRobot(viewer=not args.headless)
    try:
        if args.command == "joint":
            if not 1 <= args.joint <= 7:
                parser.error("--joint must be between 1 and 7")
            goal = RIGHT_HOME_RAD.copy()
            goal[args.joint - 1] += args.delta_rad
            points = joint_trajectory(RIGHT_HOME_RAD, goal, 0.5, 0.01)
        else:
            kinematics = Kinematics(robot.sim)
            start = kinematics.fk(RIGHT_HOME_RAD)
            goal = start.copy()
            goal[2, 3] += args.dz_m
            points = cartesian_trajectory(
                kinematics, start, goal, RIGHT_HOME_RAD, 0.5, 0.01
            )
        robot.validate_targets([point.joints_rad for point in points])
        for point in points:
            robot.command(point.joints_rad)
            robot.step(0.01)
        return 0
    finally:
        robot.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="twin-sim")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("view")
    gamepad = commands.add_parser("gamepad-teleop")
    gamepad.add_argument("--device", type=Path, default=TeleopConfig().device)
    gamepad.add_argument("--speed-mm-s", type=float, default=100.0)
    gamepad.add_argument("--workspace-radius-mm", type=float, default=350.0)
    gamepad.add_argument("--deadzone", type=float, default=0.15)
    gamepad.add_argument("--arm", choices=("right", "left"), default="right")
    hand_demo = commands.add_parser("hand-demo")
    hand_demo.add_argument("--headless", action="store_true")
    hand_demo.add_argument("--slow", action="store_true")
    joint = commands.add_parser("joint")
    joint.add_argument("--joint", type=int, required=True)
    joint.add_argument("--delta-rad", type=float, required=True)
    joint.add_argument("--headless", action="store_true")
    cartesian = commands.add_parser("cartesian")
    cartesian.add_argument("--dz-m", type=float, required=True)
    cartesian.add_argument("--headless", action="store_true")
    chop = commands.add_parser("chop")
    chop.add_argument("--headless", action="store_true")
    chop.add_argument("--log", type=Path, required=True)
    chop.add_argument("--control-dt", type=float, default=0.01)
    chop.add_argument("--slow", action="store_true")
    chop.add_argument("--full-motion", action="store_true")
    line_chop = commands.add_parser("line-chop")
    line_chop.add_argument("--cuts", type=int, default=5)
    line_chop.add_argument("--spacing-m", type=float, default=0.03)
    line_chop.add_argument("--headless", action="store_true")
    line_chop.add_argument("--log", type=Path, required=True)
    line_chop.add_argument("--plot", type=Path, required=True)
    line_chop.add_argument("--control-dt", type=float, default=0.01)
    line_chop.add_argument("--slow", action="store_true")
    pick_place = commands.add_parser("pick-place")
    pick_place.add_argument("--headless", action="store_true")
    pick_place.add_argument("--slow", action="store_true")
    pick_place.add_argument("--final-hold", type=float)
    guarded_chop = commands.add_parser("guarded-chop")
    guarded_chop.add_argument("--headless", action="store_true")
    guarded_chop.add_argument(
        "--scene", choices=("plane", "object"), default="plane"
    )
    guarded_chop.add_argument(
        "--final-hold",
        type=float,
        default=GuardedChopConfig().final_hold_s,
    )
    guarded_chop.add_argument("--replay-rate", type=float)
    guarded_chop.add_argument(
        "--record",
        type=Path,
        nargs="?",
        const=Path("recordings/guarded_chop_latest.npz"),
    )
    recorded_guarded_chop = commands.add_parser(
        "recorded-hand-guarded-chop"
    )
    recorded_guarded_chop.add_argument("--headless", action="store_true")
    recorded_guarded_chop.add_argument(
        "--hand-mcap",
        type=Path,
        default=Path(
            "recordings/wuji/august_02/"
            "session_20260802_174440_936_right_to_left_wuji_hand.mcap"
        ),
    )
    recorded_guarded_chop.add_argument(
        "--final-hold",
        type=float,
        default=RecordedHandGuardedChopConfig().final_hold_s,
    )
    guarded_replay = commands.add_parser("guarded-chop-replay")
    guarded_replay.add_argument(
        "--recording",
        type=Path,
        default=Path("recordings/guarded_chop_latest.npz"),
    )
    guarded_replay.add_argument("--rate", type=float, default=2.0)
    return parser


_parser = build_parser


if __name__ == "__main__":
    raise SystemExit(main())
