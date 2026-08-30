"""Safely plan, and optionally execute, a small right index-MCP motion."""

import argparse

from tianji_robotics.hardware.wuji_hand.index_mcp_test import (
    build_index_mcp_test_plan,
    run_index_mcp_test,
)


def _create_hand(serial_number: str):
    """Connect to the requested hand only after the CLI is invoked."""
    import wujihandpy

    return wujihandpy.Hand(serial_number=serial_number)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan or explicitly execute a guarded right index-MCP test."
    )
    parser.add_argument("--serial-number", required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="enable the planned motion; omit for a no-motion dry-run",
    )
    parser.add_argument("--dwell-s", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        hand = _create_hand(args.serial_number)
        plan = build_index_mcp_test_plan(hand)
        print(f"current angle: {plan.current_rad:.6f} rad")
        print("targets: " + ", ".join(f"{target:.6f}" for target in plan.targets_rad) + " rad")
        print(f"max temperature: {plan.max_temperature_c:.1f}°C")
        if not args.execute:
            print("dry-run: no motion will be commanded")
        run_index_mcp_test(hand, plan, execute=args.execute, dwell_s=args.dwell_s)
        return 0
    except Exception as error:
        print(f"preflight failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
