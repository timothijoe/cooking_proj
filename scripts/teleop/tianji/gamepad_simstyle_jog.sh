#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
export PYTHONPATH="${COOKING_LOCAL_ROOT:-$ROOT/local}:$ROOT/packages/robot_core/src:$ROOT/packages/tianji_arm/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$ROOT/.venv-hardware/bin/python" -m tianji_arm.experiments.gamepad_cartesian_jog_simstyle "$@"
