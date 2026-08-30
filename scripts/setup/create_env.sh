#!/bin/sh
set -eu

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
python3.12 -m venv "$PROJECT_ROOT/.venv"
"$PROJECT_ROOT/.venv/bin/python" -m pip install --upgrade pip setuptools wheel
"$PROJECT_ROOT/.venv/bin/python" -m pip install -e "$PROJECT_ROOT[simulation,data,teleop,dev]"
"$PROJECT_ROOT/.venv/bin/python" -m pip install -e "$PROJECT_ROOT/packages/robot_core"
"$PROJECT_ROOT/.venv/bin/python" -m pip install -e "$PROJECT_ROOT/packages/wuji_hand[data,simulation]"
"$PROJECT_ROOT/.venv/bin/python" -m pip install -e "$PROJECT_ROOT/packages/cooking_simulation"
"$PROJECT_ROOT/.venv/bin/python" -m pip install -e "$PROJECT_ROOT/packages/tianji_arm"
"$PROJECT_ROOT/.venv/bin/python" -m pip install -e "$PROJECT_ROOT/packages/cooking_workflows"
