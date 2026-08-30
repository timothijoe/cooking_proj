#!/bin/sh
set -eu

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
python3.12 -m venv "$PROJECT_ROOT/.venv"
"$PROJECT_ROOT/.venv/bin/python" -m pip install --upgrade pip setuptools wheel
"$PROJECT_ROOT/.venv/bin/python" -m pip install -e "$PROJECT_ROOT[simulation,data,teleop,dev]"
