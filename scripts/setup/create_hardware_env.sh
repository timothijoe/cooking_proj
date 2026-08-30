#!/bin/sh
set -eu

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
python3.12 -m venv "$PROJECT_ROOT/.venv-hardware"
"$PROJECT_ROOT/.venv-hardware/bin/python" -m pip install --upgrade pip setuptools wheel
"$PROJECT_ROOT/.venv-hardware/bin/python" -m pip install -e "$PROJECT_ROOT[hardware,data,teleop]"

printf '%s\n' "Place vendor SDKs under $PROJECT_ROOT/local/vendor before using hardware commands."
