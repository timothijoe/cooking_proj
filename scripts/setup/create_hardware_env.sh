#!/bin/sh
set -eu

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
python3.12 -m venv "$PROJECT_ROOT/.venv-hardware"
"$PROJECT_ROOT/.venv-hardware/bin/python" -m pip install --upgrade pip setuptools wheel
"$PROJECT_ROOT/.venv-hardware/bin/python" -m pip install -e "$PROJECT_ROOT[hardware,data,teleop]"
"$PROJECT_ROOT/.venv-hardware/bin/python" -m pip install -e "$PROJECT_ROOT/packages/robot_core"
"$PROJECT_ROOT/.venv-hardware/bin/python" -m pip install -e "$PROJECT_ROOT/packages/wuji_hand[data]"
"$PROJECT_ROOT/.venv-hardware/bin/python" -m pip install -e "$PROJECT_ROOT/packages/tianji_arm"
"$PROJECT_ROOT/.venv-hardware/bin/python" -m pip install -e "$PROJECT_ROOT/packages/cooking_workflows"

LOCAL_ROOT=${COOKING_LOCAL_ROOT:-$PROJECT_ROOT/local}
SITE_PACKAGES=$("$PROJECT_ROOT/.venv-hardware/bin/python" -c 'import site; print(site.getsitepackages()[0])')
printf '%s\n' "$LOCAL_ROOT" > "$SITE_PACKAGES/cooking_local_vendor.pth"

printf '%s\n' "Place vendor SDKs under $PROJECT_ROOT/local/vendor before using hardware commands."
