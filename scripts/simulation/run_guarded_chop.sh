#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
TWIN_SIM="${REPO_ROOT}/.venv/bin/twin-sim"

if [[ ! -x "${TWIN_SIM}" ]]; then
  printf '%s\n' \
    "Missing executable: ${TWIN_SIM}" \
    "Install the simulation environment from ${REPO_ROOT}:" \
    "  ./scripts/setup/create_env.sh" >&2
  exit 1
fi

cd -- "${REPO_ROOT}"
exec "${TWIN_SIM}" guarded-chop --scene plane
