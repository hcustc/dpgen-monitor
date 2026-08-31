#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
credentials_file="${DPGEN_MONITOR_ENV_FILE:-$script_dir/runtime/private/dpgen_monitor.env}"
conda_env="${DPGEN_MONITOR_CONDA_ENV:-feishu_bot}"

if [[ -f "$credentials_file" ]]; then
    # Export assignments from the private credentials file to the monitor process.
    set -a
    # shellcheck disable=SC1090
    source "$credentials_file"
    set +a
fi

cd "$script_dir"
if [[ -n "${CONDA_EXE:-}" ]]; then
    conda_bin="$CONDA_EXE"
elif command -v conda >/dev/null 2>&1; then
    conda_bin="$(command -v conda)"
else
    echo "dpgen-monitor: conda not found; set CONDA_EXE or add conda to PATH" >&2
    exit 127
fi

exec "$conda_bin" run --no-capture-output -n "$conda_env" python -m dpgen_monitor "$@"
