#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_common.sh
source "${script_dir}/_common.sh"

config_path="${1:-conf/train.yaml}"
python_cmd="$(resolve_python_cmd)"

cd "${project_root}"
exec "${python_cmd}" src/train.py --config "${config_path}"
