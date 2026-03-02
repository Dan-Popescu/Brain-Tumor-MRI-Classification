#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_common.sh
source "${script_dir}/_common.sh"

config_path="${1:-conf/spark_transform.yaml}"
python_cmd="$(resolve_python_cmd)"

cd "${project_root}"
exec "${python_cmd}" src/spark_jobs/transform_job.py --config "${config_path}"
