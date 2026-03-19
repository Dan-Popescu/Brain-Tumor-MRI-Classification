#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/_common.sh"

python_cmd="$(resolve_python_cmd)"
config_path="${1:-conf/spark_inference.yaml}"

cd "${project_root}"
exec "${python_cmd}" src/spark_jobs/inference_job.py --config "${config_path}"
