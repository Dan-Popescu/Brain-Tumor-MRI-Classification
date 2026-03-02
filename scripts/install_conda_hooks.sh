#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
hooks_src_dir="${script_dir}/conda_hooks"
env_name="${1:-mri-brain-tumor}"

if ! command -v conda >/dev/null 2>&1; then
  printf 'conda is required but was not found in PATH.\n' >&2
  exit 1
fi

run_conda() {
  CONDA_SOLVER=classic conda --no-plugins "$@"
}

if ! envs_json="$(run_conda env list --json 2>&1)"; then
  printf '[hooks] Failed to run `conda env list --json`.\n%s\n' "${envs_json}" >&2
  exit 1
fi

if ! env_prefix="$(
  printf '%s' "${envs_json}" | python3 -c '
import json
import os
import sys

target = sys.argv[1]
payload = json.load(sys.stdin)
default_prefix = payload.get("default_prefix")

if target == "base" and default_prefix:
    print(default_prefix)
    raise SystemExit(0)

for prefix in payload.get("envs", []):
    name = "base" if default_prefix and prefix == default_prefix else os.path.basename(prefix)
    if name == target or prefix == target:
        print(prefix)
        break
' "${env_name}"
)"; then
  printf '[hooks] Failed to parse `conda env list --json` output.\n' >&2
  exit 1
fi

if [[ -z "${env_prefix}" ]]; then
  printf 'Conda environment "%s" was not found.\n' "${env_name}" >&2
  exit 1
fi

activate_dir="${env_prefix}/etc/conda/activate.d"
deactivate_dir="${env_prefix}/etc/conda/deactivate.d"
mkdir -p "${activate_dir}" "${deactivate_dir}"

install -m 0644 \
  "${hooks_src_dir}/activate-java-pyspark.sh" \
  "${activate_dir}/mri-java-pyspark.sh"

install -m 0644 \
  "${hooks_src_dir}/deactivate-java-pyspark.sh" \
  "${deactivate_dir}/mri-java-pyspark.sh"

printf 'Installed conda hooks in %s\n' "${env_prefix}"
