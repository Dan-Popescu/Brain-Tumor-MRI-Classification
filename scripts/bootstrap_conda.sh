#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
env_file="${project_root}/environment.yml"

env_name="mri-brain-tumor"
with_gpu=0
update_existing=0

usage() {
  cat <<'USAGE'
Usage: scripts/bootstrap_conda.sh [options]

Options:
  -n, --name <env_name>   Conda environment name (default: mri-brain-tumor)
  --update                Update existing environment (safe-guard is off)
  --with-gpu              Install tensorflow CUDA extras (Linux only)
  -h, --help              Show help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--name)
      env_name="$2"
      shift 2
      ;;
    --update)
      update_existing=1
      shift
      ;;
    --with-gpu)
      with_gpu=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n' "$1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  printf 'conda is required but was not found in PATH.\n' >&2
  exit 1
fi

run_conda() {
  CONDA_SOLVER=classic conda --no-plugins "$@"
}

if [[ ! -f "${env_file}" ]]; then
  printf 'Environment file not found: %s\n' "${env_file}" >&2
  exit 1
fi

cd "${project_root}"

if ! envs_json="$(run_conda env list --json 2>&1)"; then
  printf '[bootstrap] Failed to run `conda env list --json`.\n%s\n' "${envs_json}" >&2
  exit 1
fi

if ! env_exists="$(
  printf '%s' "${envs_json}" | python3 -c '
import json
import os
import sys

target = sys.argv[1]
payload = json.load(sys.stdin)
default_prefix = payload.get("default_prefix")

def env_name(prefix: str) -> str:
    if default_prefix and prefix == default_prefix:
        return "base"
    return os.path.basename(prefix)

exists = any(
    env_name(prefix) == target or prefix == target
    for prefix in payload.get("envs", [])
)
print("1" if exists else "0")
' "${env_name}"
)"; then
  printf '[bootstrap] Failed to parse `conda env list --json` output.\n' >&2
  exit 1
fi

if [[ "${env_exists}" == "1" ]]; then
  if [[ "${update_existing}" != "1" ]]; then
    cat <<EOF >&2
[bootstrap] Safety check: conda env "${env_name}" already exists.
No update was performed to avoid overwriting local setup.

If you want to update this environment, run:
  scripts/bootstrap_conda.sh --update --name "${env_name}"

If you want to keep your current env untouched, create another one:
  scripts/bootstrap_conda.sh --name "${env_name}-<yourname>"
EOF
    exit 2
  fi

  printf '[bootstrap] Updating existing conda env "%s"...\n' "${env_name}"
  run_conda env update -n "${env_name}" -f "${env_file}" --prune
else
  printf '[bootstrap] Creating conda env "%s"...\n' "${env_name}"
  run_conda env create -n "${env_name}" -f "${env_file}"
fi

printf '[bootstrap] Installing activation hooks...\n'
"${script_dir}/install_conda_hooks.sh" "${env_name}"

if [[ "${with_gpu}" == "1" ]]; then
  if [[ "$(uname -s)" != "Linux" ]]; then
    printf '[bootstrap] --with-gpu is currently supported only on Linux.\n' >&2
    exit 1
  fi

  printf '[bootstrap] Installing TensorFlow CUDA extras in "%s"...\n' "${env_name}"
  run_conda run -n "${env_name}" pip install --upgrade "tensorflow[and-cuda]==2.16.2"
fi

cat <<EOF

Bootstrap complete.
Next steps:
  1) conda activate ${env_name}
  2) python scripts/doctor.py
  3) scripts/run_preprocess.sh
EOF
