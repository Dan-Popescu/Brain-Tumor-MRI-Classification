#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"

resolve_python_cmd() {
  if [[ -n "${PYSPARK_PYTHON:-}" ]]; then
    printf '%s\n' "${PYSPARK_PYTHON}"
    return
  fi

  if command -v python >/dev/null 2>&1; then
    printf '%s\n' "python"
    return
  fi

  if command -v python3 >/dev/null 2>&1; then
    printf '%s\n' "python3"
    return
  fi

  printf 'No python executable found in PATH.\n' >&2
  exit 1
}

