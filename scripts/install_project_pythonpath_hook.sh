#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
project_src="${project_root}/src"
env_name="${1:-mri-brain-tumor}"

if ! command -v conda >/dev/null 2>&1; then
  printf 'conda is required but was not found in PATH.\n' >&2
  exit 1
fi

run_conda() {
  CONDA_SOLVER=classic conda --no-plugins "$@"
}

env_prefix=""

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  current_env_name="$(basename "${CONDA_PREFIX}")"
  if [[ "${env_name}" == "${current_env_name}" || "${env_name}" == "${CONDA_PREFIX}" ]]; then
    env_prefix="${CONDA_PREFIX}"
  fi
fi

if [[ -z "${env_prefix}" ]]; then
  if ! envs_json="$(run_conda env list --json 2>&1)"; then
    printf '[pythonpath-hooks] Failed to run `conda env list --json`.\n%s\n' "${envs_json}" >&2
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
    printf '[pythonpath-hooks] Failed to parse `conda env list --json` output.\n' >&2
    exit 1
  fi
fi

if [[ -z "${env_prefix}" ]]; then
  printf 'Conda environment "%s" was not found.\n' "${env_name}" >&2
  exit 1
fi

activate_dir="${env_prefix}/etc/conda/activate.d"
deactivate_dir="${env_prefix}/etc/conda/deactivate.d"
mkdir -p "${activate_dir}" "${deactivate_dir}"

cat > "${activate_dir}/mri-project-pythonpath.sh" <<EOF
#!/usr/bin/env bash
export _MRI_PROJECT_OLD_PYTHONPATH="\${PYTHONPATH-}"

if [[ ":\${PYTHONPATH-}:" == *":${project_src}:"* ]]; then
  :
elif [[ -z "\${PYTHONPATH-}" ]]; then
  export PYTHONPATH="${project_src}"
else
  export PYTHONPATH="${project_src}:\${PYTHONPATH}"
fi
EOF

cat > "${deactivate_dir}/mri-project-pythonpath.sh" <<'EOF'
#!/usr/bin/env bash

if [[ -n "${_MRI_PROJECT_OLD_PYTHONPATH+x}" ]]; then
  if [[ -n "${_MRI_PROJECT_OLD_PYTHONPATH}" ]]; then
    export PYTHONPATH="${_MRI_PROJECT_OLD_PYTHONPATH}"
  else
    unset PYTHONPATH
  fi
  unset _MRI_PROJECT_OLD_PYTHONPATH
fi
EOF

chmod 0644 \
  "${activate_dir}/mri-project-pythonpath.sh" \
  "${deactivate_dir}/mri-project-pythonpath.sh"

printf 'Installed project PYTHONPATH hooks in %s\n' "${env_prefix}"
printf 'Project src added on activation: %s\n' "${project_src}"
