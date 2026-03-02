#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_train=1

usage() {
  cat <<'USAGE'
Usage: scripts/run_pipeline.sh [--no-train]

Runs:
  1) preprocess
  2) transform
  3) split
  4) training_tfrecord
  5) train (unless --no-train)

Environment overrides:
  PREPROCESS_CONFIG
  TRANSFORM_CONFIG
  SPLIT_CONFIG
  TFRECORD_CONFIG
  TRAIN_CONFIG
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-train)
      run_train=0
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

"${script_dir}/run_preprocess.sh" "${PREPROCESS_CONFIG:-conf/spark_preprocess.yaml}"
"${script_dir}/run_transform.sh" "${TRANSFORM_CONFIG:-conf/spark_transform.yaml}"
"${script_dir}/run_split.sh" "${SPLIT_CONFIG:-conf/spark_split.yaml}"
"${script_dir}/run_training_tfrecord.sh" "${TFRECORD_CONFIG:-conf/spark_training_tfrecord.yaml}"

if [[ "${run_train}" == "1" ]]; then
  "${script_dir}/run_train.sh" "${TRAIN_CONFIG:-conf/train.yaml}"
fi
