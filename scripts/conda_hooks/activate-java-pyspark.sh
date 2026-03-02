#!/usr/bin/env bash

# Persist previous values so deactivate can restore them.
export _MRI_OLD_JAVA_HOME="${JAVA_HOME:-}"
export _MRI_OLD_PYSPARK_PYTHON="${PYSPARK_PYTHON:-}"
export _MRI_OLD_PYSPARK_DRIVER_PYTHON="${PYSPARK_DRIVER_PYTHON:-}"

# Use OpenJDK installed in the active conda environment.
export JAVA_HOME="${CONDA_PREFIX}"
export PYSPARK_PYTHON="${CONDA_PREFIX}/bin/python"
export PYSPARK_DRIVER_PYTHON="${CONDA_PREFIX}/bin/python"

if [[ ":${PATH}:" != *":${JAVA_HOME}/bin:"* ]]; then
  export PATH="${JAVA_HOME}/bin:${PATH}"
fi

