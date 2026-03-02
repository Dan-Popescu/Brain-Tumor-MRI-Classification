#!/usr/bin/env bash

if [[ "${_MRI_OLD_JAVA_HOME+x}" == "x" ]]; then
  if [[ -n "${_MRI_OLD_JAVA_HOME}" ]]; then
    export JAVA_HOME="${_MRI_OLD_JAVA_HOME}"
  else
    unset JAVA_HOME
  fi
  unset _MRI_OLD_JAVA_HOME
fi

if [[ "${_MRI_OLD_PYSPARK_PYTHON+x}" == "x" ]]; then
  if [[ -n "${_MRI_OLD_PYSPARK_PYTHON}" ]]; then
    export PYSPARK_PYTHON="${_MRI_OLD_PYSPARK_PYTHON}"
  else
    unset PYSPARK_PYTHON
  fi
  unset _MRI_OLD_PYSPARK_PYTHON
fi

if [[ "${_MRI_OLD_PYSPARK_DRIVER_PYTHON+x}" == "x" ]]; then
  if [[ -n "${_MRI_OLD_PYSPARK_DRIVER_PYTHON}" ]]; then
    export PYSPARK_DRIVER_PYTHON="${_MRI_OLD_PYSPARK_DRIVER_PYTHON}"
  else
    unset PYSPARK_DRIVER_PYTHON
  fi
  unset _MRI_OLD_PYSPARK_DRIVER_PYTHON
fi

