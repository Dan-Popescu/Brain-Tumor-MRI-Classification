#!/usr/bin/env python3
"""Local environment diagnostics for Spark + TensorFlow project setup."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout.strip()


def _ok(message: str) -> None:
    print(f"[OK]   {message}")


def _warn(message: str) -> None:
    print(f"[WARN] {message}")


def _fail(message: str) -> None:
    print(f"[FAIL] {message}")


def main() -> int:
    has_error = False

    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)

    print("=== MRI Brain Tumor Classification: Doctor ===")
    print(f"Project root: {project_root}")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")

    if os.environ.get("CONDA_PREFIX"):
        _ok(f"Conda env active: {os.environ['CONDA_PREFIX']}")
    else:
        _warn("No active conda environment detected.")

    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        _ok(f"JAVA_HOME={java_home}")
    else:
        _warn("JAVA_HOME is not set. Spark may fail depending on local Java setup.")

    code, java_output = _run(["java", "-version"])
    if code == 0:
        first_line = java_output.splitlines()[0] if java_output else "java detected"
        _ok(first_line)
    else:
        has_error = True
        _fail("`java -version` failed. Install or configure Java (OpenJDK 17 recommended).")

    try:
        import pyspark  # type: ignore

        _ok(f"pyspark {pyspark.__version__}")
    except Exception as exc:  # pragma: no cover - diagnostic script
        has_error = True
        _fail(f"Cannot import pyspark: {exc}")

    try:
        import tensorflow as tf  # type: ignore

        _ok(f"tensorflow {tf.__version__}")
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            _ok(f"TensorFlow GPU devices: {len(gpus)}")
            for idx, gpu in enumerate(gpus):
                print(f"      - GPU[{idx}]: {gpu.name}")
        else:
            _warn("No TensorFlow GPU detected.")
    except Exception as exc:  # pragma: no cover - diagnostic script
        has_error = True
        _fail(f"Cannot import tensorflow: {exc}")

    try:
        from pyspark.sql import SparkSession  # type: ignore

        spark = SparkSession.builder.master("local[1]").appName("doctor-check").getOrCreate()
        count = spark.range(1).count()
        spark.stop()
        if count == 1:
            _ok("SparkSession local test passed.")
        else:
            has_error = True
            _fail("SparkSession started but returned unexpected test result.")
    except Exception as exc:  # pragma: no cover - diagnostic script
        has_error = True
        _fail(f"Spark local test failed: {exc}")

    required_paths = [
        project_root / "conf" / "spark_preprocess.yaml",
        project_root / "conf" / "spark_transform.yaml",
        project_root / "conf" / "spark_split.yaml",
        project_root / "conf" / "spark_training_tfrecord.yaml",
        project_root / "conf" / "train.yaml",
    ]

    for path in required_paths:
        if path.exists():
            _ok(f"Found {path.relative_to(project_root)}")
        else:
            has_error = True
            _fail(f"Missing {path.relative_to(project_root)}")

    raw_data_dir = project_root / "data" / "raw"
    if raw_data_dir.exists():
        _ok("Found data/raw directory.")
    else:
        _warn("data/raw directory missing. Spark preprocess will fail until data is present.")

    if has_error:
        print("\nDoctor finished with errors.")
        return 1

    print("\nDoctor finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
