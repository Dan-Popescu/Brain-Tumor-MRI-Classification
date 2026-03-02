# Setup Conda (Spark + Java + TensorFlow)

## 1) Create environment

```bash
scripts/bootstrap_conda.sh
```

Default environment name: `mri-brain-tumor`.

Safety behavior: if that env already exists, bootstrap stops and asks for explicit update.
Bootstrap also forces conda solver `classic` for its own commands to avoid
local `libmamba` backend compatibility issues.

To update an existing environment:

```bash
scripts/bootstrap_conda.sh --update
```

To use a dedicated per-user environment:

```bash
scripts/bootstrap_conda.sh --name mri-brain-tumor-alice
```

With TensorFlow CUDA extras on Linux:

```bash
scripts/bootstrap_conda.sh --with-gpu
```

## 2) Activate environment

```bash
conda activate mri-brain-tumor
```

Java/PySpark variables are configured automatically on activation via conda hooks:

- `JAVA_HOME=$CONDA_PREFIX`
- `PYSPARK_PYTHON=$CONDA_PREFIX/bin/python`
- `PYSPARK_DRIVER_PYTHON=$CONDA_PREFIX/bin/python`

## 3) Validate setup

```bash
python scripts/doctor.py
```

## 4) Run jobs with stable paths (recommended)

These wrappers always run from repository root, so relative paths in `conf/*.yaml` are stable.

```bash
scripts/run_preprocess.sh
scripts/run_transform.sh
scripts/run_split.sh
scripts/run_training_tfrecord.sh
scripts/run_train.sh
```

Or run the full pipeline:

```bash
scripts/run_pipeline.sh
```

Without training:

```bash
scripts/run_pipeline.sh --no-train
```
