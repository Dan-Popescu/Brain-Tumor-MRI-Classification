# Brain Tumor MRI Classification

Projet de classification d'IRM cérébrales avec pipeline Spark (préparation des données) et TensorFlow (entraînement).

## Démarrage rapide (équipe)

Ces étapes sont celles à suivre sur une nouvelle machine.

1. Cloner le repo et se placer à la racine:
```bash
git clone <repo-url>
cd MRI-Brain-Tumor-Classification
```

2. Créer l'environnement conda standardisé:
```bash
scripts/bootstrap_conda.sh
```

Par sécurité, le script **ne met pas à jour un environnement existant** sans flag explicite.
Pour mettre à jour un env déjà créé:
```bash
scripts/bootstrap_conda.sh --update
```

Le bootstrap force aussi le solver conda `classic` (sans modifier ta config globale)
pour éviter les erreurs locales liées à `libmamba`.

3. Activer l'environnement:
```bash
conda activate mri-brain-tumor
```

4. Vérifier la configuration locale (Java, Spark, TensorFlow, chemins):
```bash
python scripts/doctor.py
```

5. Lancer le pipeline:
```bash
scripts/run_pipeline.sh
```

## Option GPU (Linux)

Pour installer les extras TensorFlow CUDA lors du bootstrap:
```bash
scripts/bootstrap_conda.sh --with-gpu
```

Puis:
```bash
conda activate mri-brain-tumor
python scripts/doctor.py
```

## Exécution par étape

Si vous voulez lancer les jobs un par un:

```bash
scripts/run_preprocess.sh
scripts/run_transform.sh
scripts/run_split.sh
scripts/run_training_tfrecord.sh
scripts/run_train.sh
```

Sans entraînement final:
```bash
scripts/run_pipeline.sh --no-train
```

## Configurations

Les configs se trouvent dans `conf/`:

- `conf/spark_preprocess.yaml`
- `conf/spark_transform.yaml`
- `conf/spark_split.yaml`
- `conf/spark_training_tfrecord.yaml`
- `conf/train.yaml`

Chaque wrapper accepte un chemin de config en argument:
```bash
scripts/run_preprocess.sh conf/spark_preprocess.yaml
```

Pour utiliser un nom d'environnement différent:
```bash
scripts/bootstrap_conda.sh --name mri-brain-tumor-alice
```
Puis, pour le mettre à jour:
```bash
scripts/bootstrap_conda.sh --name mri-brain-tumor-alice --update
```

## Pourquoi ces scripts évitent les erreurs fréquentes

- Les wrappers `scripts/run_*.sh` forcent l'exécution depuis la racine du projet, ce qui stabilise les chemins relatifs.
- L'activation conda configure automatiquement:
  - `JAVA_HOME=$CONDA_PREFIX`
  - `PYSPARK_PYTHON=$CONDA_PREFIX/bin/python`
  - `PYSPARK_DRIVER_PYTHON=$CONDA_PREFIX/bin/python`

## Dépannage rapide

1. `No module named pyspark` / `tensorflow`:
   - vérifier que l'environnement est activé:
   ```bash
   conda activate mri-brain-tumor
   python scripts/doctor.py
   ```

2. Erreur Java/Spark:
   - relancer le bootstrap puis réactiver l'env:
   ```bash
   scripts/bootstrap_conda.sh --update
   conda activate mri-brain-tumor
   python scripts/doctor.py
   ```

3. Erreur de chemin:
   - ne pas lancer les modules Python directement depuis un sous-dossier;
   - utiliser `scripts/run_*.sh`.
