# CPU & I/O Usage Prediction (Resource Models)

This repository includes an optional **resource prediction subsystem** that learns to predict:

- **CPU usage** (%)
- **Disk I/O** as **ReadIO** and **WriteIO** (MB/s)

from a DB configuration (knobs) + lightweight workload context. This is primarily used to enable *fast*, resource-aware evaluation (especially in `BenchEnv`) without running a full benchmark each time.

---

## What’s in the codebase

- **Data collection**: `scripts/collect_resource_data.py`
- **Training**: `scripts/train_resource_model.py`
- **Accuracy eval (CPU)**: `scripts/eval_cpu_accuracy.py`
- **Accuracy eval (Write I/O)**: `scripts/eval_io_accuracy.py`
- **Runtime loader (compat layer)**: `autotune/utils/resource_model_loader.py`
- **On-DB resource monitor (ground truth)**: `autotune/resource_monitor.py`
- **Collected data (default location)**: `resource_data/resource_data.json`
- **Trained model bundle (default location)**: `resource_models/resource_predictor.joblib`
- **Training metadata**: `resource_models/training_metadata.json`

> Note on I/O evaluation: `ReadIO` is frequently near-zero in typical runs, which makes MAPE unstable. The repo’s evaluation scripts report **WriteIO-only MAPE** by design.

---

## Prerequisites

- Python dependencies: `pip install -r requirements.txt`
- Export: `export PYTHONPATH="."`
- For collection (ground-truth labels), you must have a DB + workload runnable (example below uses **MySQL + sysbench**):
  - MySQL running and accessible via socket (often `/var/run/mysqld/mysqld.sock`)
  - Sysbench installed
  - A prepared dataset (see `README.md` for sysbench dataset setup examples)

---

## 1) Collect training data (CPU/IO labels)

This runs multiple short benchmark trials and records knobs + CPU/IO measurements.

```bash
export PYTHONPATH="."
export MYSQL_SOCK=/var/run/mysqld/mysqld.sock

python scripts/collect_resource_data.py \
  --num_samples 60 \
  --output_dir resource_data \
  --workload_time 30 \
  --workload_warmup_time 5
```

### Output files

- `resource_data/resource_data.json`: final dataset (appended by default)
- `resource_data/resource_data_intermediate.json`: checkpoint saves during collection
- `resource_data/collection_config.ini`: auto-generated collection config (if you didn’t pass `--config`)

### Tips for better accuracy

- Increase signal (less noise) by increasing benchmark time:
  - e.g. `--workload_time 90 --workload_warmup_time 10`
- Collect more samples (80–150) if CPU MAPE is high.

---

## 2) Train the resource models

```bash
export PYTHONPATH="."

python scripts/train_resource_model.py \
  --data_file resource_data/resource_data.json \
  --knob_config scripts/experiment/gen_knobs/mysql_cpu_io_dynamic_15.json \
  --knob_num 15 \
  --output_dir resource_models \
  --num_trees 200
```

### Model artifact

The training script writes a single bundle:

- `resource_models/resource_predictor.joblib`

and metadata:

- `resource_models/training_metadata.json`

The bundle format supports two backends (selected by `--model_backend`):

- `rfwi` (default): `RandomForestWithInstances` (may be non-pickleable → bundle stores training data and retrains in-memory at load time)
- `sklearn_extratrees`: stores picklable sklearn models directly (often works very well on small datasets)

---

## 3) Evaluate accuracy (MAPE)

CPU held-out MAPE (preferred: reads from `resource_models/training_metadata.json`):

```bash
export PYTHONPATH="."
python scripts/eval_cpu_accuracy.py
```

Write I/O held-out MAPE:

```bash
export PYTHONPATH="."
python scripts/eval_io_accuracy.py
```

---

## 4) Use the model at runtime

### In `BenchEnv` (surrogate benchmark environment)

`autotune/dbenv_bench.py` will automatically load `resource_models/resource_predictor.joblib` if present and then augment `get_states()` with predicted CPU/IO for each configuration.

### Loader behavior (important)

The loader `autotune/utils/resource_model_loader.py` exists because one training backend stores *training data* instead of pickled model objects. The loader guarantees a consistent interface by:

- returning `model_cpu`, `model_read_io`, `model_write_io` objects that support `.predict(X) -> (mean, var)`
- retraining in memory when needed

---

## Troubleshooting

- **No model found**: ensure `resource_models/resource_predictor.joblib` exists.
- **No data found**: ensure `resource_data/resource_data.json` exists (run collection first).
- **High CPU MAPE**:
  - increase `--workload_time`
  - increase `--num_samples`
  - try `--model_backend sklearn_extratrees`
- **I/O MAPE looks “weird”**: this repo evaluates **WriteIO-only**; ReadIO is often too close to zero for stable MAPE.


