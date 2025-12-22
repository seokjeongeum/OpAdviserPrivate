# DB Knob Suggestion (Tuning for Higher TPS)

This README explains how this repository **suggests DB knob settings** to improve **TPS**.

In this codebase, a “knob suggestion” is the **best configuration found by the tuner** for your target workload, stored in:

- `repo/history_<task_id>.json`

You can then **apply** that configuration to the database (online or offline) and optionally rerun the benchmark to verify TPS.

---

## What’s in the codebase

### Main entrypoints

- **Run tuning**: `scripts/optimize.py`
- **Tuning driver**: `autotune/tuner.py` (`DBTuner`)
- **DB + benchmark environment**: `autotune/dbenv.py` (`DBEnv`)
- **Knob definitions**: `scripts/experiment/gen_knobs/*.json`
- **Apply a suggested config from a history file**: `scripts/experiment/apply_knob.py`

### Outputs you should expect

- `repo/history_<task_id>.json`: tuning history (each trial includes `configuration` and `external_metrics.tps`)
- `logs/` and `log/`: run logs
- `<task_id>.png`: convergence plot for single-objective runs

---

## Prerequisites

- Python deps: `pip install -r requirements.txt`
- Export: `export PYTHONPATH="."`
- A runnable DB + benchmark setup (MySQL or PostgreSQL)
- For MySQL, many configs assume:
  - `export MYSQL_SOCK=/var/run/mysqld/mysqld.sock`

See:
- `documents/database_setting.md` (DB setup options)
- `documents/tuning_setting.md` (tuning knobs/optimizer/transfer settings)

---

## 1) Choose your knob search space

The tuner searches within a knob JSON file under `scripts/experiment/gen_knobs/`.

Common examples for MySQL:

- **Dynamic-only (online mode)**:
  - `scripts/experiment/gen_knobs/mysql_dynamic_10.json`
  - `scripts/experiment/gen_knobs/mysql_cpu_io_dynamic_15.json`
- **CPU/IO-focused (broader, may require restarts depending on knobs)**:
  - `scripts/experiment/gen_knobs/mysql_cpu_io_40.json`
- **Larger spaces**:
  - `scripts/experiment/gen_knobs/mysql_all_197_32G.json`

If you’re just starting, use a smaller dynamic set first (faster iterations).

---

## 2) Run tuning (generate a “suggested” config)

The normal workflow is to run:

```bash
export PYTHONPATH="."
python scripts/optimize.py --config=scripts/config.ini
```

Most experiments in this repo are driven by `.ini` files under `scripts/` (see also the root `README.md` for ready-to-run presets like fast/ultrafast).

### Key config settings that control TPS tuning

In your `.ini`:

- `performance_metric = ['tps']` (single-objective: maximize TPS)
- `task_id = <unique_name>` (output history will be `repo/history_<task_id>.json`)
- `knob_config_file = <path/to/json>`
- `knob_num = <how many knobs from the JSON to tune>`
- `optimize_method = SMAC | DDPG | MBO | GA | ...`
- `space_transfer = True|False` (transfer/pruning)
- `transfer_framework = none | rgpe | workload_map | finetune`
- `online_mode = True|False`

See `documents/tuning_setting.md` for details.

---

## 3) Locate the suggested config in the history file

After the run, find:

- `repo/history_<task_id>.json`

The best-TPS configuration is the trial whose `external_metrics.tps` is maximal.

### Quick extraction (one-liner)

```bash
python - <<'PY'
import json, sys
path = sys.argv[1]
j = json.load(open(path))
best = max(j["data"], key=lambda x: x.get("external_metrics", {}).get("tps", float("-inf")))
print("best_tps:", best.get("external_metrics", {}).get("tps"))
print("configuration:")
print(best["configuration"])
PY repo/history_<task_id>.json
```

---

## 4) Apply the suggested config and verify TPS

This repo includes a helper that:
1) picks a configuration from `repo/history_<task_id>.json`
2) applies it
3) runs **one** benchmark trial to report TPS/lat/qps

```bash
export PYTHONPATH="."

python scripts/experiment/apply_knob.py \
  --history repo/history_<task_id>.json \
  --pick best_tps \
  --config scripts/config.ini
```

You can also use `--pick default` to apply the first configuration in the history.

---

## Notes: online vs offline knob application

### Online mode (`online_mode=True`)

- Faster iterations (no DB restart)
- Only knobs supported by `apply_knobs_online()` are safe to tune
- Recommended knob sets: dynamic-only JSONs (e.g. `mysql_dynamic_10.json`)

### Offline mode (`online_mode=False`)

- Allows static knobs, but each trial may require DB restart
- Slower, but can reach higher peak performance when important knobs are static

---

## Where “knob suggestion” fits in this project

- `scripts/optimize.py` runs the tuner (`DBTuner`) against a real DB+benchmark (`DBEnv`)
- each trial records:
  - `configuration` (the knob values)
  - `external_metrics` (TPS/lat/QPS)
  - resource metrics (CPU/IO/etc) collected by `ResourceMonitor` when enabled
- the suggested knob setting is simply **the best trial so far** for your objective


