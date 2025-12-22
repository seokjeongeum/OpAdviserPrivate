import argparse
import json
from typing import Any, Dict, Literal

from autotune.database.mysqldb import MysqlDB
from autotune.database.postgresqldb import PostgresqlDB
from autotune.dbenv import DBEnv
from autotune.utils.config import parse_args


def _pick_configuration_from_history(
    *,
    history_path: str,
    pick: Literal["default", "best_tps"],
) -> Dict[str, Any]:
    """
    Pick a knob configuration from a `repo/history_<task_id>.json` file.

    - default: the first trial's configuration (usually defaults / initial sample)
    - best_tps: the configuration with the highest `external_metrics.tps`
    """
    with open(history_path, "r") as f:
        history: Dict[str, Any] = json.load(f)

    trials: list[Dict[str, Any]] = history.get("data", [])
    if not trials:
        raise ValueError(f"No trials found in history file: {history_path}")

    if pick == "default":
        return dict(trials[0]["configuration"])

    # pick == "best_tps"
    best_cfg: Dict[str, Any] | None = None
    best_tps: float = float("-inf")
    for t in trials:
        ext: Dict[str, Any] = dict(t.get("external_metrics", {}) or {})
        tps_val: float = float(ext.get("tps", float("-inf")))
        if tps_val > best_tps and "configuration" in t:
            best_tps = tps_val
            best_cfg = dict(t["configuration"])

    if best_cfg is None:
        raise ValueError(
            f"Could not find a trial with external_metrics.tps in history file: {history_path}"
        )
    return best_cfg


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply a configuration from a tuning history and run 1 benchmark to verify TPS/lat/qps.",
    )
    parser.add_argument(
        "--history",
        required=True,
        help="Path to `repo/history_<task_id>.json` produced by `python scripts/optimize.py ...`",
    )
    parser.add_argument(
        "--pick",
        default="best_tps",
        choices=["default", "best_tps"],
        help="Which configuration to apply from history (default: best_tps).",
    )
    parser.add_argument(
        "--config",
        default="scripts/config.ini",
        help="DBTune/OpAdviser INI config used to connect to the DB + run the benchmark (default: scripts/config.ini).",
    )
    args = parser.parse_args()

    knobs: Dict[str, Any] = _pick_configuration_from_history(
        history_path=str(args.history),
        pick=args.pick,  # type: ignore[arg-type]
    )

    args_db, args_tune = parse_args(str(args.config))
    if args_db["db"] == "mysql":
        db = MysqlDB(args_db)
    elif args_db["db"] == "postgresql":
        db = PostgresqlDB(args_db)
    else:
        raise ValueError(f"Unsupported db type: {args_db['db']}")

    env = DBEnv(args_db, args_tune, db)

    # Apply + run a single benchmark trial to verify.
    timeout, metrics, internal_metrics, resource = env.step_GP(knobs, collect_resource=True)
    tps: float = float(metrics[0]) if metrics and len(metrics) >= 1 else float("nan")
    lat: float = float(metrics[1]) if metrics and len(metrics) >= 2 else float("nan")
    qps: float = float(metrics[2]) if metrics and len(metrics) >= 3 else float("nan")

    print(f"pick={args.pick} history={args.history}")
    print(f"tps={tps} lat={lat} qps={qps} timeout={timeout}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())








