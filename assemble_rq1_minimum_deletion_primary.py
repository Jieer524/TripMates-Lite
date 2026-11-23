"""Assemble the RQ1 minimum-deletion comparison from recorded runs.

The 198 fixed-order static rows were evaluated separately. The remaining 540
rows come unchanged from the recorded skip-and-continue benchmark, because the
static execution rule does not affect Greedy Repair or either adaptive planner.
This script does not claim that all 738 rows were generated in one run.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from run_rq1_full_benchmark import (
    build_overall_planner_summary,
    build_planner_scenario_summary,
)


ROOT = Path(__file__).resolve().parent
RECORDED_DIR = ROOT / "results/rq1_full_benchmark_corrected_static_rule"
MINIMUM_DIR = ROOT / "results/rq1_minimal_edit_static"
OUTPUT_DIR = ROOT / "results/rq1_full_benchmark_minimum_deletion_primary"
MINIMUM_MODE = "minimum_deletion_fixed_order_execution"
STATIC_PLANNERS = {"Static Distance", "Static Utility"}
PAIRING_FIELDS = (
    "planner", "instance_id", "seed", "initial_route", "completed_prefix",
    "original_remaining_route", "scenario_id", "trigger_variant",
)


def fingerprint(path: Path) -> dict:
    data = path.read_bytes()
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def key(row: pd.Series) -> tuple:
    seed = None if pd.isna(row["seed"]) else int(row["seed"])
    return row["planner"], row["instance_id"], seed


def main() -> None:
    recorded_path = RECORDED_DIR / "rq1_raw_runs.csv"
    static_path = MINIMUM_DIR / "minimal_static_utility_raw_runs.csv"
    config_path = RECORDED_DIR / "rq1_run_config.json"
    recorded = pd.read_csv(recorded_path)
    minimum = pd.read_csv(static_path)
    if list(recorded.columns) != list(minimum.columns):
        raise ValueError("The two raw files have different columns")
    if len(recorded) != 738 or len(minimum) != 198:
        raise ValueError("Unexpected source row counts")
    if set(minimum["planner"]) != STATIC_PLANNERS:
        raise ValueError("Minimum-deletion source contains nonstatic planners")
    if set(minimum["static_execution_mode"]) != {MINIMUM_MODE}:
        raise ValueError("Minimum-deletion rows have an incorrect execution mode")

    minimum_by_key = {}
    for _, row in minimum.iterrows():
        row_key = key(row)
        if row_key in minimum_by_key:
            raise ValueError(f"Duplicate minimum-deletion row: {row_key}")
        minimum_by_key[row_key] = row

    used = set()
    rows = []
    for _, row in recorded.iterrows():
        if row["planner"] not in STATIC_PLANNERS:
            rows.append(row)
            continue
        row_key = key(row)
        if row_key not in minimum_by_key:
            raise ValueError(f"Missing minimum-deletion row: {row_key}")
        replacement = minimum_by_key[row_key]
        for field in PAIRING_FIELDS:
            old, new = row[field], replacement[field]
            if pd.isna(old) and pd.isna(new):
                continue
            if old != new:
                raise ValueError(f"Input mismatch for {row_key}: {field}")
        used.add(row_key)
        rows.append(replacement)
    if used != set(minimum_by_key):
        raise ValueError("Unmatched minimum-deletion rows")

    combined = pd.DataFrame(rows, columns=recorded.columns)
    if len(combined) != 738 or combined["planner"].value_counts().to_dict() != {
        "Static Distance": 18,
        "Static Utility": 180,
        "Greedy Repair": 180,
        "Adaptive Without Thermal": 180,
        "TripMates Lite": 180,
    }:
        raise ValueError("Unexpected assembled comparison composition")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_DIR / "rq1_raw_runs.csv", index=False, float_format="%.6f")
    build_planner_scenario_summary(combined).to_csv(
        OUTPUT_DIR / "rq1_planner_scenario_summary.csv",
        index=False, float_format="%.6f",
    )
    build_overall_planner_summary(combined).to_csv(
        OUTPUT_DIR / "rq1_overall_planner_summary.csv",
        index=False, float_format="%.6f",
    )
    combined[~combined["feasible"]].to_csv(
        OUTPUT_DIR / "rq1_failure_cases.csv", index=False, float_format="%.6f",
    )

    recorded_config = json.loads(config_path.read_text(encoding="utf-8"))
    config = {
        "experiment": "RQ1 minimum-deletion primary comparison",
        "mode": "assembled_reanalysis",
        "assembled_utc": datetime.now(timezone.utc).isoformat(),
        "source_artifacts": {
            "minimum_deletion_static_rows": fingerprint(static_path),
            "recorded_nonstatic_rows": fingerprint(recorded_path),
            "recorded_run_config": fingerprint(config_path),
        },
        "composition": {
            "minimum_deletion_static_rows": 198,
            "unchanged_greedy_and_adaptive_rows": 540,
            "total_rows": 738,
        },
        "static_execution_mode": MINIMUM_MODE,
        "static_rule": (
            "Keep the maximum number of original unfinished POIs in a feasible "
            "order-preserving subsequence; break ties by utility, travel time, "
            "then POI ID. No reordering or substitution."
        ),
        "sensitivity_comparison": (
            "results/rq1_full_benchmark_corrected_static_rule"
        ),
        "planners": recorded_config["planners"],
        "instance_ids": recorded_config["instance_ids"],
        "seeds": recorded_config["seeds"],
        "candidate_window_limit": recorded_config["candidate_window_limit"],
        "ga_configuration": recorded_config["ga_configuration"],
        "stability_weights": recorded_config["stability_weights"],
        "dataset": recorded_config["dataset"],
        "matrix": recorded_config["matrix"],
        "initial_routes": recorded_config["initial_routes"],
        "seed_interpretation": recorded_config["seed_interpretation"],
        "failure_count": int((~combined["feasible"]).sum()),
    }
    (OUTPUT_DIR / "rq1_run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(OUTPUT_DIR)
    print(build_overall_planner_summary(combined)[
        ["planner", "run_count", "mean_itinerary_completion_ratio", "mean_original_poi_retention"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
