"""Exploratory candidate-window sensitivity for the bounded replanner."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from adaptive_ga_core import BoundedRepairConfig
from ablation_common import load_study_inputs, run_cell, PLANNER_ADAPTIVE, PLANNER_TRIPMATES, NOMINAL_INSTANCE_IDS
from benchmark_common import SEEDS, runtime_metadata, file_fingerprint, POI_FILE, MATRIX_FILE, write_json

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/ablation_studies/candidate_window_sensitivity"
LEVELS = (5, 10, 15, "all")
PLANNERS = (PLANNER_ADAPTIVE, PLANNER_TRIPMATES)

def main(smoke=False):
    out = OUT / "smoke" if smoke else OUT
    out.mkdir(parents=True, exist_ok=True)
    inputs = load_study_inputs()
    seeds = SEEDS[:1] if smoke else SEEDS
    instances = NOMINAL_INSTANCE_IDS[:2] if smoke else NOMINAL_INSTANCE_IDS
    rows = []
    for level in LEVELS:
        cfg = BoundedRepairConfig(candidate_window_size=level, max_modifiable_suffix_activities=10)
        for seed in seeds:
            route = inputs.routes_by_seed[seed]
            for instance_id in instances:
                records = run_cell(data=inputs.data, constraints=inputs.constraints,
                    initial_route=route, initial_schedule=inputs.schedules_by_seed[seed],
                    scenario_instance=inputs.instance(instance_id), seed=seed,
                    planners=PLANNERS, bounded_repair_config=cfg)
                for r in records:
                    r.update({"experiment_id":"candidate_window_sensitivity", "candidate_window_size":level,
                              "maximum_modifiable_suffix_activities":10,
                              "candidate_window_condition":str(level), "exploratory":True})
                rows.extend(records)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "raw_runs.csv", index=False)
    summaries = frame.groupby(["planner","candidate_window_size"], dropna=False).agg(
        count=("planner","size"), feasibility_rate=("feasible","mean"),
        mean_utility=("achieved_utility","mean"), median_utility=("achieved_utility","median"),
        std_utility=("achieved_utility","std"), mean_wes=("wes","mean"),
        mean_edit_distance=("itinerary_edit_distance","mean"), mean_runtime=("runtime_seconds","mean")
    ).reset_index()
    summaries.to_csv(out / "planner_summary.csv", index=False)
    frame.loc[frame.feasible.eq(False)].to_csv(out / "failure_cases.csv", index=False)
    frame.groupby(["scenario_id","seed","planner","candidate_window_size"], dropna=False).first().reset_index().to_csv(out / "instance_summary.csv", index=False)
    pd.DataFrame(columns=["comparison","metric","valid_pair_count","missing_pair_count","paired_difference","effect_size","p_value_holm"]).to_csv(out / "paired_statistical_tests.csv", index=False)
    manifest={"experiment_id":"candidate_window_sensitivity","classification":"exploratory","levels":list(LEVELS),"planners":list(PLANNERS),"seeds":list(seeds),"instances":list(instances),"candidate_window_controls_max_modifiable_suffix_activities":10,"smoke":smoke,"run_count":len(frame),"inputs":{"poi":file_fingerprint(POI_FILE),"walking_matrix":file_fingerprint(MATRIX_FILE)},"runtime":runtime_metadata()}
    write_json(out / "experiment_manifest.json", manifest); write_json(out / "run_config.json", manifest)
    (out / "README.md").write_text("Exploratory candidate-window sensitivity. Window 10 remains the proposed configuration; 15 and all-eligible test search-space access only. The modification limit is fixed at 10.\n", encoding="utf-8")

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--smoke",action="store_true"); main(p.parse_args().smoke)
