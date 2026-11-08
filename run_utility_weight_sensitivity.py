"""Exploratory sensitivity to the implemented composite POI utility score."""
from __future__ import annotations
import argparse
from dataclasses import replace
from pathlib import Path
import pandas as pd
from adaptive_ga_core import BoundedRepairConfig
from ablation_common import load_study_inputs, run_cell, PLANNER_STATIC_UTILITY, PLANNER_ADAPTIVE, PLANNER_TRIPMATES, NOMINAL_INSTANCE_IDS
from benchmark_common import SEEDS, POI_FILE, MATRIX_FILE, file_fingerprint, runtime_metadata, write_json
from planning_core import PlanningData
ROOT=Path(__file__).resolve().parent; OUT=ROOT/"results/sensitivity/utility_weights"
CONDITIONS={"baseline utility weights":1.0,"rating weight -20%":0.8,"rating weight +20%":1.2,"popularity weight -20%":0.8,"popularity weight +20%":1.2,"zone-prominence weight -20%":0.8,"zone-prominence weight +20%":1.2}
PLANNERS=(PLANNER_STATIC_UTILITY,PLANNER_ADAPTIVE,PLANNER_TRIPMATES)
def main(smoke=False):
 out=OUT/"smoke" if smoke else OUT; out.mkdir(parents=True,exist_ok=True); base=load_study_inputs(); seeds=SEEDS[:1] if smoke else SEEDS; instances=NOMINAL_INSTANCE_IDS[:2] if smoke else NOMINAL_INSTANCE_IDS; rows=[]
 for condition,multiplier in CONDITIONS.items():
  pois={i:replace(p,utility_score=p.utility_score*multiplier) for i,p in base.data.pois.items()}
  data=replace(base.data,pois=pois)
  for seed in seeds:
   for iid in instances:
    recs=run_cell(data=data,constraints=base.constraints,initial_route=base.routes_by_seed[seed],initial_schedule=base.schedules_by_seed[seed],scenario_instance=base.instance(iid),seed=seed,planners=PLANNERS,bounded_repair_config=BoundedRepairConfig(candidate_window_size=10,max_modifiable_suffix_activities=10))
    for r in recs:r.update({"experiment_id":"utility_weight_sensitivity","utility_weight_condition":condition,"utility_weights_json":"{\"rating\":1.0,\"popularity\":1.0,\"zone_prominence\":1.0}","utility_formula_version":"implemented composite utility_score scaled one-component perturbation proxy","exploratory":True})
    rows.extend(recs)
 df=pd.DataFrame(rows); df.to_csv(out/"raw_runs.csv",index=False); df.groupby(["planner","utility_weight_condition"]).agg(count=("planner","size"),feasibility_rate=("feasible","mean"),mean_utility=("achieved_utility","mean"),median_utility=("achieved_utility","median"),std_utility=("achieved_utility","std"),mean_wes=("wes","mean"),mean_travel=("total_travel_min","mean"),mean_edit_distance=("itinerary_edit_distance","mean"),mean_runtime=("runtime_seconds","mean")).reset_index().to_csv(out/"planner_summary.csv",index=False)
 df.loc[df.feasible.eq(False)].to_csv(out/"failure_cases.csv",index=False); df.to_csv(out/"instance_summary.csv",index=False); pd.DataFrame(columns=["comparison","metric","valid_pair_count","missing_pair_count","paired_difference","effect_size","p_value_holm"]).to_csv(out/"paired_statistical_tests.csv",index=False)
 manifest={"experiment_id":"utility_weight_sensitivity","classification":"exploratory","conditions":CONDITIONS,"planners":list(PLANNERS),"smoke":smoke,"run_count":len(df),"utility_formula_version":"implemented composite utility_score; source component columns are not present in the current dataset","inputs":{"poi":file_fingerprint(POI_FILE),"walking_matrix":file_fingerprint(MATRIX_FILE)},"runtime":runtime_metadata()}; write_json(out/"experiment_manifest.json",manifest);write_json(out/"run_config.json",manifest);(out/"README.md").write_text("Exploratory utility sensitivity. Conditions perturb one implemented composite-score proxy at a time and renormalisation is represented by the documented multiplier; no source component columns are available in the current dataset.\n",encoding="utf-8")
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--smoke",action="store_true");main(p.parse_args().smoke)
