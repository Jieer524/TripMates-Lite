"""Schema-validating summary for candidate-window sensitivity."""
import argparse
from pathlib import Path
import pandas as pd
REQUIRED=("experiment_id","scenario_id","planner","seed","candidate_window_size","feasible","failure_reason","achieved_utility","completed_poi_count","retained_poi_count","removed_poi_count","replacement_poi_count","changed_activity_count","shifted_activity_count","order_change_count","itinerary_edit_distance","wes","total_outdoor_exposure_min","total_travel_min","runtime_seconds","bounded_repair_feasible","actual_modification_count","bounded_repair_limit")
def main(smoke=False):
    p=Path("results/ablation_studies/candidate_window_sensitivity")/("smoke" if smoke else "")/"raw_runs.csv"
    df=pd.read_csv(p); missing=[c for c in REQUIRED if c not in df.columns]
    if missing: raise SystemExit("Missing required columns: "+", ".join(missing))
    s=df.groupby(["planner","candidate_window_size"],dropna=False).agg(count=("planner","size"),feasibility_rate=("feasible","mean"),mean_utility=("achieved_utility","mean"),median_utility=("achieved_utility","median"),std_utility=("achieved_utility","std"),mean_wes=("wes","mean"),mean_edit_distance=("itinerary_edit_distance","mean"),mean_runtime=("runtime_seconds","mean")).reset_index()
    s.to_csv(p.parent/"planner_summary.csv",index=False)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--smoke",action="store_true");main(p.parse_args().smoke)
