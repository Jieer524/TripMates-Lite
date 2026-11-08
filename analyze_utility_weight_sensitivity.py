"""Schema-validating utility-weight sensitivity analysis."""
import argparse
from pathlib import Path
import pandas as pd
REQUIRED=("experiment_id","scenario_id","planner","seed","utility_weight_condition","feasible","failure_reason","achieved_utility","completed_poi_count","retained_poi_count","wes","total_travel_min","itinerary_edit_distance","runtime_seconds")
def main(smoke=False):
 p=Path("results/sensitivity/utility_weights")/("smoke" if smoke else "")/"raw_runs.csv"; df=pd.read_csv(p); missing=[c for c in REQUIRED if c not in df]
 if missing: raise SystemExit("Missing required columns: "+", ".join(missing))
 df.groupby(["planner","utility_weight_condition"]).agg(count=("planner","size"),feasibility_rate=("feasible","mean"),mean_utility=("achieved_utility","mean"),median_utility=("achieved_utility","median"),std_utility=("achieved_utility","std"),mean_wes=("wes","mean"),mean_travel=("total_travel_min","mean"),mean_edit_distance=("itinerary_edit_distance","mean"),mean_runtime=("runtime_seconds","mean")).reset_index().to_csv(p.parent/"planner_summary.csv",index=False)
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--smoke",action="store_true");main(p.parse_args().smoke)
