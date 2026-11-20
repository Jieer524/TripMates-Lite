"""Summarise all preregistered thermal alpha levels without post-hoc selection."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent; OUT=ROOT/"output/research_final"
def main():
 d=pd.read_csv(ROOT/"results/thermal_tradeoff_sensitivity_alpha_independent_v2/thermal_tradeoff_raw_runs.csv")
 s=d.groupby("alpha",as_index=False).agg(median_wes_reduction_percent=("wes_reduction_percent","median"),median_utility_change=("utility_difference","median"),median_utility_loss_percent=("utility_loss_percent","median"),median_completed_visit_loss=("completed_visit_loss","median"),median_travel_change=("travel_change_min","median"),route_change_count=("route_changed","sum"),fallback_count=("guardrail_fallback_used","sum"),feasibility_rate=("feasible","mean"),run_count=("alpha","size"))
 s["wes_pass"]=s.median_wes_reduction_percent>=15; s["utility_pass"]=s.median_utility_loss_percent<=10; s["visit_pass"]=s.median_completed_visit_loss<=1; s["feasibility_pass"]=s.feasibility_rate>=1; s["all_criteria_pass"]=s[["wes_pass","utility_pass","visit_pass","feasibility_pass"]].all(axis=1)
 s.to_csv(OUT/"thermal_alpha_decision_table.csv",index=False)
 cols=list(s.columns); lines=["| "+" | ".join(cols)+" |","| "+" | ".join("---" for _ in cols)+" |"]; lines += ["| "+" | ".join(str(row[c]) for c in cols)+" |" for _,row in s.iterrows()]; (OUT/"thermal_alpha_decision_table.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
 print(s.to_string(index=False))
if __name__=="__main__": main()
