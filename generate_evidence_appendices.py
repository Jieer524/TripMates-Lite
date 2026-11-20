"""Generate transparent evidence appendices from completed experiment artefacts."""
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parent; OUT=ROOT/"output/research_final"

def md_table(df, path):
    cols=list(df.columns); lines=["| "+" | ".join(cols)+" |","| "+" | ".join("---" for _ in cols)+" |"]
    lines += ["| "+" | ".join(str(row[c]).replace("|","\\|") for c in cols)+" |" for _,row in df.iterrows()]
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    sources={
      "rq1":ROOT/"results/rq1_full_benchmark_corrected_static_rule/rq1_raw_runs.csv",
      "full_replanning":ROOT/"results/ablation_studies/full_replanning_ablation/raw_runs.csv",
      "stability":ROOT/"results/ablation_studies/stability_ablation/raw_runs.csv",
      "candidate_window":ROOT/"results/ablation_studies/candidate_window_sensitivity/raw_runs.csv",
      "utility_weights":ROOT/"results/sensitivity/utility_weights/raw_runs.csv",
      "thermal":ROOT/"results/thermal_tradeoff_sensitivity_alpha_independent_v2/thermal_tradeoff_raw_runs.csv",
    }
    frames=[]
    for exp,path in sources.items():
        if path.exists():
            d=pd.read_csv(path); d["experiment"]=exp; frames.append(d)
    allf=pd.concat(frames,ignore_index=True,sort=False)
    group_cols=[c for c in ["experiment","planner","scenario_id","trigger_variant","seed","utility_weight_condition","candidate_window_condition","replanning_scope","failure_reason"] if c in allf]
    failures=allf[allf.feasible.astype(bool).eq(False)].copy()
    cat=failures.groupby(group_cols,dropna=False).size().reset_index(name="failure_count") if not failures.empty else pd.DataFrame(columns=group_cols+["failure_count"])
    cat.to_csv(OUT/"failure_categories.csv",index=False); md_table(cat.head(200),OUT/"failure_categories.md")
    rate=allf.groupby(["experiment"]+(["planner"] if "planner" in allf else []),dropna=False).agg(run_count=("experiment","size"),failure_count=("feasible",lambda x:int((~x.astype(bool)).sum())),failure_rate=("feasible",lambda x:float((~x.astype(bool)).mean()))).reset_index()
    rate.to_csv(OUT/"failure_rate_summary.csv",index=False); md_table(rate,OUT/"failure_rate_summary.md")
    # Representative route audit: deterministic, one row per requested evidence family.
    picks=[]
    selectors=[("feasible_static","rq1",lambda d:d.planner.eq("Static Utility")&d.feasible.astype(bool)),("static_failure","rq1",lambda d:d.planner.eq("Static Utility")&~d.feasible.astype(bool)),("bounded_adaptive","rq1",lambda d:d.planner.eq("Adaptive Without Thermal")&d.feasible.astype(bool)),("full_replanning","full_replanning",lambda d:d.feasible.astype(bool)),("no_stability","stability",lambda d:d.feasible.astype(bool)),("candidate_window","candidate_window",lambda d:d.feasible.astype(bool)),("utility_failure","utility_weights",lambda d:~d.feasible.astype(bool)),]
    for label,exp,fn in selectors:
        d=allf[allf.experiment.eq(exp)]; q=d[fn(d)]
        if not q.empty:
            row=q.iloc[0].to_dict(); row["audit_case"]=label; picks.append(row)
    thermal=pd.read_csv(sources["thermal"]); row=thermal[thermal.alpha.eq(0.90)].iloc[0].to_dict(); row["audit_case"]="thermal_alpha_0.90"; picks.append(row)
    audit=pd.DataFrame(picks); audit.to_json(OUT/"route_audit.json",orient="records",indent=2); audit.to_csv(OUT/"route_audit.csv",index=False)
    md_table(audit[[c for c in ["audit_case","experiment","planner","instance_id","scenario_id","trigger_variant","seed","initial_route","completed_prefix","original_remaining_route","repaired_route","final_route","feasible","failure_reason","removed_poi_count","added_poi_count","retained_poi_count","total_travel_min","total_cost_rm","wes","candidate_window_size","utility_weight_condition"] if c in audit]],OUT/"route_audit.md")
    utility=pd.read_csv(ROOT/"poi_data_calculated_final.csv")
    util_report=pd.DataFrame([{"field":"rating","present": "google_rating" in utility.columns,"source_column":"google_rating" if "google_rating" in utility.columns else None,"used_by_planning_data":False},{"field":"popularity/review count","present":"review_count" in utility.columns,"source_column":"review_count" if "review_count" in utility.columns else None,"used_by_planning_data":False},{"field":"zone prominence","present":"zone_score" in utility.columns,"source_column":"zone_score" if "zone_score" in utility.columns else None,"used_by_planning_data":False},{"field":"implemented utility","present":True,"source_column":"utility_score","used_by_planning_data":True}])
    util_report.to_csv(OUT/"utility_data_limitation.csv",index=False); md_table(util_report,OUT/"utility_data_limitation.md")
    limitations=pd.DataFrame([
      ["Static-baseline execution-rule asymmetry","RQ1 feasibility/recovery","High","Original result retained; deviation reported","Rerun with predeclared skip-and-continue static rule"],["50-POI catalogue","All generalisation claims","High","Fixed input hashes and explicit scope","Evaluate larger multi-city catalogue"],["Synthetic disruptions","RO4 sensitivity","Medium","Scenario definitions and factors are frozen","Validate against observed disruptions"],["Composite utility score","Utility sensitivity","High","Perturbation is explicitly composite, not causal","Expose source components and rerun"],["WES proxy validity","RQ2","High","Planning proxy documented; no medical claim","Validate exposure model externally"],["Full replanning upper bound","Full-replanning ablation","Medium","Clearly labelled infeasible upper bound","Compare only deployable policies"],["Repeated seeds are not independent disruptions","Statistical inference","Medium","Instance-level interpretation","Use independent disruption samples"],["Latency threshold failure","RQ3","High","Research and quick-preview settings reported separately","Profile and optimise decoder/GA"],["No operator-level GA tests","Implementation validity","Medium","Contract and integration tests retained","Add operator property tests"],["Candidate-window/adaptivity confounding","Sensitivity interpretation","Medium","Matched planners and fixed modification limit","Use factorial design"],["Retained failures","All sensitivities","Medium","Raw rows and failure tables retained","Investigate condition-specific causes"],
    ],columns=["Limitation","Affected result","Severity","Current mitigation","Future improvement"])
    limitations.to_csv(OUT/"validity_limitations.csv",index=False); md_table(limitations,OUT/"validity_limitations.md")
    print(f"Generated evidence appendices from {len(allf)} rows; failures={len(failures)}")
if __name__=="__main__": main()
