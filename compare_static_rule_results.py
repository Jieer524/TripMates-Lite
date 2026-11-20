from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent
OUT=ROOT/"output/research_final"
def main():
    legacy=pd.read_csv(ROOT/"results/rq1_full_benchmark/rq1_raw_runs.csv")
    corr=pd.read_csv(ROOT/"results/rq1_full_benchmark_corrected_static_rule/rq1_raw_runs.csv")
    rows=[]
    for name,d in [("legacy",legacy),("corrected",corr)]:
        s=d[d.planner.isin(["Static Distance","Static Utility"])]
        rows.append({"mode":name,"run_count":len(s),"feasible_count":int(s.feasible.astype(bool).sum()),"failure_count":int((~s.feasible.astype(bool)).sum()),"mean_completed_poi_count":s.completed_poi_count.mean(),"mean_achieved_utility":s.achieved_utility.mean(),"mean_total_travel_min":s.total_travel_min.mean(),"mean_wes":s.wes.mean(),"skipped_optional_poi_records":int(s.skipped_optional_pois.notna().sum()) if "skipped_optional_pois" in s else 0})
    f=pd.DataFrame(rows); f.to_csv(OUT/"static_rule_comparison.csv",index=False)
    cols=list(f.columns); lines=["| "+" | ".join(cols)+" |","| "+" | ".join("---" for _ in cols)+" |"]+["| "+" | ".join(str(row[c]) for c in cols)+" |" for _,row in f.iterrows()]
    (OUT/"static_rule_comparison.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f.to_string(index=False))
if __name__=='__main__': main()
