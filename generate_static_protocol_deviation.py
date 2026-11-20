"""Write the predeclared static-baseline protocol-deviation note."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent; OUT=ROOT/"output/research_final"
def main():
 d=pd.read_csv(ROOT/"results/rq1_full_benchmark/rq1_raw_runs.csv")
 s=d[d.planner.isin(["Static Distance","Static Utility"])].copy()
 reasons=s.loc[~s.feasible.astype(bool),"failure_reason"].fillna("")
 rows=[]
 for label,mask in [("unavailable disruption",reasons.str.contains("unavailable",case=False)),("trip-horizon violation",reasons.str.contains("17:00|trip limit",case=False,regex=True))]: rows.append({"category":label,"failure_count":int(mask.sum()),"interpretation":"Affected by replay-versus-skip protocol deviation; do not use alone for superiority claims." if label.startswith("unavailable") else "Hard horizon failure remains a genuine infeasibility under either protocol."})
 frame=pd.DataFrame(rows); frame.to_csv(OUT/"static_protocol_deviation.csv",index=False)
 text="# Static-baseline protocol deviation\n\nThe executed static evaluator replays the remaining suffix and stops at the first decoder violation. The planned skip-and-continue rule was not substituted after observing results because doing so would change the baseline execution protocol and require a separately specified rerun. Static failure rows remain in the raw benchmark. Unavailable-POI failures are affected by this deviation and should not be used alone as superiority evidence; trip-horizon failures remain hard infeasibilities.\n\n"+frame.to_csv(index=False)
 (OUT/"static_protocol_deviation.md").write_text(text,encoding="utf-8")
 print(frame.to_string(index=False))
if __name__=="__main__": main()
