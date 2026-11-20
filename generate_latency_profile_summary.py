from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent
OUT=ROOT/"output/research_final"
def main():
    rows=[{"rank":1,"function":"planning_core.decode_route","cumulative_seconds":"dominant; 236,206 calls in smoke profile","interpretation":"Shared decoder is the principal bottleneck."},{"rank":2,"function":"adaptive_ga_core.decode_adaptive_chromosome","cumulative_seconds":"11.885 post-cache profile","interpretation":"Insertion testing repeatedly invokes the shared decoder."},{"rank":3,"function":"tripmates_lite_ga.run_tripmates_lite_ga","cumulative_seconds":"8.611 post-cache profile","interpretation":"Thermal planner adds Pareto/exposure/selection work after shared decoding."},{"rank":4,"function":"adaptive_ga_core.run_adaptive_without_thermal_ga","cumulative_seconds":"7.984 post-cache profile","interpretation":"Research GA search budget dominates adaptive execution."}]
    d=pd.DataFrame(rows); d.to_csv(OUT/"latency_profile_summary.csv",index=False); cols=list(d.columns); lines=["| "+" | ".join(cols)+" |","| "+" | ".join("---" for _ in cols)+" |"]+["| "+" | ".join(str(row[c]) for c in cols)+" |" for _,row in d.iterrows()]; (OUT/"latency_profile_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
if __name__=='__main__':main()
