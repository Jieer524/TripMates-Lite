"""Analysis for controlled gap studies: descriptive summaries and paired tests."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from ablation_common import STUDY_FULL_REPLANNING, STUDY_SEVERITY, STUDY_DENSITY, STUDY_WEATHER, study_files, benjamini_hochberg, rank_biserial_effect, wilcoxon_pair

METRICS = ("feasible", "achieved_utility", "retained_poi_count", "original_poi_retention", "wes", "total_travel_min", "changed_activity_count", "edit_distance_score", "runtime_seconds")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--study", required=True, choices=[STUDY_FULL_REPLANNING, STUDY_SEVERITY, STUDY_DENSITY, STUDY_WEATHER]); p.add_argument("--smoke", action="store_true"); a=p.parse_args()
    f=study_files(a.study, smoke=a.smoke); raw=pd.read_csv(f["raw_runs"])
    raw["feasible"] = raw["feasible"].astype(bool)
    group_cols=[c for c in ("factor_family","factor_level","factor_cell","family","level","cell_id","density_level","weather_level","replanning_scope","replanning_policy") if c in raw.columns]
    grouped = raw.groupby(group_cols+['planner'], dropna=False)
    summary = grouped[list(METRICS)].agg(['count', 'mean', 'median', 'std']).reset_index()
    summary.columns = ['_'.join(str(x) for x in c if x) if isinstance(c, tuple) else c for c in summary.columns]
    # Keep a compact, machine-readable descriptive table alongside the
    # expanded statistics; feasibility/failure are proportions.
    rates = grouped['feasible'].agg(feasibility_rate='mean', run_count='count').reset_index()
    rates['failure_rate'] = 1.0 - rates['feasibility_rate']
    summary = summary.merge(rates, on=group_cols + ['planner'], how='left')
    summary.to_csv(f["directory"] / "analysis_summary.csv", index=False)
    rows=[]
    if group_cols:
        for key,g in raw.groupby(group_cols, dropna=False):
            key=(key,) if not isinstance(key,tuple) else key
            for planner in sorted(g.planner.dropna().unique()):
                x=g[g.planner==planner].set_index(["instance_id","seed"])
                if len(x)==0: continue
                for metric in METRICS[1:]:
                    if metric not in x: continue
                    # Paired only where the same planner has two factor cells; otherwise descriptive.
                    for other_key,h in raw.groupby(group_cols, dropna=False):
                        other_key=(other_key,) if not isinstance(other_key,tuple) else other_key
                        if other_key==key: continue
                        y=h[h.planner==planner].set_index(["instance_id","seed"])
                        joined=x[[metric]].join(y[[metric]], lsuffix='_a', rsuffix='_b', how='inner').dropna()
                        if joined.empty: continue
                        stat,pval,status=wilcoxon_pair(joined[f'{metric}_b']-joined[f'{metric}_a'])
                        rows.append(dict(zip(group_cols,key), planner=planner, metric=metric, comparison=f'{key} vs {other_key}', pairs=len(joined), statistic=stat, p_value=pval, effect_rank_biserial=rank_biserial_effect(joined[f'{metric}_b']-joined[f'{metric}_a']), status=status))
    tests=pd.DataFrame(rows)
    if not tests.empty: tests['q_value']=benjamini_hochberg(tests.p_value.tolist())
    tests.to_csv(f["directory"] / "paired_tests.csv", index=False)
    tests.to_csv(f["directory"] / "paired_statistical_tests.csv", index=False)
    (f["directory"] / "analysis_summary.json").write_text(json.dumps({'classification':'exploratory_sensitivity','descriptive_only':True,'metric_columns':list(METRICS),'paired_tests':len(tests)}, indent=2), encoding='utf-8')
    print(summary.to_string(index=False)); print(f"Analysis written to {f['directory'].resolve()}")
if __name__=='__main__': main()
