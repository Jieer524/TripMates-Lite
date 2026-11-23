"""Descriptive matched-pair analysis for the ILS adaptation comparison.

The 180 seed-matched pairs are computational repeats. Overall Wilcoxon rows
are descriptive; the six disruption families remain the interpretation units.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon


DIRECTORY = Path("results/rq1_ils_external_comparison")
INPUT = DIRECTORY / "rq1_with_ils_raw_runs.csv"
OUTPUT = DIRECTORY / "ils_paired_analysis.csv"
SUMMARY = DIRECTORY / "ils_analysis_summary.json"
ILS = "Single-route AILS adaptation"
ADAPTIVE = "Adaptive Without Thermal"
METRICS = {
    "itinerary_completion_ratio": "higher",
    "original_poi_retention": "higher",
    "achieved_utility": "higher",
    "total_travel_min": "lower",
    "wes": "lower",
    "edit_distance_score": "lower",
}


def effect(differences: pd.Series) -> float:
    nonzero = differences[differences.abs() > 1e-12]
    if nonzero.empty:
        return 0.0
    ranks = nonzero.abs().rank(method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def test_pair(frame: pd.DataFrame, metric: str, direction: str, scope: str, instance_id: str) -> dict:
    left = frame[frame.planner == ILS]
    right = frame[frame.planner == ADAPTIVE]
    paired = left.merge(right, on=["instance_id", "seed"], suffixes=("_ils", "_adaptive"))
    paired = paired.dropna(subset=[f"{metric}_ils", f"{metric}_adaptive"])
    if direction == "higher":
        difference = paired[f"{metric}_adaptive"] - paired[f"{metric}_ils"]
    else:
        difference = paired[f"{metric}_ils"] - paired[f"{metric}_adaptive"]
    difference = difference.astype(float)
    nonzero = int((difference.abs() > 1e-12).sum())
    if nonzero:
        statistic = wilcoxon(difference, zero_method="wilcox", alternative="two-sided", method="auto")
        value, p_value = float(statistic.statistic), float(statistic.pvalue)
        status = "computed"
    else:
        value, p_value, status = 0.0, 1.0, "all_paired_differences_zero"
    return {
        "scope": scope, "instance_id": instance_id, "metric": metric,
        "advantage_interpretation": "positive favours Adaptive Without Thermal",
        "pair_count": len(paired), "nonzero_pair_count": nonzero,
        "median_advantage": float(difference.median()),
        "mean_advantage": float(difference.mean()),
        "wilcoxon_statistic": value, "two_sided_p_value": p_value,
        "rank_biserial_effect": effect(difference), "test_status": status,
    }


def main() -> None:
    frame = pd.read_csv(INPUT)
    subset = frame[frame.planner.isin([ILS, ADAPTIVE])].copy()
    assert len(subset) == 360
    rows = []
    for metric, direction in METRICS.items():
        rows.append(test_pair(subset, metric, direction, "overall_pairs", "ALL"))
        for instance_id in sorted(subset.instance_id.unique()):
            rows.append(test_pair(subset[subset.instance_id == instance_id], metric, direction, "instance", instance_id))
    output = pd.DataFrame(rows)
    output.to_csv(OUTPUT, index=False, float_format="%.9f")
    overall = output[output.scope == "overall_pairs"]
    SUMMARY.write_text(json.dumps({
        "comparison": "Adapted single-route ILS family versus Adaptive Without Thermal",
        "test": "two-sided Wilcoxon signed-rank; descriptive for matched computational repeats",
        "interpretation_unit": "six disruption families; instance rows are early/nominal/late stress variants",
        "overall": overall.to_dict(orient="records"),
    }, indent=2), encoding="utf-8")
    print(f"Wrote {len(output)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
