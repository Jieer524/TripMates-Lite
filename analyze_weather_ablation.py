"""Paired statistical analysis for the frozen thermal-ablation results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon


INPUT_FILE = Path("results/weather_ablation/weather_ablation_raw_90_pairs.csv")
OUTPUT_FILE = Path("results/weather_ablation/weather_ablation_statistical_tests.csv")
SUMMARY_FILE = Path("results/weather_ablation/weather_ablation_statistical_summary.json")
EXPECTED_PAIR_COUNT = 90
ALPHA = 0.05

METRICS = {
    "weather_weighted_exposure": (
        "adaptive_wes",
        "tripmates_wes",
        "lower",
    ),
    "achieved_utility": (
        "adaptive_utility",
        "tripmates_utility",
        "higher",
    ),
    "completed_visits": (
        "adaptive_completed_visits",
        "tripmates_completed_visits",
        "higher",
    ),
    "travel_time": (
        "adaptive_travel_min",
        "tripmates_travel_min",
        "lower",
    ),
}


def paired_advantage(
    frame: pd.DataFrame,
    baseline_column: str,
    comparison_column: str,
    preferred_direction: str,
) -> pd.Series:
    """Return a signed value where positive always favours TripMates Lite."""

    if preferred_direction == "lower":
        return frame[baseline_column] - frame[comparison_column]
    return frame[comparison_column] - frame[baseline_column]


def rank_biserial_effect(differences: pd.Series) -> float:
    nonzero = differences[differences.abs() > 1e-12]
    if nonzero.empty:
        return 0.0

    ranks = nonzero.abs().rank(method="average")
    positive_rank_sum = float(ranks[nonzero > 0].sum())
    negative_rank_sum = float(ranks[nonzero < 0].sum())
    rank_total = positive_rank_sum + negative_rank_sum
    if rank_total == 0.0:
        return 0.0
    return (positive_rank_sum - negative_rank_sum) / rank_total


def analyse_group(
    frame: pd.DataFrame,
    *,
    scope: str,
    instance_id: str,
) -> list[dict]:
    rows: list[dict] = []
    for metric_name, (
        baseline_column,
        comparison_column,
        preferred_direction,
    ) in METRICS.items():
        differences = paired_advantage(
            frame,
            baseline_column,
            comparison_column,
            preferred_direction,
        ).astype(float)
        nonzero_count = int((differences.abs() > 1e-12).sum())

        if nonzero_count == 0:
            statistic = 0.0
            p_value = 1.0
            test_status = "all_paired_differences_zero"
        else:
            result = wilcoxon(
                differences,
                zero_method="wilcox",
                alternative="two-sided",
                method="auto",
            )
            statistic = float(result.statistic)
            p_value = float(result.pvalue)
            test_status = "computed"

        rows.append(
            {
                "scope": scope,
                "instance_id": instance_id,
                "metric": metric_name,
                "preferred_direction": preferred_direction,
                "pair_count": len(frame),
                "nonzero_pair_count": nonzero_count,
                "median_tripmates_advantage": float(differences.median()),
                "mean_tripmates_advantage": float(differences.mean()),
                "wilcoxon_statistic": statistic,
                "two_sided_p_value": p_value,
                "rank_biserial_effect": rank_biserial_effect(differences),
                "significant_at_0_05": bool(p_value < ALPHA),
                "test_status": test_status,
            }
        )
    return rows


def main() -> None:
    frame = pd.read_csv(INPUT_FILE)
    if len(frame) != EXPECTED_PAIR_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_PAIR_COUNT} paired rows, found {len(frame)}."
        )
    if frame[["seed", "instance_id"]].duplicated().any():
        raise RuntimeError("Duplicate seed-instance pairs exist in the raw results.")

    analysis_rows = analyse_group(
        frame,
        scope="overall",
        instance_id="ALL",
    )
    for instance_id, instance_frame in frame.groupby("instance_id", sort=True):
        analysis_rows.extend(
            analyse_group(
                instance_frame,
                scope="instance",
                instance_id=str(instance_id),
            )
        )

    output = pd.DataFrame(analysis_rows)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False, float_format="%.9f")

    wes_overall = output[
        (output["scope"] == "overall")
        & (output["metric"] == "weather_weighted_exposure")
    ].iloc[0]
    summary = {
        "analysis": "paired thermal-ablation statistical tests",
        "pair_count": int(len(frame)),
        "instance_count": int(frame["instance_id"].nunique()),
        "seed_count": int(frame["seed"].nunique()),
        "alpha": ALPHA,
        "test": "two-sided Wilcoxon signed-rank",
        "effect_size": "matched-pairs rank-biserial correlation",
        "positive_effect_interpretation": "favours TripMates Lite",
        "overall_wes_p_value": float(wes_overall["two_sided_p_value"]),
        "overall_wes_effect": float(wes_overall["rank_biserial_effect"]),
        "overall_wes_status": str(wes_overall["test_status"]),
        "conclusion": (
            "No observed thermal advantage: every paired WES difference is zero."
            if int(wes_overall["nonzero_pair_count"]) == 0
            else "See the paired test table for the observed thermal effect."
        ),
    }
    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"Wrote {len(output)} paired test rows to {OUTPUT_FILE}.")
    print(summary["conclusion"])


if __name__ == "__main__":
    main()
