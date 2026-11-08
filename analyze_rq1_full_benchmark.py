"""
Descriptive and paired analysis for the RQ1 full benchmark.

The ten seeds are repeated stochastic optimisation runs, not
independent real-world disruptions.  This script therefore keeps
scenario-level and seed-level aggregation explicit:

- ``scope="overall_pairs"`` pools all matched (instance, seed) pairs
  for description only.
- ``scope="instance"`` reports each scenario instance separately.

Paired comparisons use a two-sided Wilcoxon signed-rank test and a
matched-pairs rank-biserial effect size.  Comparisons are only made
between planners that share the same initial route and seed.

Outputs:
- rq1_analysis_paired_tests.csv
- rq1_analysis_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon


OUTPUT_DIRECTORY = Path("results/rq1_full_benchmark")
INPUT_FILE = OUTPUT_DIRECTORY / "rq1_raw_runs.csv"
PAIRED_TESTS_FILE = (
    OUTPUT_DIRECTORY / "rq1_analysis_paired_tests.csv"
)
SUMMARY_FILE = (
    OUTPUT_DIRECTORY / "rq1_analysis_summary.json"
)

ALPHA = 0.05

PLANNER_STATIC_UTILITY = "Static Utility"
PLANNER_GREEDY = "Greedy Repair"
PLANNER_ADAPTIVE = "Adaptive Without Thermal"
PLANNER_TRIPMATES = "TripMates Lite"

# All of these share the seed-matched Static Utility initial route.
PAIRINGS = (
    (PLANNER_TRIPMATES, PLANNER_ADAPTIVE),
    (PLANNER_GREEDY, PLANNER_ADAPTIVE),
    (PLANNER_GREEDY, PLANNER_TRIPMATES),
    (PLANNER_STATIC_UTILITY, PLANNER_ADAPTIVE),
    (PLANNER_STATIC_UTILITY, PLANNER_TRIPMATES),
)

METRICS = {
    "wes": "lower",
    "achieved_utility": "higher",
    "completed_poi_count": "higher",
    "total_travel_min": "lower",
    "total_outdoor_exposure_min": "lower",
}


def rank_biserial_effect(
    differences: pd.Series,
) -> float:
    nonzero = differences[
        differences.abs() > 1e-12
    ]
    if nonzero.empty:
        return 0.0

    ranks = nonzero.abs().rank(method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    total = positive + negative
    if total == 0.0:
        return 0.0
    return (positive - negative) / total


def paired_test(
    frame: pd.DataFrame,
    *,
    first_planner: str,
    second_planner: str,
    metric: str,
    preferred_direction: str,
    scope: str,
    instance_id: str,
) -> dict | None:
    first = frame[
        frame["planner"] == first_planner
    ]
    second = frame[
        frame["planner"] == second_planner
    ]

    merged = first.merge(
        second,
        on=["instance_id", "seed"],
        suffixes=("_first", "_second"),
    )

    # Only compare pairs where both planners produced a metric.
    merged = merged.dropna(
        subset=[
            f"{metric}_first",
            f"{metric}_second",
        ]
    )

    if merged.empty:
        return None

    if preferred_direction == "lower":
        advantage = (
            merged[f"{metric}_first"]
            - merged[f"{metric}_second"]
        )
    else:
        advantage = (
            merged[f"{metric}_second"]
            - merged[f"{metric}_first"]
        )

    advantage = advantage.astype(float)
    nonzero = int((advantage.abs() > 1e-12).sum())

    if nonzero == 0:
        statistic = 0.0
        p_value = 1.0
        status = "all_paired_differences_zero"
    else:
        result = wilcoxon(
            advantage,
            zero_method="wilcox",
            alternative="two-sided",
            method="auto",
        )
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        status = "computed"

    return {
        "scope": scope,
        "instance_id": instance_id,
        "first_planner": first_planner,
        "second_planner": second_planner,
        "metric": metric,
        "preferred_direction": preferred_direction,
        "advantage_interpretation": (
            f"positive favours {second_planner}"
        ),
        "pair_count": int(len(merged)),
        "nonzero_pair_count": nonzero,
        "median_advantage": float(advantage.median()),
        "mean_advantage": float(advantage.mean()),
        "wilcoxon_statistic": statistic,
        "two_sided_p_value": p_value,
        "rank_biserial_effect": (
            rank_biserial_effect(advantage)
        ),
        "significant_at_0_05": bool(p_value < ALPHA),
        "test_status": status,
    }


def main() -> None:
    frame = pd.read_csv(INPUT_FILE)

    if frame.empty:
        raise RuntimeError(
            "The RQ1 raw-runs file is empty; run the benchmark first."
        )

    rows: list[dict] = []

    for (
        first_planner,
        second_planner,
    ) in PAIRINGS:
        pair_frame = frame[
            frame["planner"].isin(
                [first_planner, second_planner]
            )
        ]

        for metric, direction in METRICS.items():
            overall = paired_test(
                pair_frame,
                first_planner=first_planner,
                second_planner=second_planner,
                metric=metric,
                preferred_direction=direction,
                scope="overall_pairs",
                instance_id="ALL",
            )
            if overall is not None:
                rows.append(overall)

            for instance_id, _ in pair_frame.groupby(
                "instance_id", sort=True
            ):
                instance_row = paired_test(
                    pair_frame[
                        pair_frame["instance_id"]
                        == instance_id
                    ],
                    first_planner=first_planner,
                    second_planner=second_planner,
                    metric=metric,
                    preferred_direction=direction,
                    scope="instance",
                    instance_id=str(instance_id),
                )
                if instance_row is not None:
                    rows.append(instance_row)

    output = pd.DataFrame(rows)
    output.to_csv(
        PAIRED_TESTS_FILE,
        index=False,
        float_format="%.9f",
    )

    overall_wes = output[
        (output["scope"] == "overall_pairs")
        & (output["metric"] == "wes")
    ]

    summary = {
        "analysis": "RQ1 full benchmark paired analysis",
        "alpha": ALPHA,
        "test": "two-sided Wilcoxon signed-rank",
        "effect_size": (
            "matched-pairs rank-biserial correlation"
        ),
        "seed_interpretation": (
            "Seeds are repeated stochastic runs, not independent "
            "disruptions. Instance-level rows are the appropriate "
            "scenario-level unit; overall_pairs rows are descriptive."
        ),
        "pairings": [
            {
                "first_planner": first,
                "second_planner": second,
            }
            for first, second in PAIRINGS
        ],
        "overall_wes_tests": [
            {
                "first_planner": row["first_planner"],
                "second_planner": row["second_planner"],
                "pair_count": int(row["pair_count"]),
                "nonzero_pair_count": int(
                    row["nonzero_pair_count"]
                ),
                "median_advantage": float(
                    row["median_advantage"]
                ),
                "mean_advantage": float(
                    row["mean_advantage"]
                ),
                "two_sided_p_value": float(
                    row["two_sided_p_value"]
                ),
                "rank_biserial_effect": float(
                    row["rank_biserial_effect"]
                ),
                "test_status": str(row["test_status"]),
            }
            for _, row in overall_wes.iterrows()
        ],
        "pair_count_total": int(len(frame)),
        "instance_count": int(
            frame["instance_id"].nunique()
        ),
    }

    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(
        f"Wrote {len(output)} paired test rows to "
        f"{PAIRED_TESTS_FILE}."
    )
    print(
        f"Wrote analysis summary to {SUMMARY_FILE}."
    )


if __name__ == "__main__":
    main()
