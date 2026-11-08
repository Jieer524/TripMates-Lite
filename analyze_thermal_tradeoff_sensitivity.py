"""
Paired statistical analysis for the C2 thermal trade-off sensitivity
experiment.

This is an EXPLORATORY / sensitivity analysis.  It does not replace the
frozen confirmatory 90-pair thermal ablation and must not be presented
as such.

For each utility-retention threshold ``alpha`` and for the pooled set
across alphas, it computes paired differences

    Adaptive Without Thermal WES - TripMates Lite WES

so that positive values favour TripMates Lite.  It uses a two-sided
Wilcoxon signed-rank test when non-zero paired differences exist,
handles the all-zero case without error, reports a matched-pairs
rank-biserial effect size, and applies a Holm multiple-comparison
adjustment across the four alpha-level primary tests.

Outputs:
- thermal_tradeoff_statistical_tests.csv
- thermal_tradeoff_statistical_summary.json

Usage:
    python analyze_thermal_tradeoff_sensitivity.py
    python analyze_thermal_tradeoff_sensitivity.py \
        --directory results/thermal_tradeoff_sensitivity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon


DEFAULT_OUTPUT_DIRECTORY = Path(
    "results/thermal_tradeoff_sensitivity_alpha_independent_v2"
)

SEARCH_POLICY_MODE = "alpha_independent_frozen_policy"
FINAL_SELECTION_POLICY_MODE = "alpha_sensitive_exploratory"
SUPERSEDES_DIRECTORY = (
    "results/thermal_tradeoff_sensitivity/"
)
IS_CONFIRMATORY = False

RAW_RUNS_NAME = "thermal_tradeoff_raw_runs.csv"
TESTS_NAME = "thermal_tradeoff_statistical_tests.csv"
SUMMARY_NAME = (
    "thermal_tradeoff_statistical_summary.json"
)

OUTPUT_DIRECTORY = DEFAULT_OUTPUT_DIRECTORY
INPUT_FILE = OUTPUT_DIRECTORY / RAW_RUNS_NAME
OUTPUT_FILE = OUTPUT_DIRECTORY / TESTS_NAME
SUMMARY_FILE = OUTPUT_DIRECTORY / SUMMARY_NAME

ALPHAS = (1.00, 0.95, 0.90, 0.85)
SIGNIFICANCE_LEVEL = 0.05

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

PRIMARY_METRIC = "weather_weighted_exposure"


def paired_advantage(
    frame: pd.DataFrame,
    baseline_column: str,
    comparison_column: str,
    preferred_direction: str,
) -> pd.Series:
    """Signed value where positive always favours TripMates Lite."""

    if preferred_direction == "lower":
        return (
            frame[baseline_column]
            - frame[comparison_column]
        )
    return (
        frame[comparison_column]
        - frame[baseline_column]
    )


def rank_biserial_effect(
    differences: pd.Series,
) -> float:
    nonzero = differences[
        differences.abs() > 1e-12
    ]

    if nonzero.empty:
        return 0.0

    ranks = nonzero.abs().rank(method="average")
    positive_rank_sum = float(
        ranks[nonzero > 0].sum()
    )
    negative_rank_sum = float(
        ranks[nonzero < 0].sum()
    )
    rank_total = (
        positive_rank_sum + negative_rank_sum
    )

    if rank_total == 0.0:
        return 0.0

    return (
        positive_rank_sum - negative_rank_sum
    ) / rank_total


def analyse_group(
    frame: pd.DataFrame,
    *,
    scope: str,
    alpha,
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

        nonzero_count = int(
            (differences.abs() > 1e-12).sum()
        )

        if nonzero_count == 0:
            statistic = 0.0
            p_value = 1.0
            test_status = (
                "all_paired_differences_zero"
            )
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
                "alpha": alpha,
                "instance_id": instance_id,
                "metric": metric_name,
                "preferred_direction": (
                    preferred_direction
                ),
                "pair_count": len(frame),
                "nonzero_pair_count": nonzero_count,
                "median_tripmates_advantage": float(
                    differences.median()
                ),
                "mean_tripmates_advantage": float(
                    differences.mean()
                ),
                "wilcoxon_statistic": statistic,
                "two_sided_p_value": p_value,
                "rank_biserial_effect": (
                    rank_biserial_effect(
                        differences
                    )
                ),
                "significant_at_0_05": bool(
                    p_value < SIGNIFICANCE_LEVEL
                ),
                "test_status": test_status,
            }
        )

    return rows


def holm_adjust(
    p_values: list[float],
) -> list[float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""

    count = len(p_values)

    if count == 0:
        return []

    order = sorted(
        range(count),
        key=lambda index: p_values[index],
    )

    adjusted = [0.0] * count

    running_max = 0.0

    for rank, index in enumerate(order):
        factor = count - rank
        value = min(
            1.0,
            p_values[index] * factor,
        )
        running_max = max(running_max, value)
        adjusted[index] = running_max

    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        default=str(DEFAULT_OUTPUT_DIRECTORY),
        help=(
            "Directory holding thermal_tradeoff_raw_runs.csv "
            "(defaults to the alpha-independent V2 run)."
        ),
    )
    args = parser.parse_args()

    output_directory = Path(args.directory)
    input_file = output_directory / RAW_RUNS_NAME
    output_file = output_directory / TESTS_NAME
    summary_file = output_directory / SUMMARY_NAME

    frame = pd.read_csv(input_file)

    if frame.empty:
        raise RuntimeError(
            "The C2 raw-runs file is empty; run the experiment first."
        )

    if frame[
        ["seed", "instance_id", "alpha"]
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate seed-instance-alpha pairs exist in the "
            "raw results."
        )

    analysis_rows: list[dict] = []

    # ------------------------------------------------------------
    # Per-alpha overall (across instances and seeds)
    # ------------------------------------------------------------

    for alpha in ALPHAS:
        alpha_frame = frame[frame["alpha"] == alpha]

        if alpha_frame.empty:
            continue

        analysis_rows.extend(
            analyse_group(
                alpha_frame,
                scope="alpha_overall",
                alpha=alpha,
                instance_id="ALL",
            )
        )

    # ------------------------------------------------------------
    # Pooled across alphas
    # ------------------------------------------------------------

    analysis_rows.extend(
        analyse_group(
            frame,
            scope="pooled_all_alphas",
            alpha="ALL",
            instance_id="ALL",
        )
    )

    # ------------------------------------------------------------
    # Per instance, per alpha
    # ------------------------------------------------------------

    for alpha in ALPHAS:
        alpha_frame = frame[frame["alpha"] == alpha]

        for instance_id, group in alpha_frame.groupby(
            "instance_id", sort=True
        ):
            analysis_rows.extend(
                analyse_group(
                    group,
                    scope="instance",
                    alpha=alpha,
                    instance_id=str(instance_id),
                )
            )

    output = pd.DataFrame(analysis_rows)

    # ------------------------------------------------------------
    # Holm adjustment across the four alpha-level primary tests
    # ------------------------------------------------------------

    primary_mask = (
        (output["scope"] == "alpha_overall")
        & (output["metric"] == PRIMARY_METRIC)
    )

    primary_indices = output.index[primary_mask].tolist()

    raw_p_values = output.loc[
        primary_indices, "two_sided_p_value"
    ].tolist()

    adjusted_p_values = holm_adjust(raw_p_values)

    output["holm_adjusted_p_value"] = None

    for index, adjusted in zip(
        primary_indices, adjusted_p_values
    ):
        output.at[index, "holm_adjusted_p_value"] = (
            adjusted
        )

    output["multiple_comparison_policy"] = (
        "Holm step-down across the four alpha-level primary "
        "WES tests; instance-level tests are descriptive only."
    )

    output.to_csv(
        output_file, index=False, float_format="%.9f"
    )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    alpha_overall_wes = output[
        (output["scope"] == "alpha_overall")
        & (output["metric"] == PRIMARY_METRIC)
    ]

    per_alpha_summary = []

    for _, row in alpha_overall_wes.iterrows():
        per_alpha_summary.append(
            {
                "alpha": float(row["alpha"]),
                "pair_count": int(row["pair_count"]),
                "nonzero_pair_count": int(
                    row["nonzero_pair_count"]
                ),
                "median_tripmates_wes_advantage": float(
                    row["median_tripmates_advantage"]
                ),
                "mean_tripmates_wes_advantage": float(
                    row["mean_tripmates_advantage"]
                ),
                "wilcoxon_statistic": float(
                    row["wilcoxon_statistic"]
                ),
                "raw_p_value": float(
                    row["two_sided_p_value"]
                ),
                "holm_adjusted_p_value": (
                    None
                    if pd.isna(
                        row["holm_adjusted_p_value"]
                    )
                    else float(
                        row["holm_adjusted_p_value"]
                    )
                ),
                "rank_biserial_effect": float(
                    row["rank_biserial_effect"]
                ),
                "test_status": str(row["test_status"]),
            }
        )

    pooled_wes = output[
        (output["scope"] == "pooled_all_alphas")
        & (output["metric"] == PRIMARY_METRIC)
    ].iloc[0]

    any_significant = bool(
        alpha_overall_wes[
            "significant_at_0_05"
        ].any()
    )

    summary = {
        "analysis": (
            "C2 thermal trade-off sensitivity - paired tests"
        ),
        "classification": (
            "exploratory sensitivity analysis"
        ),
        "is_confirmatory": False,
        "replaces_original_result": False,

        # Alpha-independence provenance of the analysed run.
        "search_policy_mode": SEARCH_POLICY_MODE,
        "final_selection_policy_mode": (
            FINAL_SELECTION_POLICY_MODE
        ),
        "supersedes": SUPERSEDES_DIRECTORY,
        "input_directory": output_directory.as_posix(),
        "alpha_independence_note": (
            "The analysed run pins every internal GA decision to one "
            "frozen policy, so only the final route selection depends "
            "on alpha. Paired differences therefore reflect the "
            "final-selection policy alone."
        ),

        "original_result_note": (
            "The frozen 90-pair thermal ablation remains the "
            "confirmatory fixed-policy result and is not replaced "
            "by this sensitivity analysis."
        ),

        "pair_count": int(len(frame)),
        "instance_count": int(
            frame["instance_id"].nunique()
        ),
        "seed_count": int(frame["seed"].nunique()),
        "alphas": list(ALPHAS),

        "alpha": SIGNIFICANCE_LEVEL,
        "test": "two-sided Wilcoxon signed-rank",
        "effect_size": (
            "matched-pairs rank-biserial correlation"
        ),
        "positive_effect_interpretation": (
            "favours TripMates Lite (lower WES)"
        ),
        "multiple_comparison_policy": (
            "Holm step-down across the four alpha-level primary "
            "WES tests"
        ),

        "per_alpha_wes_tests": per_alpha_summary,

        "pooled_wes_p_value": float(
            pooled_wes["two_sided_p_value"]
        ),
        "pooled_wes_effect": float(
            pooled_wes["rank_biserial_effect"]
        ),
        "pooled_wes_status": str(
            pooled_wes["test_status"]
        ),

        "any_alpha_significant_at_0_05": (
            any_significant
        ),
        "conclusion": (
            "No alpha-level WES difference is statistically "
            "significant at 0.05 after Holm adjustment; the "
            "trade-off is descriptive only."
            if not any_significant
            else "At least one alpha-level WES difference is "
            "significant at 0.05 before/after adjustment; see the "
            "per-alpha table."
        ),
    }

    summary_file.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(
        f"Wrote {len(output)} paired test rows to "
        f"{output_file}."
    )
    print(summary["conclusion"])


if __name__ == "__main__":
    main()
