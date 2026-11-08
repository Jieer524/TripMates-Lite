"""
C2 thermal trade-off sensitivity experiment.

Purpose
-------
Quantify the utility-exposure trade-off under predefined
utility-retention policies.  This is an exploratory sensitivity
analysis that accompanies the frozen 90-pair thermal ablation; it does
not replace it and does not attempt to "make thermal win".

For every (weather scenario instance, seed, alpha) pair it runs:

- Adaptive Without Thermal (the alpha-independent baseline), and
- TripMates Lite with the C2 utility-retention threshold ``alpha``.

The C2 policy uses:

- ``U*`` = the highest achieved utility among the feasible
  alternatives considered for selection (first-front maximum and the
  seed-matched non-thermal reference, whichever is larger);
- utility guardrail: ``route.utility >= alpha * U*``;
- visit guardrail: ``route.completed_poi_count >= reference.count - 1``
  (margin configurable, recorded explicitly);
- selection priority: lowest WES, then lower travel, then higher
  utility, then lower POI-ID sequence;
- structured fallback to the non-thermal reference when no first-front
  alternative passes both guardrails.

Alpha-independence (V2)
-----------------------
``alpha`` affects ONLY the final route selection.  Every internal GA
decision - the generation-zero progress reference, the in-loop
stagnation comparison, the convergence generation and the evaluated
Pareto front - uses one explicit frozen default policy, so the search
trajectory is identical across the four alphas.  Only the returned
route, its guardrail status and its fallback flag may vary.

This V2 run supersedes the original ``results/thermal_tradeoff_sensitivity/``
run, whose search was unintentionally alpha-dependent.  It remains
exploratory and does not replace the frozen confirmatory 90-pair
thermal ablation.

Outputs (in
results/thermal_tradeoff_sensitivity_alpha_independent_v2/):
- thermal_tradeoff_raw_runs.csv
- thermal_tradeoff_pareto_fronts.csv
- thermal_tradeoff_instance_summary.csv
- thermal_tradeoff_run_config.json
- chart-ready CSV files

Statistical tests are produced separately by
analyze_thermal_tradeoff_sensitivity.py.

Usage:
    python run_thermal_tradeoff_sensitivity.py            # full run
    python run_thermal_tradeoff_sensitivity.py --smoke     # smoke only
"""

from __future__ import annotations

from dataclasses import asdict
import argparse
import json
import statistics
import time
from pathlib import Path

import pandas as pd

from adaptive_ga_core import (
    REPLANNING_GA_CONFIG,
    prepare_adaptive_problem,
    run_adaptive_without_thermal_ga,
)
from benchmark_common import (
    C2_ALPHAS,
    POI_FILE,
    MATRIX_FILE,
    SEEDS,
    WEATHER_INSTANCE_IDS,
    build_initial_schedules,
    build_scenario_lookup,
    encode_ids,
    file_fingerprint,
    load_initial_routes,
    runtime_metadata,
    utc_timestamp,
    write_json,
)
from exposure_core import calculate_exposure_metrics
from planning_core import load_planning_data
from scenario_core import simulate_replanning_state
from stability_metrics import (
    StabilityWeightConfig,
    compute_stability_from_state,
)
from tripmates_lite_ga import (
    DEFAULT_THERMAL_SELECTION_POLICY,
    ThermalSelectionPolicy,
    run_tripmates_lite_ga,
)


OUTPUT_DIRECTORY = Path(
    "results/thermal_tradeoff_sensitivity_alpha_independent_v2"
)

# The V2 run supersedes the original alpha-dependent C2 run.  Its
# search is pinned to one explicit frozen policy while the final
# selection stays alpha-sensitive.
SEARCH_POLICY_MODE = "alpha_independent_frozen_policy"
FINAL_SELECTION_POLICY_MODE = "alpha_sensitive_exploratory"
SUPERSEDES_DIRECTORY = (
    "results/thermal_tradeoff_sensitivity/"
)
IS_CONFIRMATORY = False

RAW_RUNS_FILE = (
    OUTPUT_DIRECTORY / "thermal_tradeoff_raw_runs.csv"
)
PARETO_FRONTS_FILE = (
    OUTPUT_DIRECTORY / "thermal_tradeoff_pareto_fronts.csv"
)
INSTANCE_SUMMARY_FILE = (
    OUTPUT_DIRECTORY
    / "thermal_tradeoff_instance_summary.csv"
)
RUN_CONFIG_FILE = (
    OUTPUT_DIRECTORY / "thermal_tradeoff_run_config.json"
)
CHART_WES_FILE = (
    OUTPUT_DIRECTORY
    / "thermal_tradeoff_wes_by_alpha.csv"
)
CHART_TRADEOFF_FILE = (
    OUTPUT_DIRECTORY
    / "thermal_tradeoff_utility_exposure_by_alpha.csv"
)


def isolate_smoke_outputs() -> None:
    """Redirect a ``--smoke`` run into an isolated sub-folder.

    Smoke runs must never overwrite a completed full benchmark, so a
    ``--smoke`` invocation writes every artefact under
    ``<output-directory>/smoke/`` instead of the shared full-run paths.
    """
    global OUTPUT_DIRECTORY
    global RAW_RUNS_FILE, PARETO_FRONTS_FILE, INSTANCE_SUMMARY_FILE
    global RUN_CONFIG_FILE, CHART_WES_FILE, CHART_TRADEOFF_FILE

    smoke_directory = OUTPUT_DIRECTORY / "smoke"

    RAW_RUNS_FILE = smoke_directory / RAW_RUNS_FILE.name
    PARETO_FRONTS_FILE = (
        smoke_directory / PARETO_FRONTS_FILE.name
    )
    INSTANCE_SUMMARY_FILE = (
        smoke_directory / INSTANCE_SUMMARY_FILE.name
    )
    RUN_CONFIG_FILE = smoke_directory / RUN_CONFIG_FILE.name
    CHART_WES_FILE = smoke_directory / CHART_WES_FILE.name
    CHART_TRADEOFF_FILE = (
        smoke_directory / CHART_TRADEOFF_FILE.name
    )

    OUTPUT_DIRECTORY = smoke_directory
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)


EXPERIMENT_NAME = "C2 thermal trade-off sensitivity"

# The non-thermal baseline is alpha-independent and deterministic, so
# it is computed once per (instance, seed) and reused across alphas.
# This is an efficiency measure only; it does not change any value.
REUSE_NON_THERMAL_BASELINE = True

C2_POLICY_TEMPLATE = {
    "visit_guardrail_margin": 1,
    "utility_guardrail_baseline": (
        "front_maximum_plus_reference"
    ),
    "priority": "c2",
}


def build_policy(alpha: float) -> ThermalSelectionPolicy:
    return ThermalSelectionPolicy(
        alpha=float(alpha),
        visit_guardrail_margin=(
            C2_POLICY_TEMPLATE[
                "visit_guardrail_margin"
            ]
        ),
        utility_guardrail_baseline=(
            C2_POLICY_TEMPLATE[
                "utility_guardrail_baseline"
            ]
        ),
        priority=C2_POLICY_TEMPLATE["priority"],
    )


def reduction_percent(
    reference_value: float,
    comparison_value: float,
) -> float:
    if abs(reference_value) <= 1e-12:
        return 0.0
    return (
        (reference_value - comparison_value)
        / reference_value
        * 100.0
    )


def build_front_rows(
    *,
    result,
    instance_id: str,
    scenario_instance,
    seed: int,
    alpha: float,
) -> list[dict]:
    rows: list[dict] = []

    front = result.first_front
    front_size = len(front)

    for index, solution in enumerate(front):
        rows.append(
            {
                "instance_id": instance_id,
                "scenario_id": (
                    scenario_instance.scenario.scenario_id
                ),
                "scenario_name": (
                    scenario_instance.scenario.name
                ),
                "trigger_variant": (
                    scenario_instance.trigger_variant
                ),
                "seed": seed,
                "alpha": alpha,
                "planner": "TripMates Lite",
                "front_size": front_size,
                "front_index": index,
                "route": encode_ids(solution.route),
                "achieved_utility": (
                    solution.achieved_utility
                ),
                "completed_poi_count": (
                    solution.completed_poi_count
                ),
                "total_travel_min": (
                    solution.total_travel_min
                ),
                "wes": solution.wes,
                "total_outdoor_exposure_min": (
                    solution.exposure
                    .total_outdoor_exposure_min
                ),
                "utility_guardrail_passed": bool(
                    result.front_utility_guardrail[index]
                ),
                "visit_guardrail_passed": bool(
                    result.front_visit_guardrail[index]
                ),
                "guardrail_eligible": bool(
                    result.front_guardrail_eligible[index]
                ),
                "selected": bool(
                    result.front_selected[index]
                ),
                "fallback_used": bool(
                    result.guardrail_fallback_used
                ),
                "feasible": True,
                "u_star": result.u_star,
                "reference_utility": (
                    result.reference_utility
                ),
                "front_max_utility": (
                    result.front_max_utility
                ),
                "utility_retention_ratio": (
                    solution.achieved_utility
                    / result.u_star
                    if result.u_star > 1e-12
                    else None
                ),
            }
        )

    return rows


def build_instance_summary(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    summary_rows: list[dict] = []

    for instance_id in WEATHER_INSTANCE_IDS:
        for alpha in C2_ALPHAS:
            group = raw[
                (raw["instance_id"] == instance_id)
                & (raw["alpha"] == alpha)
            ]

            if group.empty:
                continue

            wes_difference = group["wes_difference"]
            utility_difference = group[
                "utility_difference"
            ]
            visit_difference = group[
                "completed_visit_loss"
            ]
            travel_difference = group["travel_change_min"]

            summary_rows.append(
                {
                    "instance_id": instance_id,
                    "scenario_id": group[
                        "scenario_id"
                    ].iloc[0],
                    "scenario_name": group[
                        "scenario_name"
                    ].iloc[0],
                    "trigger_variant": group[
                        "trigger_variant"
                    ].iloc[0],
                    "alpha": alpha,
                    "thermal_multiplier": group[
                        "thermal_multiplier"
                    ].iloc[0],

                    "paired_run_count": int(len(group)),
                    "feasible_run_count": int(
                        group["feasible"].sum()
                    ),

                    "route_change_count": int(
                        group["route_changed"].sum()
                    ),
                    "route_change_percent": float(
                        group["route_changed"].mean()
                        * 100.0
                    ),

                    # ------------------------------------------
                    # Itinerary stability
                    # ------------------------------------------

                    "median_adaptive_edit_distance": float(
                        group[
                            "adaptive_edit_distance_score"
                        ].median()
                    ),
                    "median_tripmates_edit_distance": float(
                        group[
                            "tripmates_edit_distance_score"
                        ].median()
                    ),
                    "median_edit_distance_difference": float(
                        group[
                            "edit_distance_difference"
                        ].median()
                    ),
                    "adaptive_locked_prefix_violation_count": int(
                        group[
                            "adaptive_locked_prefix_changed"
                        ].sum()
                    ),
                    "tripmates_locked_prefix_violation_count": int(
                        group[
                            "tripmates_locked_prefix_changed"
                        ].sum()
                    ),

                    "median_wes_difference": float(
                        wes_difference.median()
                    ),
                    "mean_wes_difference": float(
                        wes_difference.mean()
                    ),
                    "median_wes_reduction_percent": float(
                        group[
                            "wes_reduction_percent"
                        ].median()
                    ),
                    "mean_wes_reduction_percent": float(
                        group[
                            "wes_reduction_percent"
                        ].mean()
                    ),

                    "median_utility_difference": float(
                        utility_difference.median()
                    ),
                    "mean_utility_difference": float(
                        utility_difference.mean()
                    ),
                    "median_utility_loss_percent": float(
                        group[
                            "utility_loss_percent"
                        ].median()
                    ),

                    "median_completed_visit_difference": float(
                        visit_difference.median()
                    ),
                    "median_travel_difference_min": float(
                        travel_difference.median()
                    ),

                    "utility_guardrail_pass_count": int(
                        group[
                            "utility_guardrail_passed"
                        ].sum()
                    ),
                    "visit_guardrail_pass_count": int(
                        group[
                            "visit_guardrail_passed"
                        ].sum()
                    ),
                    "fallback_count": int(
                        group[
                            "guardrail_fallback_used"
                        ].sum()
                    ),

                    "median_pareto_front_size": float(
                        group["pareto_front_size"].median()
                    ),
                    "mean_pareto_front_size": float(
                        group["pareto_front_size"].mean()
                    ),
                    "minimum_pareto_front_size": int(
                        group["pareto_front_size"].min()
                    ),
                    "maximum_pareto_front_size": int(
                        group["pareto_front_size"].max()
                    ),

                    "seeds_lower_wes": int(
                        (wes_difference > 1e-9).sum()
                    ),
                    "seeds_same_wes": int(
                        (wes_difference.abs() <= 1e-9).sum()
                    ),
                    "seeds_higher_wes": int(
                        (wes_difference < -1e-9).sum()
                    ),
                }
            )

    return pd.DataFrame(summary_rows)


def build_chart_tables(
    summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    wes_chart = summary[
        [
            "instance_id",
            "alpha",
            "median_wes_difference",
            "median_wes_reduction_percent",
            "seeds_lower_wes",
            "seeds_same_wes",
            "seeds_higher_wes",
        ]
    ].copy()

    tradeoff_chart = summary[
        [
            "instance_id",
            "alpha",
            "median_utility_loss_percent",
            "median_wes_reduction_percent",
            "route_change_count",
            "fallback_count",
            "median_pareto_front_size",
        ]
    ].copy()

    return wes_chart, tradeoff_chart


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run a single instance, two seeds, all alphas, "
            "writing to an isolated smoke folder."
        ),
    )
    args = parser.parse_args()

    if args.smoke:
        isolate_smoke_outputs()

    started_at = time.perf_counter()

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    data = load_planning_data(
        poi_file=POI_FILE,
        matrix_file=MATRIX_FILE,
    )

    routes_by_seed = load_initial_routes()
    initial_schedules = build_initial_schedules(
        data, routes_by_seed
    )
    scenario_lookup = build_scenario_lookup()

    if args.smoke:
        instance_ids = ("S1_nominal",)
        seeds = (1001, 1005)
    else:
        instance_ids = WEATHER_INSTANCE_IDS
        seeds = SEEDS

    raw_rows: list[dict] = []
    front_rows: list[dict] = []

    total_runs = len(instance_ids) * len(seeds) * len(C2_ALPHAS)
    completed_runs = 0

    for seed in seeds:
        initial_route = routes_by_seed[seed]
        initial_schedule = initial_schedules[seed]

        for instance_id in instance_ids:
            scenario_instance = scenario_lookup[instance_id]

            state = simulate_replanning_state(
                data=data,
                initial_schedule=initial_schedule,
                scenario_instance=scenario_instance,
            )

            problem = prepare_adaptive_problem(
                data=data,
                initial_schedule=initial_schedule,
                state=state,
                scenario_instance=scenario_instance,
            )

            # ----------------------------------------------------
            # Alpha-independent non-thermal baseline
            # ----------------------------------------------------

            baseline_start = time.perf_counter()

            adaptive_result = (
                run_adaptive_without_thermal_ga(
                    data=data,
                    problem=problem,
                    seed=seed,
                    config=REPLANNING_GA_CONFIG,
                )
            )
            adaptive_solution = (
                adaptive_result.selected_solution
            )
            adaptive_exposure = (
                calculate_exposure_metrics(
                    data=data,
                    initial_schedule=initial_schedule,
                    state=state,
                    repaired_schedule=(
                        adaptive_solution.schedule
                    ),
                    thermal_multiplier=(
                        scenario_instance
                        .scenario
                        .thermal_multiplier
                    ),
                )
            )

            baseline_runtime = (
                time.perf_counter() - baseline_start
            )

            for alpha in C2_ALPHAS:
                policy = build_policy(alpha)

                run_start = time.perf_counter()

                tripmates_result = (
                    run_tripmates_lite_ga(
                        data=data,
                        initial_schedule=initial_schedule,
                        problem=problem,
                        seed=seed,
                        config=REPLANNING_GA_CONFIG,
                        selection_policy=policy,
                    )
                )

                tripmates_solution = (
                    tripmates_result.selected_solution
                )
                tripmates_exposure = (
                    tripmates_solution.exposure
                )

                run_runtime = (
                    time.perf_counter() - run_start
                )

                wes_difference = (
                    adaptive_exposure
                    .weather_weighted_exposure_score
                    - tripmates_exposure
                    .weather_weighted_exposure_score
                )

                utility_difference = (
                    adaptive_solution.achieved_utility
                    - tripmates_solution.achieved_utility
                )

                # Guardrail status of the actually selected route.
                # On a structured fallback the selector returns the
                # non-thermal reference, which passes both guardrails
                # by construction.
                if tripmates_result.guardrail_fallback_used:
                    selected_utility_guardrail = True
                    selected_visit_guardrail = True
                else:
                    selected_index = next(
                        index
                        for index, flag in enumerate(
                            tripmates_result.front_selected
                        )
                        if flag
                    )
                    selected_utility_guardrail = bool(
                        tripmates_result
                        .front_utility_guardrail[selected_index]
                    )
                    selected_visit_guardrail = bool(
                        tripmates_result
                        .front_visit_guardrail[selected_index]
                    )

                # ----------------------------------------------
                # Itinerary stability (shared definition)
                #
                # Both planners start from the same trigger state and
                # the same original suffix, so the two edit distances
                # are directly comparable.
                # ----------------------------------------------

                adaptive_stability = (
                    compute_stability_from_state(
                        state=state,
                        replanned_suffix=(
                            adaptive_solution.route
                        ),
                        original_schedule=(
                            initial_schedule
                        ),
                        replanned_schedule=(
                            adaptive_solution.schedule
                        ),
                        repair_record=(
                            adaptive_result.repair_record
                        ),
                    )
                )

                tripmates_stability = (
                    compute_stability_from_state(
                        state=state,
                        replanned_suffix=(
                            tripmates_solution.route
                        ),
                        original_schedule=(
                            initial_schedule
                        ),
                        replanned_schedule=(
                            tripmates_solution.schedule
                        ),
                        repair_record=(
                            tripmates_result.repair_record
                        ),
                    )
                )

                raw_rows.append(
                    {
                        "instance_id": instance_id,
                        "scenario_id": (
                            scenario_instance
                            .scenario
                            .scenario_id
                        ),
                        "scenario_name": (
                            scenario_instance.scenario.name
                        ),
                        "trigger_variant": (
                            scenario_instance.trigger_variant
                        ),
                        "seed": seed,
                        "alpha": alpha,
                        "search_policy_alpha": (
                            DEFAULT_THERMAL_SELECTION_POLICY
                            .alpha
                        ),
                        "final_policy_alpha": alpha,
                        "search_policy_mode": (
                            SEARCH_POLICY_MODE
                        ),
                        "final_selection_policy_mode": (
                            FINAL_SELECTION_POLICY_MODE
                        ),
                        "thermal_multiplier": (
                            scenario_instance
                            .scenario
                            .thermal_multiplier
                        ),
                        "candidate_window": encode_ids(
                            problem
                            .candidate_window
                            .all_candidate_ids
                        ),
                        "candidate_count": len(
                            problem
                            .candidate_window
                            .all_candidate_ids
                        ),

                        "adaptive_route": encode_ids(
                            adaptive_solution.route
                        ),
                        "tripmates_route": encode_ids(
                            tripmates_solution.route
                        ),
                        "route_changed": (
                            adaptive_solution.route
                            != tripmates_solution.route
                        ),

                        # --------------------------------------
                        # Itinerary stability
                        # --------------------------------------

                        "adaptive_preserved_poi_count": (
                            adaptive_stability
                            .preserved_poi_count
                        ),
                        "adaptive_removed_poi_count": len(
                            adaptive_stability.removed_pois
                        ),
                        "adaptive_added_poi_count": len(
                            adaptive_stability.added_pois
                        ),
                        "adaptive_changed_activity_count": (
                            adaptive_stability
                            .changed_activity_count
                        ),
                        "adaptive_time_shift_count": (
                            adaptive_stability
                            .time_shift_count
                        ),
                        "adaptive_order_deviation": (
                            adaptive_stability
                            .order_deviation
                        ),
                        "adaptive_edit_distance_score": (
                            adaptive_stability
                            .edit_distance_score
                        ),
                        "adaptive_locked_prefix_changed": (
                            adaptive_stability
                            .locked_prefix_changed
                        ),

                        "tripmates_preserved_poi_count": (
                            tripmates_stability
                            .preserved_poi_count
                        ),
                        "tripmates_removed_poi_count": len(
                            tripmates_stability.removed_pois
                        ),
                        "tripmates_added_poi_count": len(
                            tripmates_stability.added_pois
                        ),
                        "tripmates_changed_activity_count": (
                            tripmates_stability
                            .changed_activity_count
                        ),
                        "tripmates_time_shift_count": (
                            tripmates_stability
                            .time_shift_count
                        ),
                        "tripmates_order_deviation": (
                            tripmates_stability
                            .order_deviation
                        ),
                        "tripmates_edit_distance_score": (
                            tripmates_stability
                            .edit_distance_score
                        ),
                        "tripmates_locked_prefix_changed": (
                            tripmates_stability
                            .locked_prefix_changed
                        ),

                        "edit_distance_difference": (
                            tripmates_stability
                            .edit_distance_score
                            - adaptive_stability
                            .edit_distance_score
                        ),

                        "adaptive_completed_visits": (
                            adaptive_solution
                            .completed_poi_count
                        ),
                        "tripmates_completed_visits": (
                            tripmates_solution
                            .completed_poi_count
                        ),
                        "completed_visit_loss": (
                            adaptive_solution
                            .completed_poi_count
                            - tripmates_solution
                            .completed_poi_count
                        ),

                        "adaptive_utility": (
                            adaptive_solution
                            .achieved_utility
                        ),
                        "tripmates_utility": (
                            tripmates_solution
                            .achieved_utility
                        ),
                        "utility_difference": (
                            utility_difference
                        ),
                        "utility_loss_percent": (
                            reduction_percent(
                                adaptive_solution
                                .achieved_utility,
                                tripmates_solution
                                .achieved_utility,
                            )
                        ),

                        "adaptive_travel_min": (
                            adaptive_solution
                            .total_travel_min
                        ),
                        "tripmates_travel_min": (
                            tripmates_solution
                            .total_travel_min
                        ),
                        "travel_change_min": (
                            tripmates_solution
                            .total_travel_min
                            - adaptive_solution
                            .total_travel_min
                        ),

                        "adaptive_wes": (
                            adaptive_exposure
                            .weather_weighted_exposure_score
                        ),
                        "tripmates_wes": (
                            tripmates_exposure
                            .weather_weighted_exposure_score
                        ),
                        "wes_difference": wes_difference,
                        "wes_reduction_percent": (
                            reduction_percent(
                                adaptive_exposure
                                .weather_weighted_exposure_score,
                                tripmates_exposure
                                .weather_weighted_exposure_score,
                            )
                        ),

                        "utility_guardrail_passed": (
                            selected_utility_guardrail
                        ),
                        "visit_guardrail_passed": (
                            selected_visit_guardrail
                        ),
                        "guardrail_fallback_used": (
                            tripmates_result
                            .guardrail_fallback_used
                        ),
                        "selection_status": (
                            tripmates_result
                            .selection_status
                        ),

                        "pareto_front_size": len(
                            tripmates_result.first_front
                        ),
                        "u_star": tripmates_result.u_star,
                        "reference_utility": (
                            tripmates_result
                            .reference_utility
                        ),
                        "front_max_utility": (
                            tripmates_result
                            .front_max_utility
                        ),

                        "feasible": True,

                        "adaptive_generations": (
                            adaptive_result
                            .generations_executed
                        ),
                        "tripmates_generations": (
                            tripmates_result
                            .generations_executed
                        ),
                        "adaptive_unique_routes": (
                            adaptive_result
                            .unique_decoded_route_count
                        ),
                        "tripmates_unique_routes": (
                            tripmates_result
                            .unique_decoded_route_count
                        ),

                        "adaptive_runtime_seconds": round(
                            baseline_runtime, 6
                        ),
                        "tripmates_runtime_seconds": round(
                            run_runtime, 6
                        ),
                    }
                )

                front_rows.extend(
                    build_front_rows(
                        result=tripmates_result,
                        instance_id=instance_id,
                        scenario_instance=(
                            scenario_instance
                        ),
                        seed=seed,
                        alpha=alpha,
                    )
                )

                completed_runs += 1

                print(
                    f"[{completed_runs:>3}/{total_runs}] "
                    f"seed={seed} {instance_id} "
                    f"alpha={alpha:.2f} "
                    f"changed={adaptive_solution.route != tripmates_solution.route} "
                    f"WES_reduction={reduction_percent(adaptive_exposure.weather_weighted_exposure_score, tripmates_exposure.weather_weighted_exposure_score):.2f}% "
                    f"front={len(tripmates_result.first_front)} "
                    f"fallback={tripmates_result.guardrail_fallback_used}"
                )

    raw = pd.DataFrame(raw_rows)
    fronts = pd.DataFrame(front_rows)

    raw.to_csv(RAW_RUNS_FILE, index=False, float_format="%.6f")
    fronts.to_csv(
        PARETO_FRONTS_FILE, index=False, float_format="%.6f"
    )

    summary = build_instance_summary(raw)
    summary.to_csv(
        INSTANCE_SUMMARY_FILE,
        index=False,
        float_format="%.6f",
    )

    wes_chart, tradeoff_chart = build_chart_tables(summary)
    wes_chart.to_csv(
        CHART_WES_FILE, index=False, float_format="%.6f"
    )
    tradeoff_chart.to_csv(
        CHART_TRADEOFF_FILE, index=False, float_format="%.6f"
    )

    total_runtime = time.perf_counter() - started_at

    run_config = {
        "experiment": EXPERIMENT_NAME,
        "classification": (
            "exploratory sensitivity analysis"
        ),
        "mode": "smoke_test" if args.smoke else "research_benchmark",
        "timestamp_utc": utc_timestamp(),

        # --------------------------------------------------------
        # Alpha-independence provenance (V2)
        # --------------------------------------------------------
        "search_policy_mode": SEARCH_POLICY_MODE,
        "final_selection_policy_mode": (
            FINAL_SELECTION_POLICY_MODE
        ),
        "supersedes": SUPERSEDES_DIRECTORY,
        "is_confirmatory": IS_CONFIRMATORY,

        "search_selection_policy": asdict(
            DEFAULT_THERMAL_SELECTION_POLICY
        ),
        "alpha_independence_note": (
            "alpha affects ONLY the final route selection. The "
            "generation-zero progress reference, the in-loop "
            "stagnation comparison, the convergence generation and "
            "the evaluated Pareto front all use the frozen default "
            "search policy above, so the GA trajectory is identical "
            "across the four alphas. Only the returned route, its "
            "guardrail status and its fallback flag may vary."
        ),
        "supersedes_note": (
            "The original results/thermal_tradeoff_sensitivity/ run "
            "mixed an alpha-sensitive generation-zero selection with "
            "an alpha-independent in-loop stagnation selection, which "
            "made the search trajectory unintentionally "
            "alpha-dependent. It is preserved unchanged and is "
            "superseded by this V2 run."
        ),

        "planners": [
            "Adaptive Without Thermal",
            "TripMates Lite",
        ],

        "alphas": list(C2_ALPHAS),
        "seeds": list(seeds),
        "instance_ids": list(instance_ids),
        "weather_instance_ids": list(
            WEATHER_INSTANCE_IDS
        ),

        "c2_policy": {
            "utility_guardrail_definition": (
                "route.utility >= alpha * U*"
            ),
            "u_star_definition": (
                "max(first-front maximum utility, "
                "seed-matched non-thermal reference utility)"
            ),
            "visit_guardrail_definition": (
                "route.completed_poi_count >= "
                "reference.completed_poi_count - margin"
            ),
            **C2_POLICY_TEMPLATE,
        },

        "non_thermal_baseline_reused_across_alphas": (
            REUSE_NON_THERMAL_BASELINE
        ),
        "non_thermal_baseline_note": (
            "The non-thermal baseline is deterministic and "
            "alpha-independent, so it is computed once per "
            "(instance, seed) and replicated across the four alphas."
        ),

        "candidate_window_limit": 10,
        "ga_configuration": asdict(
            REPLANNING_GA_CONFIG
        ),

        "stability_weights": (
            StabilityWeightConfig().to_dict()
        ),
        "stability_metric_module": (
            "stability_metrics.py"
        ),

        "dataset": file_fingerprint(POI_FILE),
        "matrix": file_fingerprint(MATRIX_FILE),
        "initial_routes": file_fingerprint(
            Path(
                "results/static_utility/"
                "initial_routes_by_seed.json"
            )
        ),

        "seed_interpretation": (
            "The ten seeds are repeated stochastic optimisation "
            "runs for stability and paired comparison, not ten "
            "independent real-world disruptions."
        ),

        "raw_run_count": int(len(raw)),
        "pareto_front_row_count": int(len(fronts)),
        "total_runtime_seconds": round(total_runtime, 3),
        "runtime_metadata": runtime_metadata(),

        "original_result_preserved": (
            "results/weather_ablation/ is untouched; the frozen "
            "90-pair null result remains the confirmatory result. "
            "results/thermal_tradeoff_sensitivity/ is also left "
            "unchanged and is superseded by this V2 run."
        ),
    }

    write_json(RUN_CONFIG_FILE, run_config)

    print()
    print("=" * 100)
    print("C2 THERMAL TRADE-OFF SENSITIVITY SUMMARY")
    print("=" * 100)
    print("Mode:", run_config["mode"])
    print("Raw runs:", len(raw))
    print("Pareto-front rows:", len(fronts))
    print("Route changes:", int(raw["route_changed"].sum()))
    print(
        "Fallbacks:",
        int(raw["guardrail_fallback_used"].sum()),
    )
    print(
        "Seeds with lower WES:",
        int((raw["wes_difference"] > 1e-9).sum()),
    )
    print(
        "Total runtime:",
        f"{total_runtime:.1f} s",
    )

    if not summary.empty:
        print()
        print(
            summary[
                [
                    "instance_id",
                    "alpha",
                    "route_change_count",
                    "median_wes_reduction_percent",
                    "median_utility_loss_percent",
                    "fallback_count",
                    "median_pareto_front_size",
                    "seeds_lower_wes",
                ]
            ].to_string(index=False)
        )

    print()
    print("Raw runs:", RAW_RUNS_FILE.resolve())
    print("Pareto fronts:", PARETO_FRONTS_FILE.resolve())
    print("Instance summary:", INSTANCE_SUMMARY_FILE.resolve())
    print("Run config:", RUN_CONFIG_FILE.resolve())


if __name__ == "__main__":
    main()
