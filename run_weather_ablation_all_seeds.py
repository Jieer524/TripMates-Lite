"""
Run the complete thermal-ablation experiment.

Comparison:
    Adaptive Without Thermal
    versus
    TripMates Lite

Conditions:
    9 weather-related scenario instances
    x 10 paired seeds
    = 90 paired comparisons

The script generates:

1. One row for every paired seed-instance result.
2. One summary row for every scenario instance.
3. A frozen run-configuration JSON file.

This script records guardrail failures instead of deleting or
silently correcting them. Only hard-constraint violations stop
execution.
"""

from __future__ import annotations

from dataclasses import asdict
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

from exposure_core import (
    calculate_exposure_metrics,
)

from planning_core import (
    COST_BUDGET_RM,
    EXPERIMENT_DAY,
    TRIP_END_MIN,
    TRIP_START_MIN,
    decode_route,
    load_planning_data,
)

from scenario_core import (
    build_scenario_instances,
    simulate_replanning_state,
)

from tripmates_lite_ga import (
    run_tripmates_lite_ga,
)


# ================================================================
# Configuration
# ================================================================

POI_FILE = Path(
    "poi_data_calculated_final.csv"
)

MATRIX_FILE = Path(
    "walking_time_matrix_final.csv"
)

INITIAL_ROUTES_FILE = Path(
    "results/static_utility/"
    "initial_routes_by_seed.json"
)

OUTPUT_DIRECTORY = Path(
    "results/weather_ablation"
)

RAW_RESULTS_FILE = (
    OUTPUT_DIRECTORY
    / "weather_ablation_raw_90_pairs.csv"
)

INSTANCE_SUMMARY_FILE = (
    OUTPUT_DIRECTORY
    / "weather_ablation_instance_summary.csv"
)

RUN_CONFIGURATION_FILE = (
    OUTPUT_DIRECTORY
    / "weather_ablation_run_config.json"
)

SEEDS = tuple(
    range(1001, 1011)
)

WEATHER_INSTANCE_IDS = (
    "S1_early",
    "S1_nominal",
    "S1_late",

    "S5_early",
    "S5_nominal",
    "S5_late",

    "S6_early",
    "S6_nominal",
    "S6_late",
)


# ================================================================
# General helpers
# ================================================================

def calculate_reduction_percent(
    reference_value: float,
    comparison_value: float,
) -> float:
    """
    Calculate:

        (reference - comparison)
        / reference
        * 100

    For WES:
        positive = TripMates Lite reduced exposure

    For utility:
        positive = TripMates Lite lost utility
    """

    if abs(reference_value) <= 1e-12:
        return 0.0

    return (
        (
            reference_value
            - comparison_value
        )
        / reference_value
        * 100.0
    )


def encode_ids(
    values,
) -> str:
    """Store a POI sequence clearly inside a CSV cell."""

    return json.dumps(
        [
            int(value)
            for value in values
        ]
    )


def validate_selected_solution(
    *,
    data,
    state,
    solution,
    planner_name: str,
    seed: int,
    instance_id: str,
) -> None:
    """
    Validate hard constraints.

    Acceptance criteria such as thermal improvement are not hard
    constraints. They are recorded later rather than raised here.
    """

    schedule = solution.schedule

    label = (
        f"{planner_name}, seed {seed}, "
        f"{instance_id}"
    )

    if not schedule.feasible:
        raise RuntimeError(
            f"{label}: selected schedule "
            f"is infeasible: "
            f"{schedule.failure_reason}"
        )

    if schedule.missing_mandatory_ids:
        raise RuntimeError(
            f"{label}: still-feasible mandatory "
            "POIs were omitted: "
            f"{list(schedule.missing_mandatory_ids)}"
        )

    repaired_route = list(
        solution.route
    )

    completed_route = list(
        state.completed_poi_ids
    )

    final_visit_route = (
        completed_route
        + repaired_route
    )

    if len(final_visit_route) != len(
        set(final_visit_route)
    ):
        raise RuntimeError(
            f"{label}: duplicate POI visits "
            "exist in the final route."
        )

    unavailable_in_repair = (
        set(repaired_route)
        & set(state.unavailable_poi_ids)
    )

    if unavailable_in_repair:
        raise RuntimeError(
            f"{label}: unavailable POIs occur "
            "in the repaired route: "
            f"{sorted(unavailable_in_repair)}"
        )

    if (
        schedule.total_cost_rm
        > COST_BUDGET_RM + 1e-9
    ):
        raise RuntimeError(
            f"{label}: cost exceeds "
            f"RM{COST_BUDGET_RM:.2f}."
        )

    if (
        schedule.finish_min
        > TRIP_END_MIN + 1e-9
    ):
        raise RuntimeError(
            f"{label}: route finishes "
            "after 17:00."
        )

    unknown_ids = (
        set(final_visit_route)
        - set(data.pois)
    )

    if unknown_ids:
        raise RuntimeError(
            f"{label}: unknown POI IDs found: "
            f"{sorted(unknown_ids)}"
        )


# ================================================================
# Instance summary
# ================================================================

def build_instance_summary(
    raw_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarise each scenario instance across its ten paired seeds.

    The ten seeds measure GA stability. They are not treated as ten
    independent real-world disruption events.
    """

    summary_rows: list[dict] = []

    for instance_id in (
        WEATHER_INSTANCE_IDS
    ):
        group = raw_dataframe.loc[
            raw_dataframe[
                "instance_id"
            ] == instance_id
        ].copy()

        if len(group) != len(SEEDS):
            raise RuntimeError(
                f"{instance_id}: expected "
                f"{len(SEEDS)} seed rows, "
                f"found {len(group)}."
            )

        summary_rows.append(
            {
                "instance_id": (
                    instance_id
                ),

                "scenario_id": (
                    group[
                        "scenario_id"
                    ].iloc[0]
                ),

                "scenario_name": (
                    group[
                        "scenario_name"
                    ].iloc[0]
                ),

                "trigger_variant": (
                    group[
                        "trigger_variant"
                    ].iloc[0]
                ),

                "thermal_multiplier": (
                    group[
                        "thermal_multiplier"
                    ].iloc[0]
                ),

                "seed_count": len(
                    group
                ),

                "unique_initial_routes": (
                    group[
                        "initial_route"
                    ].nunique()
                ),

                "route_change_count": int(
                    group[
                        "route_changed"
                    ].sum()
                ),

                "route_change_percent": (
                    float(
                        group[
                            "route_changed"
                        ].mean()
                        * 100.0
                    )
                ),

                # ----------------------------------------------
                # WES
                # ----------------------------------------------

                "median_adaptive_wes": (
                    float(
                        group[
                            "adaptive_wes"
                        ].median()
                    )
                ),

                "median_tripmates_wes": (
                    float(
                        group[
                            "tripmates_wes"
                        ].median()
                    )
                ),

                "median_paired_wes_difference": (
                    float(
                        group[
                            "wes_difference"
                        ].median()
                    )
                ),

                "median_paired_wes_reduction_percent": (
                    float(
                        group[
                            "wes_reduction_percent"
                        ].median()
                    )
                ),

                "mean_paired_wes_reduction_percent": (
                    float(
                        group[
                            "wes_reduction_percent"
                        ].mean()
                    )
                ),

                "minimum_wes_reduction_percent": (
                    float(
                        group[
                            "wes_reduction_percent"
                        ].min()
                    )
                ),

                "maximum_wes_reduction_percent": (
                    float(
                        group[
                            "wes_reduction_percent"
                        ].max()
                    )
                ),

                "seeds_with_positive_wes_reduction": int(
                    (
                        group[
                            "wes_reduction_percent"
                        ]
                        > 1e-9
                    ).sum()
                ),

                # ----------------------------------------------
                # Utility
                # ----------------------------------------------

                "median_adaptive_utility": (
                    float(
                        group[
                            "adaptive_utility"
                        ].median()
                    )
                ),

                "median_tripmates_utility": (
                    float(
                        group[
                            "tripmates_utility"
                        ].median()
                    )
                ),

                "median_utility_loss_percent": (
                    float(
                        group[
                            "utility_loss_percent"
                        ].median()
                    )
                ),

                # ----------------------------------------------
                # Completed visits
                # ----------------------------------------------

                "median_adaptive_completed_visits": (
                    float(
                        group[
                            "adaptive_completed_visits"
                        ].median()
                    )
                ),

                "median_tripmates_completed_visits": (
                    float(
                        group[
                            "tripmates_completed_visits"
                        ].median()
                    )
                ),

                "median_completed_visit_loss": (
                    float(
                        group[
                            "completed_visit_loss"
                        ].median()
                    )
                ),

                # ----------------------------------------------
                # Travel
                # ----------------------------------------------

                "median_adaptive_travel_min": (
                    float(
                        group[
                            "adaptive_travel_min"
                        ].median()
                    )
                ),

                "median_tripmates_travel_min": (
                    float(
                        group[
                            "tripmates_travel_min"
                        ].median()
                    )
                ),

                "median_travel_change_min": (
                    float(
                        group[
                            "travel_change_min"
                        ].median()
                    )
                ),

                # ----------------------------------------------
                # Guardrails and feasibility
                # ----------------------------------------------

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

                "all_utility_guardrails_passed": bool(
                    group[
                        "utility_guardrail_passed"
                    ].all()
                ),

                "all_visit_guardrails_passed": bool(
                    group[
                        "visit_guardrail_passed"
                    ].all()
                ),

                "all_hard_constraints_passed": bool(
                    group[
                        "hard_constraints_passed"
                    ].all()
                ),

                # ----------------------------------------------
                # GA stability
                # ----------------------------------------------

                "median_adaptive_generations": (
                    float(
                        group[
                            "adaptive_generations"
                        ].median()
                    )
                ),

                "median_tripmates_generations": (
                    float(
                        group[
                            "tripmates_generations"
                        ].median()
                    )
                ),

                "median_adaptive_convergence_generation": (
                    float(
                        group[
                            "adaptive_convergence_generation"
                        ].median()
                    )
                ),

                "median_tripmates_convergence_generation": (
                    float(
                        group[
                            "tripmates_convergence_generation"
                        ].median()
                    )
                ),
            }
        )

    return pd.DataFrame(
        summary_rows
    )


# ================================================================
# Main experiment
# ================================================================

def main() -> None:
    experiment_start = (
        time.perf_counter()
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_planning_data(
        poi_file=POI_FILE,
        matrix_file=MATRIX_FILE,
    )

    routes_by_seed = json.loads(
        INITIAL_ROUTES_FILE.read_text(
            encoding="utf-8"
        )
    )

    expected_seed_keys = {
        str(seed)
        for seed in SEEDS
    }

    missing_seed_routes = (
        expected_seed_keys
        - set(routes_by_seed)
    )

    if missing_seed_routes:
        raise RuntimeError(
            "Initial routes are missing for seeds: "
            f"{sorted(missing_seed_routes)}"
        )

    scenario_lookup = {
        instance.instance_id: instance
        for instance in (
            build_scenario_instances()
        )
    }

    missing_instances = (
        set(WEATHER_INSTANCE_IDS)
        - set(scenario_lookup)
    )

    if missing_instances:
        raise RuntimeError(
            "Weather scenario instances "
            "are missing: "
            f"{sorted(missing_instances)}"
        )

    raw_rows: list[dict] = []

    total_pairs = (
        len(SEEDS)
        * len(WEATHER_INSTANCE_IDS)
    )

    completed_pairs = 0

    for seed in SEEDS:
        initial_route = [
            int(poi_id)
            for poi_id in (
                routes_by_seed[
                    str(seed)
                ]
            )
        ]

        initial_schedule = decode_route(
            data=data,
            visit_sequence=initial_route,

            start_poi_id=1,
            start_time_min=TRIP_START_MIN,
            trip_end_min=TRIP_END_MIN,

            day_code=EXPERIMENT_DAY,
            cost_budget_rm=COST_BUDGET_RM,

            start_poi_is_visit=True,

            required_mandatory_ids=(
                data.mandatory_ids
            ),
        )

        if not initial_schedule.feasible:
            raise RuntimeError(
                f"Seed {seed}: initial route "
                "is infeasible."
            )

        for instance_id in (
            WEATHER_INSTANCE_IDS
        ):
            pair_start = (
                time.perf_counter()
            )

            scenario_instance = (
                scenario_lookup[
                    instance_id
                ]
            )

            state = (
                simulate_replanning_state(
                    data=data,
                    initial_schedule=(
                        initial_schedule
                    ),
                    scenario_instance=(
                        scenario_instance
                    ),
                )
            )

            problem = (
                prepare_adaptive_problem(
                    data=data,
                    initial_schedule=(
                        initial_schedule
                    ),
                    state=state,
                    scenario_instance=(
                        scenario_instance
                    ),
                )
            )

            candidate_window = (
                problem
                .candidate_window
                .all_candidate_ids
            )

            if len(candidate_window) > 10:
                raise RuntimeError(
                    f"Seed {seed}, "
                    f"{instance_id}: candidate "
                    "window exceeds ten POIs."
                )

            if not set(
                state.remaining_mandatory_ids
            ).issubset(
                candidate_window
            ):
                raise RuntimeError(
                    f"Seed {seed}, "
                    f"{instance_id}: candidate "
                    "window omits a mandatory POI."
                )

            # ====================================================
            # Adaptive Without Thermal
            # ====================================================

            adaptive_result = (
                run_adaptive_without_thermal_ga(
                    data=data,
                    problem=problem,
                    seed=seed,
                    config=(
                        REPLANNING_GA_CONFIG
                    ),
                )
            )

            adaptive_solution = (
                adaptive_result
                .selected_solution
            )

            validate_selected_solution(
                data=data,
                state=state,
                solution=(
                    adaptive_solution
                ),
                planner_name=(
                    "Adaptive Without Thermal"
                ),
                seed=seed,
                instance_id=instance_id,
            )

            adaptive_exposure = (
                calculate_exposure_metrics(
                    data=data,
                    initial_schedule=(
                        initial_schedule
                    ),
                    state=state,
                    repaired_schedule=(
                        adaptive_solution
                        .schedule
                    ),
                    thermal_multiplier=(
                        scenario_instance
                        .scenario
                        .thermal_multiplier
                    ),
                )
            )

            # ====================================================
            # TripMates Lite
            # ====================================================

            tripmates_result = (
                run_tripmates_lite_ga(
                    data=data,
                    initial_schedule=(
                        initial_schedule
                    ),
                    problem=problem,
                    seed=seed,
                    config=(
                        REPLANNING_GA_CONFIG
                    ),
                )
            )

            tripmates_solution = (
                tripmates_result
                .selected_solution
            )

            validate_selected_solution(
                data=data,
                state=state,
                solution=(
                    tripmates_solution
                ),
                planner_name=(
                    "TripMates Lite"
                ),
                seed=seed,
                instance_id=instance_id,
            )

            if not (
                tripmates_result
                .thermal_objective_active
            ):
                raise RuntimeError(
                    f"Seed {seed}, "
                    f"{instance_id}: thermal "
                    "objective should be active."
                )

            tripmates_exposure = (
                tripmates_solution
                .exposure
            )

            # ====================================================
            # Paired comparison
            # ====================================================

            wes_difference = (
                adaptive_exposure
                .weather_weighted_exposure_score
                - tripmates_exposure
                .weather_weighted_exposure_score
            )

            wes_reduction_percent = (
                calculate_reduction_percent(
                    reference_value=(
                        adaptive_exposure
                        .weather_weighted_exposure_score
                    ),
                    comparison_value=(
                        tripmates_exposure
                        .weather_weighted_exposure_score
                    ),
                )
            )

            utility_difference = (
                adaptive_solution
                .achieved_utility
                - tripmates_solution
                .achieved_utility
            )

            utility_loss_percent = (
                calculate_reduction_percent(
                    reference_value=(
                        adaptive_solution
                        .achieved_utility
                    ),
                    comparison_value=(
                        tripmates_solution
                        .achieved_utility
                    ),
                )
            )

            completed_visit_loss = (
                adaptive_solution
                .completed_poi_count
                - tripmates_solution
                .completed_poi_count
            )

            travel_change_min = (
                tripmates_solution
                .total_travel_min
                - adaptive_solution
                .total_travel_min
            )

            route_changed = (
                adaptive_solution.route
                != tripmates_solution.route
            )

            utility_guardrail_passed = (
                utility_loss_percent
                <= 10.0 + 1e-9
            )

            visit_guardrail_passed = (
                completed_visit_loss
                <= 1
            )

            pair_runtime_seconds = (
                time.perf_counter()
                - pair_start
            )

            raw_rows.append(
                {
                    "seed": seed,

                    "instance_id": (
                        instance_id
                    ),

                    "scenario_id": (
                        scenario_instance
                        .scenario
                        .scenario_id
                    ),

                    "scenario_name": (
                        scenario_instance
                        .scenario
                        .name
                    ),

                    "trigger_variant": (
                        scenario_instance
                        .trigger_variant
                    ),

                    "trigger_clock": (
                        state.trigger_clock
                    ),

                    "replan_start_clock": (
                        state.replan_start_clock
                    ),

                    "committed_component": (
                        state
                        .committed_component
                    ),

                    "current_poi_id": (
                        state.current_poi_id
                    ),

                    "thermal_multiplier": (
                        scenario_instance
                        .scenario
                        .thermal_multiplier
                    ),

                    "initial_route": (
                        encode_ids(
                            initial_route
                        )
                    ),

                    "completed_before_replan": (
                        encode_ids(
                            state
                            .completed_poi_ids
                        )
                    ),

                    "remaining_mandatory_ids": (
                        encode_ids(
                            state
                            .remaining_mandatory_ids
                        )
                    ),

                    "unavailable_poi_ids": (
                        encode_ids(
                            state
                            .unavailable_poi_ids
                        )
                    ),

                    "candidate_window": (
                        encode_ids(
                            candidate_window
                        )
                    ),

                    "candidate_count": len(
                        candidate_window
                    ),

                    "adaptive_route": (
                        encode_ids(
                            adaptive_solution
                            .route
                        )
                    ),

                    "tripmates_route": (
                        encode_ids(
                            tripmates_solution
                            .route
                        )
                    ),

                    "route_changed": (
                        route_changed
                    ),

                    # ------------------------------------------
                    # Completed visits
                    # ------------------------------------------

                    "adaptive_completed_visits": (
                        adaptive_solution
                        .completed_poi_count
                    ),

                    "tripmates_completed_visits": (
                        tripmates_solution
                        .completed_poi_count
                    ),

                    "completed_visit_loss": (
                        completed_visit_loss
                    ),

                    # ------------------------------------------
                    # Utility
                    # ------------------------------------------

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
                        utility_loss_percent
                    ),

                    # ------------------------------------------
                    # Travel
                    # ------------------------------------------

                    "adaptive_travel_min": (
                        adaptive_solution
                        .total_travel_min
                    ),

                    "tripmates_travel_min": (
                        tripmates_solution
                        .total_travel_min
                    ),

                    "travel_change_min": (
                        travel_change_min
                    ),

                    # ------------------------------------------
                    # Exposure
                    # ------------------------------------------

                    "adaptive_owm": (
                        adaptive_exposure
                        .outdoor_walking_min
                    ),

                    "tripmates_owm": (
                        tripmates_exposure
                        .outdoor_walking_min
                    ),

                    "adaptive_ovm": (
                        adaptive_exposure
                        .outdoor_visit_min
                    ),

                    "tripmates_ovm": (
                        tripmates_exposure
                        .outdoor_visit_min
                    ),

                    "adaptive_toem": (
                        adaptive_exposure
                        .total_outdoor_exposure_min
                    ),

                    "tripmates_toem": (
                        tripmates_exposure
                        .total_outdoor_exposure_min
                    ),

                    "adaptive_wes": (
                        adaptive_exposure
                        .weather_weighted_exposure_score
                    ),

                    "tripmates_wes": (
                        tripmates_exposure
                        .weather_weighted_exposure_score
                    ),

                    "wes_difference": (
                        wes_difference
                    ),

                    "wes_reduction_percent": (
                        wes_reduction_percent
                    ),

                    # ------------------------------------------
                    # Final feasibility
                    # ------------------------------------------

                    "adaptive_final_cost_rm": (
                        adaptive_solution
                        .schedule
                        .total_cost_rm
                    ),

                    "tripmates_final_cost_rm": (
                        tripmates_solution
                        .schedule
                        .total_cost_rm
                    ),

                    "adaptive_finish_time": (
                        adaptive_solution
                        .schedule
                        .finish_clock
                    ),

                    "tripmates_finish_time": (
                        tripmates_solution
                        .schedule
                        .finish_clock
                    ),

                    "hard_constraints_passed": (
                        True
                    ),

                    "utility_guardrail_passed": (
                        utility_guardrail_passed
                    ),

                    "visit_guardrail_passed": (
                        visit_guardrail_passed
                    ),

                    # ------------------------------------------
                    # GA behaviour
                    # ------------------------------------------

                    "adaptive_generations": (
                        adaptive_result
                        .generations_executed
                    ),

                    "tripmates_generations": (
                        tripmates_result
                        .generations_executed
                    ),

                    "adaptive_convergence_generation": (
                        adaptive_result
                        .convergence_generation
                    ),

                    "tripmates_convergence_generation": (
                        tripmates_result
                        .convergence_generation
                    ),

                    "adaptive_evaluated_chromosomes": (
                        adaptive_result
                        .evaluated_chromosome_count
                    ),

                    "tripmates_evaluated_chromosomes": (
                        tripmates_result
                        .evaluated_chromosome_count
                    ),

                    "adaptive_unique_routes": (
                        adaptive_result
                        .unique_decoded_route_count
                    ),

                    "tripmates_unique_routes": (
                        tripmates_result
                        .unique_decoded_route_count
                    ),

                    # This is total pair execution time,
                    # not the formal replanning-latency metric.
                    "pair_runtime_seconds": (
                        pair_runtime_seconds
                    ),
                }
            )

            completed_pairs += 1

            print(
                f"[{completed_pairs:>2}/"
                f"{total_pairs}] "
                f"seed={seed}, "
                f"instance={instance_id}, "
                f"changed={route_changed}, "
                f"WES reduction="
                f"{wes_reduction_percent:.2f}%, "
                f"utility loss="
                f"{utility_loss_percent:.2f}%, "
                f"visit loss="
                f"{completed_visit_loss}"
            )

    # ============================================================
    # Save raw and summary results
    # ============================================================

    raw_dataframe = pd.DataFrame(
        raw_rows
    )

    if len(raw_dataframe) != 90:
        raise RuntimeError(
            "Expected 90 paired rows, "
            f"generated {len(raw_dataframe)}."
        )

    raw_dataframe.to_csv(
        RAW_RESULTS_FILE,
        index=False,
        float_format="%.6f",
    )

    instance_summary = (
        build_instance_summary(
            raw_dataframe
        )
    )

    instance_summary.to_csv(
        INSTANCE_SUMMARY_FILE,
        index=False,
        float_format="%.6f",
    )

    total_runtime_seconds = (
        time.perf_counter()
        - experiment_start
    )

    run_configuration = {
        "experiment": (
            "Thermal-awareness ablation"
        ),

        "planners": [
            "Adaptive Without Thermal",
            "TripMates Lite",
        ],

        "seeds": list(
            SEEDS
        ),

        "weather_instance_ids": list(
            WEATHER_INSTANCE_IDS
        ),

        "paired_result_count": len(
            raw_dataframe
        ),

        "candidate_window_limit": 10,

        "ga_configuration": asdict(
            REPLANNING_GA_CONFIG
        ),

        "poi_file": (
            POI_FILE.name
        ),

        "matrix_file": (
            MATRIX_FILE.name
        ),

        "initial_routes_file": str(
            INITIAL_ROUTES_FILE
        ),

        "total_runtime_seconds": (
            total_runtime_seconds
        ),

        "note": (
            "Seeds measure stochastic GA stability. "
            "They are not treated as independent "
            "real-world disruption events."
        ),
    }

    RUN_CONFIGURATION_FILE.write_text(
        json.dumps(
            run_configuration,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ============================================================
    # Console summary
    # ============================================================

    instance_median_reductions = (
        instance_summary[
            "median_paired_wes_reduction_percent"
        ].tolist()
    )

    positive_instances = int(
        (
            instance_summary[
                "median_paired_wes_reduction_percent"
            ]
            > 1e-9
        ).sum()
    )

    total_route_changes = int(
        raw_dataframe[
            "route_changed"
        ].sum()
    )

    utility_guardrail_failures = int(
        (
            ~raw_dataframe[
                "utility_guardrail_passed"
            ]
        ).sum()
    )

    visit_guardrail_failures = int(
        (
            ~raw_dataframe[
                "visit_guardrail_passed"
            ]
        ).sum()
    )

    print()
    print("=" * 120)
    print(
        "COMPLETE WEATHER ABLATION SUMMARY"
    )
    print("=" * 120)

    print(
        "Paired results:",
        len(raw_dataframe),
    )

    print(
        "Instances:",
        len(instance_summary),
    )

    print(
        "Seeds per instance:",
        len(SEEDS),
    )

    print(
        "Route changes:",
        total_route_changes,
    )

    print(
        "Instances with positive median "
        "WES reduction:",
        f"{positive_instances}/"
        f"{len(instance_summary)}",
    )

    print(
        "Median of the nine instance-level "
        "median WES reductions:",
        f"{statistics.median(instance_median_reductions):.2f}%",
    )

    print(
        "Utility guardrail failures:",
        utility_guardrail_failures,
    )

    print(
        "Visit guardrail failures:",
        visit_guardrail_failures,
    )

    print(
        "Total execution time:",
        f"{total_runtime_seconds:.2f} seconds",
    )

    print()
    print(
        instance_summary[
            [
                "instance_id",
                "route_change_count",
                "median_adaptive_wes",
                "median_tripmates_wes",
                "median_paired_wes_reduction_percent",
                "median_utility_loss_percent",
                "median_completed_visit_loss",
                "median_travel_change_min",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Raw results:",
        RAW_RESULTS_FILE.resolve(),
    )

    print(
        "Instance summary:",
        INSTANCE_SUMMARY_FILE.resolve(),
    )

    print(
        "Run configuration:",
        RUN_CONFIGURATION_FILE.resolve(),
    )


if __name__ == "__main__":
    main()