"""
Weather-related thermal-ablation smoke test.

This script compares:

1. Adaptive Without Thermal
2. TripMates Lite

for one paired seed across:

- S1 Heavy Rain: early, nominal and late
- S5 Heatwave: early, nominal and late
- S6 Combined: early, nominal and late

This is an implementation test, not the final statistical analysis.
"""

from __future__ import annotations

import json
import statistics
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

ROUTES_FILE = Path(
    "results/static_utility/"
    "initial_routes_by_seed.json"
)

OUTPUT_DIRECTORY = Path(
    "results/thermal_ablation_smoke"
)

TEST_SEED = 1001

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
# Helpers
# ================================================================

def percentage_change(
    reference_value: float,
    comparison_value: float,
) -> float:
    """
    Calculate:

        (reference - comparison)
        / reference
        * 100

    Positive WES output means that TripMates Lite reduced WES.

    Positive utility output means that TripMates Lite lost utility.
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


def validate_combined_route(
    completed_poi_ids: tuple[int, ...],
    repaired_route: tuple[int, ...],
    instance_id: str,
    planner_name: str,
) -> None:
    final_route = (
        list(completed_poi_ids)
        + list(repaired_route)
    )

    if len(final_route) != len(
        set(final_route)
    ):
        raise RuntimeError(
            f"{instance_id} - {planner_name}: "
            "duplicate visits exist in the final route."
        )


def main() -> None:
    # ------------------------------------------------------------
    # Load frozen inputs
    # ------------------------------------------------------------

    data = load_planning_data(
        poi_file=POI_FILE,
        matrix_file=MATRIX_FILE,
    )

    routes_by_seed = json.loads(
        ROUTES_FILE.read_text(
            encoding="utf-8"
        )
    )

    initial_route = routes_by_seed[
        str(TEST_SEED)
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
            "The initial Static Utility route "
            "is infeasible."
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
            "Missing scenario instances: "
            f"{sorted(missing_instances)}"
        )

    result_rows: list[dict] = []

    # ------------------------------------------------------------
    # Run nine weather-related cases
    # ------------------------------------------------------------

    for instance_id in (
        WEATHER_INSTANCE_IDS
    ):
        print()
        print("=" * 100)
        print(
            f"SEED {TEST_SEED} - "
            f"{instance_id}"
        )
        print("=" * 100)

        scenario_instance = (
            scenario_lookup[
                instance_id
            ]
        )

        state = simulate_replanning_state(
            data=data,
            initial_schedule=(
                initial_schedule
            ),
            scenario_instance=(
                scenario_instance
            ),
        )

        adaptive_problem = prepare_adaptive_problem(
            data=data,
            initial_schedule=initial_schedule,
            state=state,
            scenario_instance=scenario_instance,
            thermal_aware_candidate_window=False,
        )

        tripmates_problem = prepare_adaptive_problem(
            data=data,
            initial_schedule=initial_schedule,
            state=state,
            scenario_instance=scenario_instance,
            thermal_aware_candidate_window=True,
        )

        # --------------------------------------------------------
        # Candidate-window validation
        # --------------------------------------------------------

        adaptive_candidate_ids = (
            adaptive_problem
            .candidate_window
            .all_candidate_ids
        )

        tripmates_candidate_ids = (
            tripmates_problem
            .candidate_window
            .all_candidate_ids
        )

        for planner_name, candidate_ids in (
            (
                "Adaptive Without Thermal",
                adaptive_candidate_ids,
            ),
            (
                "TripMates Lite",
                tripmates_candidate_ids,
            ),
        ):
            if len(candidate_ids) > 10:
                raise RuntimeError(
                    f"{instance_id}: "
                    f"{planner_name} candidate window "
                    "contains more than ten POIs."
                )

            if not set(
                state.remaining_mandatory_ids
            ).issubset(candidate_ids):
                raise RuntimeError(
                    f"{instance_id}: "
                    f"{planner_name} candidate window "
                    "omits a remaining mandatory POI."
                )

        # --------------------------------------------------------
        # Adaptive Without Thermal
        # --------------------------------------------------------

        no_thermal_result = (
            run_adaptive_without_thermal_ga(
                data=data,
                problem=adaptive_problem,
                seed=TEST_SEED,
                config=(
                    REPLANNING_GA_CONFIG
                ),
            )
        )

        no_thermal_solution = (
            no_thermal_result
            .selected_solution
        )

        if not (
            no_thermal_solution
            .schedule
            .feasible
        ):
            raise RuntimeError(
                f"{instance_id}: "
                "Adaptive Without Thermal "
                "produced an infeasible route."
            )

        no_thermal_exposure = (
            calculate_exposure_metrics(
                data=data,
                initial_schedule=(
                    initial_schedule
                ),
                state=state,
                repaired_schedule=(
                    no_thermal_solution
                    .schedule
                ),
                thermal_multiplier=(
                    scenario_instance
                    .scenario
                    .thermal_multiplier
                ),
            )
        )

        # --------------------------------------------------------
        # TripMates Lite
        # --------------------------------------------------------

        tripmates_result = (
            run_tripmates_lite_ga(
                data=data,
                initial_schedule=(
                    initial_schedule
                ),
                problem=tripmates_problem,
                seed=TEST_SEED,
                config=(
                    REPLANNING_GA_CONFIG
                ),
            )
        )

        thermal_solution = (
            tripmates_result
            .selected_solution
        )

        if not (
            thermal_solution
            .schedule
            .feasible
        ):
            raise RuntimeError(
                f"{instance_id}: "
                "TripMates Lite produced "
                "an infeasible route."
            )

        if not (
            tripmates_result
            .thermal_objective_active
        ):
            raise RuntimeError(
                f"{instance_id}: thermal objective "
                "should be active."
            )

        thermal_exposure = (
            thermal_solution.exposure
        )

        # --------------------------------------------------------
        # Route validation
        # --------------------------------------------------------

        validate_combined_route(
            completed_poi_ids=(
                state.completed_poi_ids
            ),
            repaired_route=(
                no_thermal_solution.route
            ),
            instance_id=instance_id,
            planner_name=(
                "Adaptive Without Thermal"
            ),
        )

        validate_combined_route(
            completed_poi_ids=(
                state.completed_poi_ids
            ),
            repaired_route=(
                thermal_solution.route
            ),
            instance_id=instance_id,
            planner_name=(
                "TripMates Lite"
            ),
        )

        if (
            no_thermal_solution
            .schedule
            .missing_mandatory_ids
        ):
            raise RuntimeError(
                f"{instance_id}: "
                "Adaptive Without Thermal "
                "omitted mandatory POIs."
            )

        if (
            thermal_solution
            .schedule
            .missing_mandatory_ids
        ):
            raise RuntimeError(
                f"{instance_id}: "
                "TripMates Lite omitted "
                "mandatory POIs."
            )

        # --------------------------------------------------------
        # Paired differences
        # --------------------------------------------------------

        wes_reduction_percent = (
            percentage_change(
                reference_value=(
                    no_thermal_exposure
                    .weather_weighted_exposure_score
                ),
                comparison_value=(
                    thermal_exposure
                    .weather_weighted_exposure_score
                ),
            )
        )

        utility_loss_percent = (
            percentage_change(
                reference_value=(
                    no_thermal_solution
                    .achieved_utility
                ),
                comparison_value=(
                    thermal_solution
                    .achieved_utility
                ),
            )
        )

        completed_visit_loss = (
            no_thermal_solution
            .completed_poi_count
            - thermal_solution
            .completed_poi_count
        )

        travel_change_min = (
            thermal_solution
            .total_travel_min
            - no_thermal_solution
            .total_travel_min
        )

        route_changed = (
            no_thermal_solution.route
            != thermal_solution.route
        )

        utility_guardrail_passed = (
            utility_loss_percent
            <= 10.0 + 1e-9
        )

        visit_guardrail_passed = (
            completed_visit_loss
            <= 1
        )

        print(
            "Adaptive candidate window:",
            list(adaptive_candidate_ids),
        )

        print(
            "TripMates candidate window:",
            list(tripmates_candidate_ids),
        )

        candidate_windows_differ = (
            adaptive_candidate_ids
            != tripmates_candidate_ids
        )

        print(
            "Candidate windows differ:",
            candidate_windows_differ,
        )

        print(
            "Adaptive route:",
            list(
                no_thermal_solution.route
            ),
        )

        print(
            "TripMates route:",
            list(
                thermal_solution.route
            ),
        )

        print(
            "Route changed:",
            route_changed,
        )

        print(
            "Adaptive WES:",
            f"{no_thermal_exposure.weather_weighted_exposure_score:.6f}",
        )

        print(
            "TripMates WES:",
            f"{thermal_exposure.weather_weighted_exposure_score:.6f}",
        )

        print(
            "WES reduction:",
            f"{wes_reduction_percent:.2f}%",
        )

        print(
            "Utility loss:",
            f"{utility_loss_percent:.2f}%",
        )

        print(
            "Visit loss:",
            completed_visit_loss,
        )

        print(
            "Travel change:",
            f"{travel_change_min:.6f} min",
        )

        print(
            "Utility guardrail:",
            utility_guardrail_passed,
        )

        print(
            "Visit guardrail:",
            visit_guardrail_passed,
        )

        result_rows.append(
            {
                "seed": TEST_SEED,
                "instance_id": (
                    instance_id
                ),
                "scenario_id": (
                    scenario_instance
                    .scenario
                    .scenario_id
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
                "thermal_multiplier": (
                    scenario_instance
                    .scenario
                    .thermal_multiplier
                ),
                "adaptive_candidate_window": (
                    json.dumps(
                        list(adaptive_candidate_ids)
                    )
                ),

                "tripmates_candidate_window": (
                    json.dumps(
                        list(tripmates_candidate_ids)
                    )
                ),

                "adaptive_candidate_count": len(
                    adaptive_candidate_ids
                ),

                "tripmates_candidate_count": len(
                    tripmates_candidate_ids
                ),

                "candidate_windows_differ": (
                    candidate_windows_differ
                ),
                "completed_before_replan": (
                    json.dumps(
                        list(
                            state
                            .completed_poi_ids
                        )
                    )
                ),
                "adaptive_route": (
                    json.dumps(
                        list(
                            no_thermal_solution
                            .route
                        )
                    )
                ),
                "tripmates_route": (
                    json.dumps(
                        list(
                            thermal_solution
                            .route
                        )
                    )
                ),
                "route_changed": (
                    route_changed
                ),
                "adaptive_completed_visits": (
                    no_thermal_solution
                    .completed_poi_count
                ),
                "tripmates_completed_visits": (
                    thermal_solution
                    .completed_poi_count
                ),
                "completed_visit_loss": (
                    completed_visit_loss
                ),
                "adaptive_utility": (
                    no_thermal_solution
                    .achieved_utility
                ),
                "tripmates_utility": (
                    thermal_solution
                    .achieved_utility
                ),
                "utility_loss_percent": (
                    utility_loss_percent
                ),
                "adaptive_travel_min": (
                    no_thermal_solution
                    .total_travel_min
                ),
                "tripmates_travel_min": (
                    thermal_solution
                    .total_travel_min
                ),
                "travel_change_min": (
                    travel_change_min
                ),
                "adaptive_owm": (
                    no_thermal_exposure
                    .outdoor_walking_min
                ),
                "tripmates_owm": (
                    thermal_exposure
                    .outdoor_walking_min
                ),
                "adaptive_ovm": (
                    no_thermal_exposure
                    .outdoor_visit_min
                ),
                "tripmates_ovm": (
                    thermal_exposure
                    .outdoor_visit_min
                ),
                "adaptive_toem": (
                    no_thermal_exposure
                    .total_outdoor_exposure_min
                ),
                "tripmates_toem": (
                    thermal_exposure
                    .total_outdoor_exposure_min
                ),
                "adaptive_wes": (
                    no_thermal_exposure
                    .weather_weighted_exposure_score
                ),
                "tripmates_wes": (
                    thermal_exposure
                    .weather_weighted_exposure_score
                ),
                "wes_reduction_percent": (
                    wes_reduction_percent
                ),
                "utility_guardrail_passed": (
                    utility_guardrail_passed
                ),
                "visit_guardrail_passed": (
                    visit_guardrail_passed
                ),
                "adaptive_generations": (
                    no_thermal_result
                    .generations_executed
                ),
                "tripmates_generations": (
                    tripmates_result
                    .generations_executed
                ),
                "adaptive_convergence_generation": (
                    no_thermal_result
                    .convergence_generation
                ),
                "tripmates_convergence_generation": (
                    tripmates_result
                    .convergence_generation
                ),
            }
        )

    # ------------------------------------------------------------
    # Save and summarise
    # ------------------------------------------------------------

    results_df = pd.DataFrame(
        result_rows
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        OUTPUT_DIRECTORY
        / (
            f"seed_{TEST_SEED}"
            "_weather_ablation.csv"
        )
    )

    results_df.to_csv(
        output_file,
        index=False,
        float_format="%.6f",
    )

    wes_reductions = (
        results_df[
            "wes_reduction_percent"
        ].tolist()
    )

    route_change_count = int(
        results_df[
            "route_changed"
        ].sum()
    )

    utility_failure_count = int(
        (
            ~results_df[
                "utility_guardrail_passed"
            ]
        ).sum()
    )

    visit_failure_count = int(
        (
            ~results_df[
                "visit_guardrail_passed"
            ]
        ).sum()
    )

    print()
    print("=" * 100)
    print(
        "WEATHER SMOKE-TEST SUMMARY"
    )
    print("=" * 100)

    print(
        "Test seed:",
        TEST_SEED,
    )

    print(
        "Weather instances:",
        len(results_df),
    )

    print(
        "Changed routes:",
        route_change_count,
    )

    print(
        "Median WES reduction:",
        f"{statistics.median(wes_reductions):.2f}%",
    )

    print(
        "Mean WES reduction:",
        f"{statistics.mean(wes_reductions):.2f}%",
    )

    print(
        "Minimum WES reduction:",
        f"{min(wes_reductions):.2f}%",
    )

    print(
        "Maximum WES reduction:",
        f"{max(wes_reductions):.2f}%",
    )

    print(
        "Utility guardrail failures:",
        utility_failure_count,
    )

    print(
        "Visit guardrail failures:",
        visit_failure_count,
    )

    print()
    print(
        results_df[
            [
                "instance_id",
                "route_changed",
                "adaptive_wes",
                "tripmates_wes",
                "wes_reduction_percent",
                "utility_loss_percent",
                "completed_visit_loss",
                "travel_change_min",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Output file:",
        output_file.resolve(),
    )

    if route_change_count == 0:
        print()
        print(
            "NOTICE: No route changed for this seed. "
            "This is not automatically an error, but "
            "thermal Pareto-front diversity should be "
            "inspected before the complete experiment."
        )


if __name__ == "__main__":
    main()