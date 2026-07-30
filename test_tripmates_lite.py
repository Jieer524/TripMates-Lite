"""
Functional comparison of Adaptive Without Thermal and TripMates Lite.
"""

from __future__ import annotations

import json
from pathlib import Path

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

TEST_SEED = 1001


def get_instance(
    instance_id: str,
):
    matches = [
        instance
        for instance in (
            build_scenario_instances()
        )
        if instance.instance_id
        == instance_id
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Scenario not found: {instance_id}"
        )

    return matches[0]


def prepare_case(
    data,
    initial_route,
    instance_id,
):
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
            "Initial Static Utility route "
            "is infeasible."
        )

    scenario_instance = (
        get_instance(instance_id)
    )

    state = simulate_replanning_state(
        data=data,
        initial_schedule=initial_schedule,
        scenario_instance=scenario_instance,
    )

    adaptive_problem = (
        prepare_adaptive_problem(
            data=data,
            initial_schedule=initial_schedule,
            state=state,
            scenario_instance=scenario_instance,
            thermal_aware_candidate_window=False,
        )
    )

    tripmates_problem = (
        prepare_adaptive_problem(
            data=data,
            initial_schedule=initial_schedule,
            state=state,
            scenario_instance=scenario_instance,
            thermal_aware_candidate_window=True,
        )
    )

    # Confirm that the correct mode was assigned.
    if (
        adaptive_problem
        .thermal_aware_candidate_window
    ):
        raise RuntimeError(
            "Adaptive Without Thermal incorrectly "
            "uses a thermal candidate window."
        )

    if not (
        tripmates_problem
        .thermal_aware_candidate_window
    ):
        raise RuntimeError(
            "TripMates Lite did not receive "
            "a thermal-aware candidate window."
        )

    return (
        initial_schedule,
        state,
        adaptive_problem,
        tripmates_problem,
    )

def print_metrics(
    name: str,
    solution,
    exposure,
) -> None:
    print()
    print(name)
    print("-" * 80)

    print(
        "Route:",
        " -> ".join(
            str(poi_id)
            for poi_id in (
                solution.route
            )
        ),
    )

    print(
        "Completed visits:",
        solution.completed_poi_count,
    )

    print(
        "Achieved utility:",
        f"{solution.achieved_utility:.6f}",
    )

    print(
        "Executed travel:",
        f"{solution.total_travel_min:.6f}",
    )

    print(
        "OWM:",
        f"{exposure.outdoor_walking_min:.6f}",
    )

    print(
        "OVM:",
        f"{exposure.outdoor_visit_min:.6f}",
    )

    print(
        "TOEM:",
        f"{exposure.total_outdoor_exposure_min:.6f}",
    )

    print(
        "WES:",
        f"{exposure.weather_weighted_exposure_score:.6f}",
    )


def main() -> None:
    data = load_planning_data(
        poi_file=POI_FILE,
        matrix_file=MATRIX_FILE,
    )

    routes = json.loads(
        ROUTES_FILE.read_text(
            encoding="utf-8"
        )
    )

    initial_route = routes[
        str(TEST_SEED)
    ]

    # ============================================================
    # Test A: M_s = 1 must produce an exact match
    # ============================================================

    (
        initial_schedule,
        state,
        adaptive_problem,
        tripmates_problem,
    ) = prepare_case(
        data=data,
        initial_route=initial_route,
        instance_id="S2_nominal",
    )

    no_thermal = (
        run_adaptive_without_thermal_ga(
            data=data,
            problem=adaptive_problem,
            seed=TEST_SEED,
            config=(
                REPLANNING_GA_CONFIG
            ),
        )
    )

    tripmates = (
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

    exact_match = (
        no_thermal
        .selected_solution
        .route
        ==
        tripmates
        .selected_solution
        .route

        and no_thermal
        .selected_solution
        .achieved_utility
        ==
        tripmates
        .selected_solution
        .achieved_utility

        and no_thermal
        .selected_solution
        .completed_poi_count
        ==
        tripmates
        .selected_solution
        .completed_poi_count

        and no_thermal
        .selected_solution
        .total_travel_min
        ==
        tripmates
        .selected_solution
        .total_travel_min
    )

    print("=" * 100)
    print("THERMAL-INACTIVE MATCH TEST")
    print("=" * 100)

    print(
        "Scenario: S2_nominal"
    )

    print(
        "Exact planner match:",
        exact_match,
    )

    if not exact_match:
        raise RuntimeError(
            "TripMates Lite did not match "
            "Adaptive Without Thermal when M_s = 1."
        )

    # ============================================================
    # Test B: weather-related comparison
    # ============================================================

    (
        initial_schedule,
        state,
        adaptive_problem,
        tripmates_problem,
    ) = prepare_case(
        data=data,
        initial_route=initial_route,
        instance_id="S1_nominal",
    )

    no_thermal = (
        run_adaptive_without_thermal_ga(
            data=data,
            problem=adaptive_problem,
            seed=TEST_SEED,
            config=(
                REPLANNING_GA_CONFIG
            ),
        )
    )

    tripmates = (
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

    no_thermal_solution = (
        no_thermal.selected_solution
    )

    thermal_solution = (
        tripmates.selected_solution
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
                adaptive_problem
                .scenario_instance
                .scenario
                .thermal_multiplier
            ),
        )
    )

    thermal_exposure = (
        thermal_solution.exposure
    )

    print()
    print("=" * 100)
    print(
        "THERMAL-AWARE COMPARISON "
        "- SEED 1001 - S1_nominal"
    )
    print("=" * 100)

    print(
        "Adaptive candidate window:",
        list(
            adaptive_problem
            .candidate_window
            .all_candidate_ids
        ),
    )

    print(
        "TripMates candidate window:",
        list(
            tripmates_problem
            .candidate_window
            .all_candidate_ids
        ),
    )

    candidate_windows_differ = (
        adaptive_problem
        .candidate_window
        .all_candidate_ids
        !=
        tripmates_problem
        .candidate_window
        .all_candidate_ids
    )

    print(
        "Candidate windows differ:",
        candidate_windows_differ,
    )

    print_metrics(
        name="Adaptive Without Thermal",
        solution=(
            no_thermal_solution
        ),
        exposure=(
            no_thermal_exposure
        ),
    )

    print_metrics(
        name="TripMates Lite",
        solution=(
            thermal_solution
        ),
        exposure=(
            thermal_exposure
        ),
    )

    wes_reduction_percent = (
        (
            no_thermal_exposure
            .weather_weighted_exposure_score
            - thermal_exposure
            .weather_weighted_exposure_score
        )
        /
        no_thermal_exposure
        .weather_weighted_exposure_score
        * 100.0
    )

    utility_loss_percent = (
        (
            no_thermal_solution
            .achieved_utility
            - thermal_solution
            .achieved_utility
        )
        /
        no_thermal_solution
        .achieved_utility
        * 100.0
    )

    visit_loss = (
        no_thermal_solution
        .completed_poi_count
        - thermal_solution
        .completed_poi_count
    )

    print()
    print("PAIRED DIFFERENCES")
    print("-" * 80)

    print(
        "WES reduction:",
        f"{wes_reduction_percent:.2f}%",
    )

    print(
        "Utility loss:",
        f"{utility_loss_percent:.2f}%",
    )

    print(
        "Completed-visit loss:",
        visit_loss,
    )

    print(
        "Utility guardrail passed:",
        utility_loss_percent <= 10.0,
    )

    print(
        "Visit guardrail passed:",
        visit_loss <= 1,
    )


if __name__ == "__main__":
    main()