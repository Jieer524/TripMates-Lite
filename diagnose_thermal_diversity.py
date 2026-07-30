"""
Diagnose why TripMates Lite and Adaptive Without Thermal select
the same route.

This is a diagnostic script only. It does not create final
experimental results or modify the frozen GA configuration.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from adaptive_ga_core import (
    REPLANNING_GA_CONFIG,
    decode_adaptive_chromosome,
    initialise_adaptive_population,
    prepare_adaptive_problem,
    run_adaptive_without_thermal_ga,
)

from exposure_core import (
    calculate_exposed_fraction,
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

TEST_SEED = 1001
TEST_INSTANCE_ID = "S1_nominal"

# This is diagnostic sampling, not an experimental GA parameter.
SAMPLED_CHROMOSOME_COUNT = 2000


# ================================================================
# Helpers
# ================================================================

def find_instance(
    instance_id: str,
):
    matches = [
        instance
        for instance in build_scenario_instances()
        if instance.instance_id == instance_id
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Could not uniquely find {instance_id}."
        )

    return matches[0]


def route_dominates(
    first: dict,
    second: dict,
) -> bool:
    """
    Thermal objective vector:

    - maximise utility;
    - maximise completed visits;
    - minimise travel;
    - minimise WES.
    """

    no_worse = (
        first["utility"]
        >= second["utility"]
        and first["completed_count"]
        >= second["completed_count"]
        and first["travel_min"]
        <= second["travel_min"]
        and first["wes"]
        <= second["wes"]
    )

    strictly_better = (
        first["utility"]
        > second["utility"]
        or first["completed_count"]
        > second["completed_count"]
        or first["travel_min"]
        < second["travel_min"]
        or first["wes"]
        < second["wes"]
    )

    return no_worse and strictly_better


def calculate_first_front(
    records: list[dict],
) -> list[dict]:
    first_front = []

    for candidate_index, candidate in enumerate(
        records
    ):
        dominated = False

        for comparison_index, comparison in enumerate(
            records
        ):
            if candidate_index == comparison_index:
                continue

            if route_dominates(
                comparison,
                candidate,
            ):
                dominated = True
                break

        if not dominated:
            first_front.append(candidate)

    return first_front


def calculate_eligible_routes(
    first_front: list[dict],
) -> list[dict]:
    if not first_front:
        raise RuntimeError(
            "No sampled first Pareto front exists."
        )

    maximum_utility = max(
        record["utility"]
        for record in first_front
    )

    maximum_count = max(
        record["completed_count"]
        for record in first_front
    )

    return [
        record
        for record in first_front
        if (
            record["utility"]
            >= 0.90 * maximum_utility
            and record["completed_count"]
            >= maximum_count - 1
        )
    ]


def format_route(
    route: tuple[int, ...],
) -> str:
    return " -> ".join(
        str(poi_id)
        for poi_id in route
    )


# ================================================================
# Main
# ================================================================

def main() -> None:
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
            "Initial route is infeasible."
        )

    scenario_instance = find_instance(
        TEST_INSTANCE_ID
    )

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

    optional_ids = list(
        problem
        .candidate_window
        .optional_ids
    )

    multiplier = (
        scenario_instance
        .scenario
        .thermal_multiplier
    )

    adjustments = (
        scenario_instance
        .scenario
        .build_travel_adjustments(data)
    )

    # ============================================================
    # Candidate exposure profile
    # ============================================================

    print("=" * 110)
    print("CANDIDATE THERMAL PROFILE")
    print("=" * 110)

    print(
        f"Scenario: {TEST_INSTANCE_ID}"
    )

    print(
        f"Current POI: {state.current_poi_id}"
    )

    print(
        f"Thermal multiplier: {multiplier}"
    )

    print(
        "Candidate window:",
        list(
            problem
            .candidate_window
            .all_candidate_ids
        ),
    )

    print()
    print(
        f"{'ID':>3}  "
        f"{'Mandatory':>9}  "
        f"{'Utility':>9}  "
        f"{'Visit':>7}  "
        f"{'Indoor':>7}  "
        f"{'Shelter':>8}  "
        f"{'Exposed visit':>14}  "
        f"{'Travel from current':>19}  "
        f"{'Local WES proxy':>15}"
    )

    print("-" * 110)

    mandatory_set = set(
        state.remaining_mandatory_ids
    )

    for poi_id in (
        problem
        .candidate_window
        .all_candidate_ids
    ):
        poi = data.pois[poi_id]

        exposed_fraction = (
            calculate_exposed_fraction(
                indoor_ratio=poi.indoor_ratio,
                shelter_ratio=poi.shelter_ratio,
            )
        )

        exposed_visit = (
            poi.visit_duration_min
            * exposed_fraction
        )

        if poi_id == state.current_poi_id:
            travel_from_current = 0.0
        else:
            base_travel = (
                data.get_travel_time(
                    state.current_poi_id,
                    poi_id,
                )
            )

            travel_from_current = (
                adjustments.apply(
                    state.current_poi_id,
                    poi_id,
                    base_travel,
                )
            )

        # This is for diagnosis only. It is not the final
        # route objective because the complete route matters.
        local_wes_proxy = (
            multiplier
            * (
                travel_from_current
                + exposed_visit
            )
        )

        print(
            f"{poi_id:>3}  "
            f"{str(poi_id in mandatory_set):>9}  "
            f"{poi.utility_score:>9.4f}  "
            f"{poi.visit_duration_min:>7.1f}  "
            f"{poi.indoor_ratio:>7.2f}  "
            f"{poi.shelter_ratio:>8.2f}  "
            f"{exposed_visit:>14.4f}  "
            f"{travel_from_current:>19.4f}  "
            f"{local_wes_proxy:>15.4f}"
        )

    # ============================================================
    # Generate diagnostic chromosome sample
    # ============================================================

    random_generator = random.Random(
        TEST_SEED
    )

    chromosomes: set[
        tuple[int, ...]
    ] = set()

    initial_population = (
        initialise_adaptive_population(
            data=data,
            problem=problem,
            config=REPLANNING_GA_CONFIG,
            random_generator=(
                random_generator
            ),
        )
    )

    chromosomes.update(
        initial_population
    )

    maximum_permutations = math.factorial(
        len(optional_ids)
    )

    target_count = min(
        SAMPLED_CHROMOSOME_COUNT,
        maximum_permutations,
    )

    while len(chromosomes) < target_count:
        chromosomes.add(
            tuple(
                random_generator.sample(
                    optional_ids,
                    len(optional_ids),
                )
            )
        )

    print()
    print("=" * 110)
    print("DECODING DIAGNOSTIC CHROMOSOMES")
    print("=" * 110)

    print(
        "Optional genes:",
        optional_ids,
    )

    print(
        "Total possible permutations:",
        maximum_permutations,
    )

    print(
        "Chromosomes sampled:",
        len(chromosomes),
    )

    # ============================================================
    # Decode and deduplicate routes
    # ============================================================

    routes_by_sequence: dict[
        tuple[int, ...],
        dict,
    ] = {}

    for index, chromosome in enumerate(
        sorted(chromosomes)
    ):
        solution = decode_adaptive_chromosome(
            data=data,
            problem=problem,
            chromosome=chromosome,
        )

        exposure = calculate_exposure_metrics(
            data=data,
            initial_schedule=initial_schedule,
            state=state,
            repaired_schedule=(
                solution.schedule
            ),
            thermal_multiplier=multiplier,
        )

        route_record = {
            "route": solution.route,
            "chromosome": chromosome,

            "utility": (
                solution.achieved_utility
            ),

            "completed_count": (
                solution.completed_poi_count
            ),

            "travel_min": (
                solution.total_travel_min
            ),

            "owm": (
                exposure.outdoor_walking_min
            ),

            "ovm": (
                exposure.outdoor_visit_min
            ),

            "toem": (
                exposure.total_outdoor_exposure_min
            ),

            "wes": (
                exposure
                .weather_weighted_exposure_score
            ),
        }

        routes_by_sequence.setdefault(
            solution.route,
            route_record,
        )

        if (
            (index + 1) % 250 == 0
            or index + 1 == len(chromosomes)
        ):
            print(
                f"Decoded {index + 1}/"
                f"{len(chromosomes)} chromosomes"
            )

    unique_routes = list(
        routes_by_sequence.values()
    )

    first_front = calculate_first_front(
        unique_routes
    )

    eligible_routes = (
        calculate_eligible_routes(
            first_front
        )
    )

    eligible_sorted = sorted(
        eligible_routes,
        key=lambda record: (
            record["wes"],
            record["travel_min"],
            -record["utility"],
            record["route"],
        ),
    )

    sampled_thermal_best = (
        eligible_sorted[0]
    )

    # ============================================================
    # Run actual planners
    # ============================================================

    adaptive_result = (
        run_adaptive_without_thermal_ga(
            data=data,
            problem=problem,
            seed=TEST_SEED,
            config=REPLANNING_GA_CONFIG,
        )
    )

    tripmates_result = (
        run_tripmates_lite_ga(
            data=data,
            initial_schedule=initial_schedule,
            problem=problem,
            seed=TEST_SEED,
            config=REPLANNING_GA_CONFIG,
        )
    )

    adaptive_route = (
        adaptive_result
        .selected_solution
        .route
    )

    tripmates_route = (
        tripmates_result
        .selected_solution
        .route
    )

    # ============================================================
    # Diagnostic results
    # ============================================================

    all_wes_values = [
        record["wes"]
        for record in unique_routes
    ]

    eligible_wes_values = [
        record["wes"]
        for record in eligible_routes
    ]

    print()
    print("=" * 110)
    print("THERMAL DIVERSITY DIAGNOSTIC")
    print("=" * 110)

    print(
        "Unique decoded routes:",
        len(unique_routes),
    )

    print(
        "Sampled first-front routes:",
        len(first_front),
    )

    print(
        "Guardrail-eligible routes:",
        len(eligible_routes),
    )

    print(
        "All-route WES range:",
        f"{min(all_wes_values):.6f}",
        "to",
        f"{max(all_wes_values):.6f}",
    )

    print(
        "Eligible-route WES range:",
        f"{min(eligible_wes_values):.6f}",
        "to",
        f"{max(eligible_wes_values):.6f}",
    )

    print()
    print(
        "Actual Adaptive route:",
        format_route(adaptive_route),
    )

    print(
        "Actual TripMates route:",
        format_route(tripmates_route),
    )

    print(
        "Sampled best thermal route:",
        format_route(
            sampled_thermal_best[
                "route"
            ]
        ),
    )

    print()
    print(
        "Sampled best thermal WES:",
        f"{sampled_thermal_best['wes']:.6f}",
    )

    print(
        "Sampled best thermal utility:",
        f"{sampled_thermal_best['utility']:.6f}",
    )

    print(
        "Sampled best completed count:",
        sampled_thermal_best[
            "completed_count"
        ],
    )

    print(
        "Sampled best travel:",
        f"{sampled_thermal_best['travel_min']:.6f}",
    )

    print()
    print("=" * 110)
    print("TOP GUARDRAIL-ELIGIBLE ROUTES BY WES")
    print("=" * 110)

    for rank, record in enumerate(
        eligible_sorted[:15],
        start=1,
    ):
        print(
            f"{rank:>2}. "
            f"WES={record['wes']:.6f}, "
            f"U={record['utility']:.6f}, "
            f"N={record['completed_count']}, "
            f"TTT={record['travel_min']:.6f}, "
            f"route={list(record['route'])}"
        )

    print()
    print("=" * 110)
    print("INTERPRETATION FLAGS")
    print("=" * 110)

    print(
        "Multiple decoded routes exist:",
        len(unique_routes) > 1,
    )

    print(
        "Eligible WES diversity exists:",
        (
            max(eligible_wes_values)
            - min(eligible_wes_values)
        ) > 1e-9,
    )

    print(
        "Sampled thermal best differs "
        "from Adaptive:",
        (
            sampled_thermal_best["route"]
            != adaptive_route
        ),
    )

    print(
        "Actual TripMates found sampled "
        "thermal best:",
        (
            tripmates_route
            == sampled_thermal_best["route"]
        ),
    )


if __name__ == "__main__":
    main()