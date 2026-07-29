"""
Test Adaptive Without Thermal for one seed and scenario.

Initial test:
    Seed: 1001
    Scenario: S2 nominal
"""

from __future__ import annotations

import json
from pathlib import Path

from adaptive_ga_core import (
    REPLANNING_GA_CONFIG,
    prepare_adaptive_problem,
    run_adaptive_without_thermal_ga,
)

from planning_core import (
    COST_BUDGET_RM,
    EXPERIMENT_DAY,
    TRIP_END_MIN,
    TRIP_START_MIN,
    decode_route,
    load_planning_data,
    print_schedule,
)

from scenario_core import (
    build_scenario_instances,
    simulate_replanning_state,
)


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

TEST_SEED = 1001
TEST_INSTANCE_ID = "S2_nominal"


def get_scenario_instance(
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
            f"Could not uniquely find "
            f"scenario {instance_id}."
        )

    return matches[0]


def main() -> None:
    data = load_planning_data(
        poi_file=POI_FILE,
        matrix_file=MATRIX_FILE,
    )

    routes_by_seed = json.loads(
        INITIAL_ROUTES_FILE.read_text(
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
            "The initial Static Utility "
            "route is infeasible."
        )

    scenario_instance = (
        get_scenario_instance(
            TEST_INSTANCE_ID
        )
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

    problem = prepare_adaptive_problem(
        data=data,
        initial_schedule=(
            initial_schedule
        ),
        state=state,
        scenario_instance=(
            scenario_instance
        ),
    )

    result = (
        run_adaptive_without_thermal_ga(
            data=data,
            problem=problem,
            seed=TEST_SEED,
            config=(
                REPLANNING_GA_CONFIG
            ),
        )
    )

    solution = (
        result.selected_solution
    )

    repaired_schedule = (
        solution.schedule
    )

    if not repaired_schedule.feasible:
        raise RuntimeError(
            "The repaired route is infeasible."
        )

    if (
        repaired_schedule
        .missing_mandatory_ids
    ):
        raise RuntimeError(
            "The repaired route omitted "
            "still-feasible mandatory POIs."
        )

    completed_route = list(
        state.completed_poi_ids
    )

    repaired_route = list(
        solution.route
    )

    final_visit_route = (
        completed_route
        + repaired_route
    )

    if len(final_visit_route) != len(
        set(final_visit_route)
    ):
        raise RuntimeError(
            "The combined route contains "
            "duplicate POI visits."
        )

    print("=" * 100)
    print(
        "ADAPTIVE WITHOUT THERMAL "
        f"- SEED {TEST_SEED} "
        f"- {TEST_INSTANCE_ID}"
    )
    print("=" * 100)

    print(
        "Initial route:",
        " -> ".join(
            str(poi_id)
            for poi_id in initial_route
        ),
    )

    print(
        "Trigger:",
        state.trigger_clock,
    )

    print(
        "Committed component:",
        state.committed_component,
    )

    print(
        "Replanning starts:",
        state.replan_start_clock,
    )

    print(
        "Current POI:",
        state.current_poi_id,
    )

    print(
        "Current POI unvisited:",
        state.current_poi_requires_visit,
    )

    print(
        "Locked completed POIs:",
        list(
            state.completed_poi_ids
        ),
    )

    print(
        "Remaining mandatory POIs:",
        list(
            state.remaining_mandatory_ids
        ),
    )

    print(
        "Mandatory backbone:",
        list(
            problem.mandatory_backbone
        ),
    )

    print(
        "Candidate window:",
        list(
            problem
            .candidate_window
            .all_candidate_ids
        ),
    )

    print(
        "Optional candidates:",
        list(
            problem
            .candidate_window
            .optional_ids
        ),
    )

    print("\nPre-screen scores:")

    for poi_id, score in (
        problem
        .candidate_window
        .prescreen_scores
    ):
        print(
            f"  POI {poi_id:>2}: "
            f"{score:.12f}"
        )

    print()
    print(
        "Repaired remaining route:",
        " -> ".join(
            str(poi_id)
            for poi_id in repaired_route
        ),
    )

    print(
        "Final completed-visit route:",
        " -> ".join(
            str(poi_id)
            for poi_id in final_visit_route
        ),
    )

    print(
        "Total completed visits:",
        solution.completed_poi_count,
    )

    print(
        "Total achieved utility:",
        f"{solution.achieved_utility:.6f}",
    )

    print(
        "Total executed travel:",
        f"{solution.total_travel_min:.6f}",
        "min",
    )

    print(
        "Final cost:",
        f"RM{repaired_schedule.total_cost_rm:.2f}",
    )

    print(
        "Final finish time:",
        repaired_schedule.finish_clock,
    )

    print(
        "Generations executed:",
        result.generations_executed,
    )

    print(
        "Convergence generation:",
        result.convergence_generation,
    )

    print(
        "Stopped by stagnation:",
        result.stopped_by_stagnation,
    )

    print(
        "Evaluated chromosomes:",
        result.evaluated_chromosome_count,
    )

    print(
        "Unique decoded routes:",
        result.unique_decoded_route_count,
    )

    print()
    print("REPAIRED-ROUTE SCHEDULE")

    print_schedule(
        repaired_schedule
    )


if __name__ == "__main__":
    main()
    