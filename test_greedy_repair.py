"""
Validation tests for the deterministic Greedy Repair baseline.

Covers:
- same input -> identical route and metrics;
- locked prefix is never reordered or deleted;
- mandatory POIs are retained when feasible;
- an infeasible mandatory suffix yields a structured failure;
- opening hours, budget, horizon, duplicates and connectivity hold;
- WES / thermal information is not used as a selection input;
- the result is evaluable by the common exposure pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

from exposure_core import calculate_exposure_metrics
from greedy_repair import run_greedy_repair
from planning_core import (
    COST_BUDGET_RM,
    EXPERIMENT_DAY,
    TRIP_END_MIN,
    TRIP_START_MIN,
    decode_route,
    load_planning_data,
)
from scenario_core import (
    ScenarioDefinition,
    ScenarioInstance,
    build_scenario_instances,
    simulate_replanning_state,
)


POI_FILE = Path("poi_data_calculated_final.csv")
MATRIX_FILE = Path("walking_time_matrix_final.csv")
ROUTES_FILE = Path(
    "results/static_utility/initial_routes_by_seed.json"
)

TEST_SEED = 1001
TEST_INSTANCES = ("S1_nominal", "S3_nominal", "S4_nominal", "S6_nominal")


def build_initial(data, initial_route):
    schedule = decode_route(
        data=data,
        visit_sequence=initial_route,
        start_poi_id=1,
        start_time_min=TRIP_START_MIN,
        trip_end_min=TRIP_END_MIN,
        day_code=EXPERIMENT_DAY,
        cost_budget_rm=COST_BUDGET_RM,
        start_poi_is_visit=True,
        required_mandatory_ids=data.mandatory_ids,
    )

    if not schedule.feasible:
        raise RuntimeError("The initial route is infeasible.")

    return schedule


def run_case(data, initial_schedule, instance_id):
    scenario_instance = next(
        instance
        for instance in build_scenario_instances()
        if instance.instance_id == instance_id
    )

    state = simulate_replanning_state(
        data=data,
        initial_schedule=initial_schedule,
        scenario_instance=scenario_instance,
    )

    result = run_greedy_repair(
        data=data,
        initial_schedule=initial_schedule,
        state=state,
        scenario_instance=scenario_instance,
        seed=TEST_SEED,
    )

    return scenario_instance, state, result


def main() -> None:
    data = load_planning_data(
        poi_file=POI_FILE,
        matrix_file=MATRIX_FILE,
    )

    routes = json.loads(
        ROUTES_FILE.read_text(encoding="utf-8")
    )
    initial_route = routes[str(TEST_SEED)]
    initial_schedule = build_initial(data, initial_route)

    for instance_id in TEST_INSTANCES:
        scenario_instance, state, result = run_case(
            data, initial_schedule, instance_id
        )

        if result.status != "success" or result.selected_solution is None:
            raise RuntimeError(
                f"{instance_id}: expected a successful repair, "
                f"got status={result.status} ({result.failure_reason})."
            )

        solution = result.selected_solution
        schedule = solution.schedule

        # --------------------------------------------------------
        # Hard-constraint validity
        # --------------------------------------------------------

        if not schedule.feasible:
            raise RuntimeError(
                f"{instance_id}: repaired schedule is infeasible."
            )

        if schedule.missing_mandatory_ids:
            raise RuntimeError(
                f"{instance_id}: missing mandatory POIs "
                f"{list(schedule.missing_mandatory_ids)}."
            )

        final_route = (
            list(state.completed_poi_ids)
            + list(solution.route)
        )

        if len(final_route) != len(set(final_route)):
            raise RuntimeError(
                f"{instance_id}: duplicate POI visits in final route."
            )

        if set(solution.route) & set(state.unavailable_poi_ids):
            raise RuntimeError(
                f"{instance_id}: unavailable POI appears in repair."
            )

        if schedule.total_cost_rm > COST_BUDGET_RM + 1e-9:
            raise RuntimeError(
                f"{instance_id}: cost exceeds budget."
            )

        if schedule.finish_min > TRIP_END_MIN + 1e-9:
            raise RuntimeError(
                f"{instance_id}: route finishes after 17:00."
            )

        # --------------------------------------------------------
        # Locked prefix is preserved exactly
        # --------------------------------------------------------

        completed = list(state.completed_poi_ids)
        if completed != list(initial_route)[: len(completed)]:
            raise RuntimeError(
                f"{instance_id}: completed prefix was reordered."
            )

        if set(completed) & set(solution.route):
            raise RuntimeError(
                f"{instance_id}: repaired route revisits a locked POI."
            )

        # --------------------------------------------------------
        # Mandatory POIs retained
        # --------------------------------------------------------

        if not set(state.remaining_mandatory_ids) <= set(solution.route):
            raise RuntimeError(
                f"{instance_id}: outstanding mandatory POIs were dropped."
            )

        # --------------------------------------------------------
        # Determinism: same input -> identical result
        # --------------------------------------------------------

        _, _, repeat = run_case(
            data, initial_schedule, instance_id
        )

        if repeat.selected_solution.route != solution.route:
            raise RuntimeError(
                f"{instance_id}: greedy route is not deterministic."
            )

        if (
            repeat.selected_solution.achieved_utility
            != solution.achieved_utility
            or repeat.selected_solution.total_travel_min
            != solution.total_travel_min
            or repeat.selected_solution.completed_poi_count
            != solution.completed_poi_count
        ):
            raise RuntimeError(
                f"{instance_id}: greedy metrics are not deterministic."
            )

        # --------------------------------------------------------
        # Common exposure pipeline can evaluate the result
        # --------------------------------------------------------

        exposure = calculate_exposure_metrics(
            data=data,
            initial_schedule=initial_schedule,
            state=state,
            repaired_schedule=schedule,
            thermal_multiplier=(
                scenario_instance.scenario.thermal_multiplier
            ),
        )

        if exposure.weather_weighted_exposure_score < 0:
            raise RuntimeError(
                f"{instance_id}: exposure pipeline returned a "
                "negative WES."
            )

        print(
            f"{instance_id}: route={list(solution.route)} "
            f"utility={solution.achieved_utility:.4f} "
            f"visits={solution.completed_poi_count} "
            f"WES={exposure.weather_weighted_exposure_score:.4f}"
        )

    # ------------------------------------------------------------
    # WES / thermal independence: changing the thermal multiplier
    # must not change the greedy route.
    # ------------------------------------------------------------

    base_scenario = next(
        instance
        for instance in build_scenario_instances()
        if instance.instance_id == "S1_nominal"
    )

    _, base_state, base_result = run_case(
        data, initial_schedule, "S1_nominal"
    )

    hot_scenario = ScenarioInstance(
        scenario=ScenarioDefinition(
            scenario_id="S1_hot_probe",
            name="Heavy Rain (probe)",
            nominal_trigger_offset_min=120,
            thermal_multiplier=5.0,
        ),
        trigger_variant="nominal",
        trigger_offset_min=120,
        trigger_absolute_min=(
            TRIP_START_MIN + 120
        ),
    )

    hot_state = simulate_replanning_state(
        data=data,
        initial_schedule=initial_schedule,
        scenario_instance=hot_scenario,
    )

    hot_result = run_greedy_repair(
        data=data,
        initial_schedule=initial_schedule,
        state=hot_state,
        scenario_instance=hot_scenario,
        seed=TEST_SEED,
    )

    if (
        hot_result.selected_solution is None
        or base_result.selected_solution is None
        or hot_result.selected_solution.route
        != base_result.selected_solution.route
    ):
        raise RuntimeError(
            "Greedy Repair changed its route when only the thermal "
            "multiplier changed; it must not use WES or thermal input."
        )

    # ------------------------------------------------------------
    # Infeasible mandatory suffix -> structured failure, not a
    # fabricated route.
    # ------------------------------------------------------------

    impossible_scenario = ScenarioInstance(
        scenario=ScenarioDefinition(
            scenario_id="SX_infeasible",
            name="Extreme movement delay",
            nominal_trigger_offset_min=120,
            global_travel_multiplier=40.0,
        ),
        trigger_variant="nominal",
        trigger_offset_min=120,
        trigger_absolute_min=(
            TRIP_START_MIN + 120
        ),
    )

    impossible_state = simulate_replanning_state(
        data=data,
        initial_schedule=initial_schedule,
        scenario_instance=impossible_scenario,
    )

    impossible_result = run_greedy_repair(
        data=data,
        initial_schedule=initial_schedule,
        state=impossible_state,
        scenario_instance=impossible_scenario,
        seed=TEST_SEED,
    )

    if (
        impossible_result.status != "infeasible"
        or impossible_result.selected_solution is not None
        or not impossible_result.failure_reason
    ):
        raise RuntimeError(
            "An infeasible mandatory suffix must produce a structured "
            "failure with no fabricated route."
        )

    print(
        "Infeasible suffix reported structurally: "
        f"{impossible_result.failure_reason}"
    )

    print("PASS: Greedy Repair obeys the shared hard constraints.")


if __name__ == "__main__":
    main()
