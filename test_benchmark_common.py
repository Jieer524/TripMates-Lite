"""
Tests for the shared benchmark helpers and the static-suffix
evaluation used by the RQ1 runner.
"""

from __future__ import annotations

import json

from benchmark_common import (
    ALL_INSTANCE_IDS,
    C2_ALPHAS,
    build_metric_record,
    build_scenario_lookup,
    encode_ids,
    evaluate_static_suffix,
    safe_ratio,
)
from planning_core import (
    PlanningConstraints,
    load_planning_data,
)
from scenario_core import simulate_replanning_state


POI_FILE = "poi_data_calculated_final.csv"
MATRIX_FILE = "walking_time_matrix_final.csv"


def main() -> None:
    # ------------------------------------------------------------
    # Static enumeration
    # ------------------------------------------------------------

    if len(ALL_INSTANCE_IDS) != 18:
        raise RuntimeError(
            "The official scenario set must contain 18 instances."
        )

    if len(set(ALL_INSTANCE_IDS)) != 18:
        raise RuntimeError("Scenario instance IDs must be unique.")

    if C2_ALPHAS != (1.00, 0.95, 0.90, 0.85):
        raise RuntimeError("C2 alphas must be 1.00/0.95/0.90/0.85.")

    # ------------------------------------------------------------
    # Encoding / ratio helpers
    # ------------------------------------------------------------

    if json.loads(encode_ids([1, 2, 3])) != [1, 2, 3]:
        raise RuntimeError("encode_ids must round-trip.")

    if safe_ratio(1.0, 0.0) is not None:
        raise RuntimeError("safe_ratio must return None for zero denom.")

    if safe_ratio(1.0, 4.0) != 0.25:
        raise RuntimeError("safe_ratio must compute a plain ratio.")

    # ------------------------------------------------------------
    # Static-suffix evaluation is deterministic and structured
    # ------------------------------------------------------------

    data = load_planning_data(
        poi_file=POI_FILE,
        matrix_file=MATRIX_FILE,
    )

    routes = json.loads(
        open(
            "results/static_utility/initial_routes_by_seed.json",
            encoding="utf-8",
        ).read()
    )

    from planning_core import decode_route

    initial_route = routes["1001"]
    initial_schedule = decode_route(
        data=data,
        visit_sequence=initial_route,
        start_poi_id=1,
        start_time_min=PlanningConstraints().start_time_min,
        trip_end_min=PlanningConstraints().trip_end_min,
        day_code=PlanningConstraints().day_code,
        cost_budget_rm=PlanningConstraints().cost_budget_rm,
        start_poi_is_visit=True,
        required_mandatory_ids=data.mandatory_ids,
    )

    scenario_lookup = build_scenario_lookup()
    constraints = PlanningConstraints()

    state = simulate_replanning_state(
        data=data,
        initial_schedule=initial_schedule,
        scenario_instance=scenario_lookup["S1_nominal"],
    )

    first = evaluate_static_suffix(
        data=data,
        state=state,
        scenario_instance=scenario_lookup["S1_nominal"],
        constraints=constraints,
    )
    second = evaluate_static_suffix(
        data=data,
        state=state,
        scenario_instance=scenario_lookup["S1_nominal"],
        constraints=constraints,
    )

    if first != second:
        raise RuntimeError(
            "Static-suffix evaluation must be deterministic."
        )

    if first.feasible and first.schedule is None:
        raise RuntimeError(
            "A feasible static suffix must carry a schedule."
        )

    # ------------------------------------------------------------
    # Metric record schema
    # ------------------------------------------------------------

    record = build_metric_record(
        planner="Test Planner",
        scenario_instance=scenario_lookup["S1_nominal"],
        seed=1001,
        initial_route=initial_route,
        initial_schedule=initial_schedule,
        initial_route_utility=10.0,
        state=state,
        repaired_route=list(first.repaired_route),
        achieved_utility=8.0,
        completed_poi_count=5,
        total_travel_min=12.0,
        locked_travel_min=3.0,
        repaired_schedule=first.schedule,
        exposure=None,
        feasible=first.feasible,
        failure_reason=first.failure_reason,
        mandatory_complete=True,
        runtime_seconds=0.01,
        deterministic=True,
    )

    required_fields = {
        "planner",
        "scenario_id",
        "instance_id",
        "trigger_variant",
        "seed",
        "initial_route",
        "final_route",
        "feasible",
        "mandatory_complete",
        "itinerary_completion_ratio",
        "original_poi_retention",
        "achieved_utility",
        "utility_retention_ratio",
        "completed_poi_count",
        "total_travel_min",
        "travel_overhead_min",
        "total_cost_rm",
        "outdoor_walking_min",
        "outdoor_visit_min",
        "total_outdoor_exposure_min",
        "wes",
        "route_changed",
        "removed_poi_count",
        "added_poi_count",
        "retained_poi_count",
        "runtime_seconds",
    }

    missing = required_fields - set(record)

    if missing:
        raise RuntimeError(
            f"Metric record is missing fields: {sorted(missing)}"
        )

    print("PASS: benchmark helpers and metric schema are valid.")


if __name__ == "__main__":
    main()
