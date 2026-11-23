"""Regression checks for the seeded, bounded single-route ILS adaptation."""

from __future__ import annotations

from benchmark_common import build_initial_schedules, build_scenario_lookup, load_initial_routes
from adaptive_ga_core import prepare_adaptive_problem
from ils_repair_baseline import run_ils_repair
from planning_core import PlanningConstraints, load_planning_data
from scenario_core import simulate_replanning_state


def test_ils_is_seeded_feasible_and_preserves_the_locked_prefix():
    data = load_planning_data("poi_data_calculated_final.csv", "walking_time_matrix_final.csv")
    routes = load_initial_routes()
    schedule = build_initial_schedules(data, routes)[1001]
    scenario = build_scenario_lookup()["S3_nominal"]
    state = simulate_replanning_state(data=data, initial_schedule=schedule, scenario_instance=scenario)
    problem = prepare_adaptive_problem(
        data=data, initial_schedule=schedule, state=state,
        scenario_instance=scenario, constraints=PlanningConstraints(),
    )
    first = run_ils_repair(data=data, problem=problem, seed=1001)
    second = run_ils_repair(data=data, problem=problem, seed=1001)
    assert first.route == second.route
    assert first.schedule.feasible
    assert set(state.remaining_mandatory_ids) <= set(first.route)
    assert len(first.route) <= problem.bounded_repair_config.max_modifiable_suffix_activities
    assert first.repair_record.locked_prefix == state.completed_poi_ids
    assert first.repair_record.failure_status == "success"
