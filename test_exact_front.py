"""Contract tests for exact candidate-window enumeration."""

from pathlib import Path

from adaptive_ga_core import REPLANNING_GA_CONFIG, prepare_adaptive_problem, run_adaptive_without_thermal_ga
from benchmark_common import build_initial_schedules, build_scenario_lookup, load_initial_routes
from exact_front import enumerate_exact_front
from planning_core import load_planning_data
from scenario_core import simulate_replanning_state
from tripmates_lite_ga import ThermalEvaluatedSolution, ThermalSelectionPolicy
from exposure_core import calculate_exposure_metrics


def main() -> None:
    data = load_planning_data(
        poi_file=Path("poi_data_calculated_final.csv"),
        matrix_file=Path("walking_time_matrix_final.csv"),
    )
    schedules = build_initial_schedules(data, load_initial_routes())
    scenario = build_scenario_lookup()["S1_nominal"]
    initial = schedules[1001]
    state = simulate_replanning_state(data=data, initial_schedule=initial, scenario_instance=scenario)
    problem = prepare_adaptive_problem(data=data, initial_schedule=initial, state=state, scenario_instance=scenario)
    baseline = run_adaptive_without_thermal_ga(data=data, problem=problem, seed=1001, config=REPLANNING_GA_CONFIG)
    multiplier = scenario.scenario.thermal_multiplier
    reference = ThermalEvaluatedSolution(
        chromosome=baseline.selected_solution.chromosome,
        route=baseline.selected_solution.route,
        schedule=baseline.selected_solution.schedule,
        achieved_utility=baseline.selected_solution.achieved_utility,
        completed_poi_count=baseline.selected_solution.completed_poi_count,
        total_travel_min=baseline.selected_solution.total_travel_min,
        exposure=calculate_exposure_metrics(data=data, initial_schedule=initial, state=state, repaired_schedule=baseline.selected_solution.schedule, thermal_multiplier=multiplier),
    )
    result = enumerate_exact_front(
        data=data,
        initial_schedule=initial,
        problem=problem,
        thermal_multiplier=multiplier,
        non_thermal_reference=reference,
        selection_policy=ThermalSelectionPolicy(alpha=0.90, utility_guardrail_baseline="front_maximum", priority="c2"),
    )
    if not result.all_solutions or not result.first_front:
        raise AssertionError("Exact enumeration must produce a non-empty population and front.")
    if result.selection.selected_solution not in result.first_front and not result.selection.guardrail_fallback_used:
        raise AssertionError("Non-fallback selection must come from the exact first front.")
    print(f"PASS: {len(result.all_solutions)} exact routes, {len(result.first_front)} first-front routes, {result.state_count} DP states.")


if __name__ == "__main__":
    main()
