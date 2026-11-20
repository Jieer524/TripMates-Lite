"""Run the predeclared exact-front thermal-selection ablation.

For every weather-related instance and seed-matched initial-route variant,
this experiment enumerates the finite bounded candidate-window domain with
the production decoder.  It then selects both the non-thermal and thermal
routes from that same exact population.  The primary guardrail is the exact
thermal-front maximum utility; alpha is a predeclared factor.

The GA and five-restart GA are retained only as implementation comparators.
They do not replace the exact-front selector evidence used for RQ2.
"""

from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path

from adaptive_ga_core import REPLANNING_GA_CONFIG, prepare_adaptive_problem
from benchmark_common import (
    MATRIX_FILE,
    POI_FILE,
    SEEDS,
    WEATHER_INSTANCE_IDS,
    build_initial_schedules,
    build_scenario_lookup,
    encode_ids,
    file_fingerprint,
    load_initial_routes,
    utc_timestamp,
)
from exact_front import enumerate_exact_front, select_exact_non_thermal
from planning_core import load_planning_data
from scenario_core import simulate_replanning_state
from tripmates_lite_ga import ThermalSelectionPolicy, evaluate_thermal_selection, run_tripmates_lite_ga


OUTPUT_DIRECTORY = Path("results/exact_front_thermal_ablation")
RAW_FILE = OUTPUT_DIRECTORY / "exact_front_raw_runs.csv"
SUMMARY_FILE = OUTPUT_DIRECTORY / "exact_front_guardrail_summary.csv"
COMPARATOR_FILE = OUTPUT_DIRECTORY / "exact_vs_ga_runtime_quality.csv"
MANIFEST_FILE = OUTPUT_DIRECTORY / "experiment_manifest.json"
ALPHAS = (1.00, 0.95, 0.90, 0.85, 0.80)
RANDOM_RESTARTS = 5


def policy(alpha: float) -> ThermalSelectionPolicy:
    return ThermalSelectionPolicy(
        alpha=alpha,
        visit_guardrail_margin=1,
        utility_guardrail_baseline="front_maximum",
        priority="c2",
    )


def percent_reduction(reference: float, selected: float) -> float:
    return 0.0 if reference == 0 else 100.0 * (reference - selected) / reference


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    data = load_planning_data(poi_file=POI_FILE, matrix_file=MATRIX_FILE)
    initial_schedules = build_initial_schedules(data, load_initial_routes())
    scenario_lookup = build_scenario_lookup()
    raw_rows: list[dict] = []
    comparator_rows: list[dict] = []

    for instance_id in WEATHER_INSTANCE_IDS:
        scenario = scenario_lookup[instance_id]
        multiplier = scenario.scenario.thermal_multiplier
        for seed in SEEDS:
            initial = initial_schedules[seed]
            state = simulate_replanning_state(
                data=data, initial_schedule=initial, scenario_instance=scenario
            )
            problem = prepare_adaptive_problem(
                data=data, initial_schedule=initial, state=state, scenario_instance=scenario
            )

            # A normal GA run supplies a reference for initial enumeration;
            # the exact non-thermal reference below replaces it for selection.
            ga_start = time.perf_counter()
            ga_result = run_tripmates_lite_ga(
                data=data, initial_schedule=initial, problem=problem, seed=seed,
                config=REPLANNING_GA_CONFIG, selection_policy=policy(0.90),
            )
            ga_seconds = time.perf_counter() - ga_start
            exact_start = time.perf_counter()
            exact = enumerate_exact_front(
                data=data, initial_schedule=initial, problem=problem,
                thermal_multiplier=multiplier,
                non_thermal_reference=ga_result.selected_solution,
                selection_policy=policy(0.90),
            )
            exact_seconds = time.perf_counter() - exact_start
            exact_reference = select_exact_non_thermal(exact.all_solutions)

            restart_results = []
            restart_start = time.perf_counter()
            for restart in range(RANDOM_RESTARTS):
                restart_results.append(run_tripmates_lite_ga(
                    data=data, initial_schedule=initial, problem=problem,
                    seed=seed * 100 + restart, config=REPLANNING_GA_CONFIG,
                    selection_policy=policy(0.90),
                ))
            restart_seconds = time.perf_counter() - restart_start
            restart_best = min(
                (result.selected_solution for result in restart_results),
                key=lambda solution: (solution.wes, -solution.achieved_utility, solution.total_travel_min, solution.route),
            )
            comparator_rows.extend([
                {
                    "instance_id": instance_id, "seed": seed, "method": "exact_front",
                    "runtime_seconds": f"{exact_seconds:.6f}", "wes": f"{exact_reference.wes:.6f}",
                    "utility": f"{exact_reference.achieved_utility:.6f}", "visits": exact_reference.completed_poi_count,
                    "travel_min": f"{exact_reference.total_travel_min:.6f}", "route": encode_ids(exact_reference.route),
                    "population_routes": len(exact.all_solutions), "front_routes": len(exact.first_front),
                },
                {
                    "instance_id": instance_id, "seed": seed, "method": "ga",
                    "runtime_seconds": f"{ga_seconds:.6f}", "wes": f"{ga_result.selected_solution.wes:.6f}",
                    "utility": f"{ga_result.selected_solution.achieved_utility:.6f}", "visits": ga_result.selected_solution.completed_poi_count,
                    "travel_min": f"{ga_result.selected_solution.total_travel_min:.6f}", "route": encode_ids(ga_result.selected_solution.route),
                    "population_routes": ga_result.unique_decoded_route_count, "front_routes": len(ga_result.first_front),
                },
                {
                    "instance_id": instance_id, "seed": seed, "method": "random_restart_ga_5",
                    "runtime_seconds": f"{restart_seconds:.6f}", "wes": f"{restart_best.wes:.6f}",
                    "utility": f"{restart_best.achieved_utility:.6f}", "visits": restart_best.completed_poi_count,
                    "travel_min": f"{restart_best.total_travel_min:.6f}", "route": encode_ids(restart_best.route),
                    "population_routes": sum(result.unique_decoded_route_count for result in restart_results), "front_routes": "",
                },
            ])

            population = [__import__("ga_core").Individual(solution=s) for s in exact.all_solutions]
            from tripmates_lite_ga import rank_thermal_population
            rank_thermal_population(population)
            for alpha in ALPHAS:
                chosen = evaluate_thermal_selection(population, exact_reference, policy(alpha))
                selected = chosen.selected_solution
                raw_rows.append({
                    "instance_id": instance_id,
                    "scenario_id": scenario.scenario.scenario_id,
                    "trigger_variant": scenario.trigger_variant,
                    "seed": seed,
                    "alpha": f"{alpha:.2f}",
                    "exact_population_routes": len(exact.all_solutions),
                    "exact_front_routes": len(chosen.first_front),
                    "exact_dp_states": exact.state_count,
                    "exact_transition_count": exact.transition_count,
                    "route_changed": selected.route != exact_reference.route,
                    "reference_route": encode_ids(exact_reference.route),
                    "selected_route": encode_ids(selected.route),
                    "reference_wes": f"{exact_reference.wes:.6f}",
                    "selected_wes": f"{selected.wes:.6f}",
                    "wes_reduction_percent": f"{percent_reduction(exact_reference.wes, selected.wes):.6f}",
                    "reference_utility": f"{exact_reference.achieved_utility:.6f}",
                    "selected_utility": f"{selected.achieved_utility:.6f}",
                    "utility_loss_percent": f"{percent_reduction(exact_reference.achieved_utility, selected.achieved_utility):.6f}",
                    "reference_visits": exact_reference.completed_poi_count,
                    "selected_visits": selected.completed_poi_count,
                    "visit_loss": exact_reference.completed_poi_count - selected.completed_poi_count,
                    "reference_travel_min": f"{exact_reference.total_travel_min:.6f}",
                    "selected_travel_min": f"{selected.total_travel_min:.6f}",
                    "u_star": f"{chosen.u_star:.6f}",
                    "fallback": chosen.guardrail_fallback_used,
                })

    summaries = []
    for alpha in ALPHAS:
        group = [row for row in raw_rows if float(row["alpha"]) == alpha]
        reductions = [float(row["wes_reduction_percent"]) for row in group]
        losses = [float(row["utility_loss_percent"]) for row in group]
        visit_losses = [int(row["visit_loss"]) for row in group]
        route_changes = sum(str(row["route_changed"]).lower() == "true" for row in group)
        median_reduction = statistics.median(reductions)
        median_loss = statistics.median(losses)
        median_visit_loss = statistics.median(visit_losses)
        summaries.append({
            "alpha": f"{alpha:.2f}", "pairs": len(group), "route_change_count": route_changes,
            "median_wes_reduction_percent": f"{median_reduction:.6f}",
            "median_utility_loss_percent": f"{median_loss:.6f}",
            "median_visit_loss": f"{median_visit_loss:.6f}",
            "supported_rule_met": bool(median_reduction >= 15 and median_loss <= 10 and median_visit_loss <= 1),
            "partial_rule_met": bool(route_changes >= 1 and median_reduction > 0 and not (median_reduction >= 15 and median_loss <= 10 and median_visit_loss <= 1)),
        })
    write_csv(RAW_FILE, raw_rows)
    write_csv(SUMMARY_FILE, summaries)
    write_csv(COMPARATOR_FILE, comparator_rows)
    MANIFEST_FILE.write_text(json.dumps({
        "experiment": "predeclared exact-front thermal-selection ablation",
        "classification": "RQ2 primary exact-selector evidence; GA comparisons are implementation ablations",
        "alphas": ALPHAS, "primary_guardrail_anchor": "exact thermal-front maximum utility",
        "visit_guardrail_margin": 1, "candidate_window_limit": 10,
        "random_restart_count": RANDOM_RESTARTS, "instances": list(WEATHER_INSTANCE_IDS),
        "seeds": list(SEEDS), "generated_utc": utc_timestamp(),
        "inputs": {"poi": file_fingerprint(POI_FILE), "matrix": file_fingerprint(MATRIX_FILE)},
    }, indent=2), encoding="utf-8")
    print(f"Wrote {RAW_FILE}, {SUMMARY_FILE}, and {COMPARATOR_FILE}.")


if __name__ == "__main__":
    main()
