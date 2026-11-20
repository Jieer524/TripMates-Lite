"""
RQ1 full 18-instance benchmark runner.

Runs five planners over the repository's official 18 held-out scenario
instances (six frozen disruption families x early / nominal / late):

- Static Distance        (deterministic, seed-independent)
- Static Utility         (seed-matched frozen initial route)
- Greedy Repair          (deterministic given the seed-matched route)
- Adaptive Without Thermal (stochastic, matched seeds)
- TripMates Lite         (stochastic, matched seeds)

Protocol
--------
- Deterministic planners run once per scenario where their input is
  seed-independent (Static Distance).
- Static Utility and Greedy Repair are deterministic but are evaluated
  once per seed-matched initial route so they are paired with the
  stochastic planners on identical inputs.  Their seed is the
  seed-matched initial route, not an RNG seed.
- Stochastic planners use matched seeds 1001-1010.
- Static planners use the predeclared skip-and-continue execution rule:
  they keep the remaining original order, skip optional POIs that become
  infeasible, and report any mandatory-POI failure explicitly.
- The frozen research GA configuration is preserved.

Outputs (results/rq1_full_benchmark_corrected_static_rule/):
- rq1_raw_runs.csv
- rq1_planner_scenario_summary.csv
- rq1_overall_planner_summary.csv
- rq1_failure_cases.csv
- rq1_run_config.json

Usage:
    python run_rq1_full_benchmark.py           # full corrected run
    python run_rq1_full_benchmark.py --smoke    # corrected smoke only
    python run_rq1_full_benchmark.py --legacy-static  # archived legacy rule
"""

from __future__ import annotations

from dataclasses import asdict
import argparse
import time
from pathlib import Path

import pandas as pd

from adaptive_ga_core import (
    REPLANNING_GA_CONFIG,
    calculate_locked_travel_minutes,
    prepare_adaptive_problem,
    run_adaptive_without_thermal_ga,
)
from benchmark_common import (
    ALL_INSTANCE_IDS,
    MATRIX_FILE,
    POI_FILE,
    SEEDS,
    build_initial_schedules,
    build_metric_record,
    build_scenario_lookup,
    encode_ids,
    evaluate_static_suffix,
    evaluate_static_suffix_skip_and_continue,
    file_fingerprint,
    load_initial_routes,
    route_utility,
    runtime_metadata,
    utc_timestamp,
    write_json,
)
from exposure_core import calculate_exposure_metrics
from greedy_repair import run_greedy_repair
from planning_core import (
    PlanningConstraints,
    decode_route,
    load_planning_data,
)
from scenario_core import simulate_replanning_state
from static_distance_baseline import build_static_distance_route
from stability_metrics import StabilityWeightConfig
from tripmates_lite_ga import run_tripmates_lite_ga


CANONICAL_OUTPUT_DIRECTORY = Path(
    "results/rq1_full_benchmark_corrected_static_rule"
)
LEGACY_OUTPUT_DIRECTORY = Path("results/rq1_full_benchmark")

OUTPUT_DIRECTORY = CANONICAL_OUTPUT_DIRECTORY

RAW_RUNS_FILE = OUTPUT_DIRECTORY / "rq1_raw_runs.csv"
PLANNER_SCENARIO_FILE = (
    OUTPUT_DIRECTORY / "rq1_planner_scenario_summary.csv"
)
OVERALL_PLANNER_FILE = (
    OUTPUT_DIRECTORY / "rq1_overall_planner_summary.csv"
)
FAILURE_CASES_FILE = (
    OUTPUT_DIRECTORY / "rq1_failure_cases.csv"
)
RUN_CONFIG_FILE = OUTPUT_DIRECTORY / "rq1_run_config.json"
STATIC_EXECUTION_MODE = "planned_skip_and_continue_execution"


def isolate_smoke_outputs() -> None:
    """Redirect a ``--smoke`` run into an isolated sub-folder.

    Smoke runs must never overwrite a completed full benchmark, so a
    ``--smoke`` invocation writes every artefact under
    ``<output-directory>/smoke/`` instead of the shared full-run paths.
    """
    global OUTPUT_DIRECTORY
    global RAW_RUNS_FILE, PLANNER_SCENARIO_FILE, OVERALL_PLANNER_FILE
    global FAILURE_CASES_FILE, RUN_CONFIG_FILE

    smoke_directory = OUTPUT_DIRECTORY / "smoke"

    RAW_RUNS_FILE = smoke_directory / RAW_RUNS_FILE.name
    PLANNER_SCENARIO_FILE = (
        smoke_directory / PLANNER_SCENARIO_FILE.name
    )
    OVERALL_PLANNER_FILE = (
        smoke_directory / OVERALL_PLANNER_FILE.name
    )
    FAILURE_CASES_FILE = (
        smoke_directory / FAILURE_CASES_FILE.name
    )
    RUN_CONFIG_FILE = smoke_directory / RUN_CONFIG_FILE.name

    OUTPUT_DIRECTORY = smoke_directory
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)


PLANNER_STATIC_DISTANCE = "Static Distance"
PLANNER_STATIC_UTILITY = "Static Utility"
PLANNER_GREEDY = "Greedy Repair"
PLANNER_ADAPTIVE = "Adaptive Without Thermal"
PLANNER_TRIPMATES = "TripMates Lite"

PLANNERS = (
    PLANNER_STATIC_DISTANCE,
    PLANNER_STATIC_UTILITY,
    PLANNER_GREEDY,
    PLANNER_ADAPTIVE,
    PLANNER_TRIPMATES,
)


def static_planner_record(
    *,
    planner: str,
    data,
    scenario_instance,
    seed,
    initial_route,
    initial_schedule,
    initial_route_utility,
    constraints,
    deterministic,
    static_execution_mode="planned_skip_and_continue_execution",
) -> dict:
    started_at = time.perf_counter()

    state = simulate_replanning_state(
        data=data,
        initial_schedule=initial_schedule,
        scenario_instance=scenario_instance,
    )

    locked_travel = calculate_locked_travel_minutes(
        initial_schedule=initial_schedule,
        state=state,
    )

    evaluator = (evaluate_static_suffix_skip_and_continue
                 if static_execution_mode == "planned_skip_and_continue_execution"
                 else evaluate_static_suffix)
    evaluation = evaluator(
        data=data,
        state=state,
        scenario_instance=scenario_instance,
        constraints=constraints,
    )

    runtime = time.perf_counter() - started_at

    if evaluation.feasible and evaluation.schedule is not None:
        exposure = calculate_exposure_metrics(
            data=data,
            initial_schedule=initial_schedule,
            state=state,
            repaired_schedule=evaluation.schedule,
            thermal_multiplier=(
                scenario_instance.scenario.thermal_multiplier
            ),
        )

        achieved_utility = (
            state.utility_achieved
            + route_utility(data, evaluation.repaired_route)
        )
        completed_poi_count = (
            len(state.completed_poi_ids)
            + len(evaluation.repaired_route)
        )
        total_travel_min = (
            locked_travel
            + evaluation.schedule.total_travel_min
        )
        mandatory_complete = set(
            state.remaining_mandatory_ids
        ) <= set(evaluation.repaired_route)

        return build_metric_record(
            planner=planner,
            scenario_instance=scenario_instance,
            seed=seed,
            initial_route=initial_route,
            initial_schedule=initial_schedule,
            initial_route_utility=initial_route_utility,
            state=state,
            repaired_route=list(evaluation.repaired_route),
            achieved_utility=achieved_utility,
            completed_poi_count=completed_poi_count,
            total_travel_min=total_travel_min,
            locked_travel_min=locked_travel,
            repaired_schedule=evaluation.schedule,
            exposure=exposure,
            feasible=True,
            failure_reason=None,
            mandatory_complete=mandatory_complete,
            runtime_seconds=runtime,
            deterministic=deterministic,
            extra={"static_execution_mode": static_execution_mode,
                   "skipped_optional_pois": list(evaluation.skipped_pois)},
        )

    return build_metric_record(
        planner=planner,
        scenario_instance=scenario_instance,
        seed=seed,
        initial_route=initial_route,
        initial_schedule=initial_schedule,
        initial_route_utility=initial_route_utility,
        state=state,
        repaired_route=list(evaluation.repaired_route),
        achieved_utility=None,
        completed_poi_count=(
            len(state.completed_poi_ids)
            + evaluation.executed_poi_count
        ),
        total_travel_min=None,
        locked_travel_min=locked_travel,
        repaired_schedule=None,
        exposure=None,
        feasible=False,
        failure_reason=evaluation.failure_reason,
        mandatory_complete=False,
        runtime_seconds=runtime,
        deterministic=deterministic,
        extra={"static_execution_mode": static_execution_mode,
               "skipped_optional_pois": list(evaluation.skipped_pois)},
    )


def greedy_planner_record(
    *,
    data,
    scenario_instance,
    seed,
    initial_route,
    initial_schedule,
    initial_route_utility,
    constraints,
) -> dict:
    started_at = time.perf_counter()

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
        constraints=constraints,
        seed=seed,
    )

    runtime = time.perf_counter() - started_at

    if result.status == "success" and result.selected_solution:
        solution = result.selected_solution
        exposure = calculate_exposure_metrics(
            data=data,
            initial_schedule=initial_schedule,
            state=state,
            repaired_schedule=solution.schedule,
            thermal_multiplier=(
                scenario_instance.scenario.thermal_multiplier
            ),
        )

        return build_metric_record(
            planner=PLANNER_GREEDY,
            scenario_instance=scenario_instance,
            seed=seed,
            initial_route=initial_route,
            initial_schedule=initial_schedule,
            initial_route_utility=initial_route_utility,
            state=state,
            repaired_route=list(solution.route),
            achieved_utility=solution.achieved_utility,
            completed_poi_count=solution.completed_poi_count,
            total_travel_min=solution.total_travel_min,
            locked_travel_min=result.locked_travel_min,
            repaired_schedule=solution.schedule,
            exposure=exposure,
            feasible=True,
            failure_reason=None,
            mandatory_complete=set(
                state.remaining_mandatory_ids
            ) <= set(solution.route),
            runtime_seconds=runtime,
            deterministic=True,
            repair_record=result.repair_record,
        )

    if result.status == "no_action":
        return build_metric_record(
            planner=PLANNER_GREEDY,
            scenario_instance=scenario_instance,
            seed=seed,
            initial_route=initial_route,
            initial_schedule=initial_schedule,
            initial_route_utility=initial_route_utility,
            state=state,
            repaired_route=[],
            achieved_utility=state.utility_achieved,
            completed_poi_count=len(
                state.completed_poi_ids
            ),
            total_travel_min=0.0,
            locked_travel_min=0.0,
            repaired_schedule=None,
            exposure=None,
            feasible=True,
            failure_reason=None,
            mandatory_complete=True,
            runtime_seconds=runtime,
            deterministic=True,
            repair_record=result.repair_record,
        )

    return build_metric_record(
        planner=PLANNER_GREEDY,
        scenario_instance=scenario_instance,
        seed=seed,
        initial_route=initial_route,
        initial_schedule=initial_schedule,
        initial_route_utility=initial_route_utility,
        state=state,
        repaired_route=None,
        achieved_utility=None,
        completed_poi_count=None,
        total_travel_min=None,
        locked_travel_min=result.locked_travel_min,
        repaired_schedule=None,
        exposure=None,
        feasible=False,
        failure_reason=result.failure_reason,
        mandatory_complete=False,
        runtime_seconds=runtime,
        deterministic=True,
        repair_record=result.repair_record,
    )


def ga_planner_record(
    *,
    planner: str,
    data,
    problem,
    scenario_instance,
    seed,
    initial_route,
    initial_schedule,
    initial_route_utility,
) -> dict:
    started_at = time.perf_counter()

    if planner == PLANNER_ADAPTIVE:
        result = run_adaptive_without_thermal_ga(
            data=data,
            problem=problem,
            seed=seed,
            config=REPLANNING_GA_CONFIG,
        )
        solution = result.selected_solution
        exposure = calculate_exposure_metrics(
            data=data,
            initial_schedule=initial_schedule,
            state=problem.state,
            repaired_schedule=solution.schedule,
            thermal_multiplier=(
                scenario_instance.scenario.thermal_multiplier
            ),
        )
        generations = result.generations_executed
        convergence = result.convergence_generation
        evaluated = result.evaluated_chromosome_count
        unique_routes = result.unique_decoded_route_count
        front_size = len(result.first_front)
    else:
        result = run_tripmates_lite_ga(
            data=data,
            initial_schedule=initial_schedule,
            problem=problem,
            seed=seed,
            config=REPLANNING_GA_CONFIG,
        )
        solution = result.selected_solution
        exposure = solution.exposure
        generations = result.generations_executed
        convergence = result.convergence_generation
        evaluated = result.evaluated_chromosome_count
        unique_routes = result.unique_decoded_route_count
        front_size = len(result.first_front)

    runtime = time.perf_counter() - started_at

    return build_metric_record(
        planner=planner,
        scenario_instance=scenario_instance,
        seed=seed,
        initial_route=initial_route,
        initial_schedule=initial_schedule,
        initial_route_utility=initial_route_utility,
        state=problem.state,
        repaired_route=list(solution.route),
        achieved_utility=solution.achieved_utility,
        completed_poi_count=solution.completed_poi_count,
        total_travel_min=solution.total_travel_min,
        locked_travel_min=problem.locked_travel_min,
        repaired_schedule=solution.schedule,
        exposure=exposure,
        feasible=True,
        failure_reason=None,
        mandatory_complete=set(
            problem.state.remaining_mandatory_ids
        ) <= set(solution.route),
        runtime_seconds=runtime,
        deterministic=False,
        generations_executed=generations,
        convergence_generation=convergence,
        evaluated_chromosomes=evaluated,
        unique_decoded_routes=unique_routes,
        pareto_front_size=front_size,
        repair_record=result.repair_record,
    )


def mean_numeric(series: pd.Series) -> float:
    """Column mean that tolerates ``None`` / mixed dtypes.

    Stability columns are ``None`` whenever the quantity could not be
    measured (for example a time shift on an infeasible repair), so the
    mean must coerce rather than raise.
    """

    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().sum() == 0:
        return float("nan")

    return float(numeric.mean())


def build_planner_scenario_summary(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    for (planner, instance_id), group in raw.groupby(
        ["planner", "instance_id"], sort=True
    ):
        feasible = group["feasible"]

        rows.append(
            {
                "planner": planner,
                "instance_id": instance_id,
                "scenario_id": group["scenario_id"].iloc[0],
                "scenario_name": group[
                    "scenario_name"
                ].iloc[0],
                "trigger_variant": group[
                    "trigger_variant"
                ].iloc[0],
                "run_count": int(len(group)),
                "feasible_count": int(feasible.sum()),
                "infeasible_count": int((~feasible).sum()),
                "feasible_percent": float(
                    feasible.mean() * 100.0
                ),
                "mandatory_complete_count": int(
                    group["mandatory_complete"]
                    .fillna(False)
                    .sum()
                ),
                "route_change_count": int(
                    group["route_changed"]
                    .fillna(False)
                    .sum()
                ),
                "mean_completed_poi_count": float(
                    group["completed_poi_count"].mean()
                ),
                "mean_achieved_utility": float(
                    group["achieved_utility"].mean()
                ),
                "mean_utility_retention_ratio": float(
                    group["utility_retention_ratio"].mean()
                ),
                "mean_itinerary_completion_ratio": float(
                    group[
                        "itinerary_completion_ratio"
                    ].mean()
                ),
                "mean_original_poi_retention": float(
                    group["original_poi_retention"].mean()
                ),
                "mean_total_travel_min": float(
                    group["total_travel_min"].mean()
                ),
                "mean_total_cost_rm": float(
                    group["total_cost_rm"].mean()
                ),
                "mean_total_outdoor_exposure_min": float(
                    group[
                        "total_outdoor_exposure_min"
                    ].mean()
                ),
                "mean_wes": float(group["wes"].mean()),
                "median_runtime_seconds": float(
                    group["runtime_seconds"].median()
                ),

                # --------------------------------------------------
                # Itinerary stability (see stability_metrics)
                # --------------------------------------------------
                "mean_preserved_poi_count": mean_numeric(
                    group["preserved_poi_count"]
                ),
                "mean_preserved_poi_percentage": mean_numeric(
                    group["preserved_poi_percentage"]
                ),
                "mean_removed_poi_count": mean_numeric(
                    group["removed_poi_count"]
                ),
                "mean_added_poi_count": mean_numeric(
                    group["added_poi_count"]
                ),
                "mean_changed_activity_count": mean_numeric(
                    group["changed_activity_count"]
                ),
                "mean_modified_suffix_activity_count": mean_numeric(
                    group["modified_suffix_activity_count"]
                ),
                "mean_time_shift_count": mean_numeric(
                    group["time_shift_count"]
                ),
                "mean_order_deviation": mean_numeric(
                    group["order_deviation"]
                ),
                "mean_edit_distance_score": mean_numeric(
                    group["edit_distance_score"]
                ),
                "locked_prefix_violation_count": int(
                    group["locked_prefix_changed"]
                    .fillna(False)
                    .sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def build_overall_planner_summary(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    for planner, group in raw.groupby(
        "planner", sort=True
    ):
        feasible = group["feasible"]

        rows.append(
            {
                "planner": planner,
                "run_count": int(len(group)),
                "feasible_count": int(feasible.sum()),
                "infeasible_count": int((~feasible).sum()),
                "feasible_percent": float(
                    feasible.mean() * 100.0
                ),
                "mandatory_complete_percent": float(
                    group["mandatory_complete"]
                    .fillna(False)
                    .mean()
                    * 100.0
                ),
                "mean_itinerary_completion_ratio": float(
                    group[
                        "itinerary_completion_ratio"
                    ].mean()
                ),
                "mean_original_poi_retention": float(
                    group["original_poi_retention"].mean()
                ),
                "mean_achieved_utility": float(
                    group["achieved_utility"].mean()
                ),
                "mean_utility_retention_ratio": float(
                    group["utility_retention_ratio"].mean()
                ),
                "mean_total_travel_min": float(
                    group["total_travel_min"].mean()
                ),
                "mean_total_cost_rm": float(
                    group["total_cost_rm"].mean()
                ),
                "mean_total_outdoor_exposure_min": float(
                    group[
                        "total_outdoor_exposure_min"
                    ].mean()
                ),
                "mean_wes": float(group["wes"].mean()),
                "median_runtime_seconds": float(
                    group["runtime_seconds"].median()
                ),
                "p95_runtime_seconds": float(
                    group["runtime_seconds"].quantile(0.95)
                ),

                # --------------------------------------------------
                # Itinerary stability (see stability_metrics)
                # --------------------------------------------------
                "mean_preserved_poi_count": mean_numeric(
                    group["preserved_poi_count"]
                ),
                "mean_preserved_poi_percentage": mean_numeric(
                    group["preserved_poi_percentage"]
                ),
                "mean_removed_poi_count": mean_numeric(
                    group["removed_poi_count"]
                ),
                "mean_added_poi_count": mean_numeric(
                    group["added_poi_count"]
                ),
                "mean_changed_activity_count": mean_numeric(
                    group["changed_activity_count"]
                ),
                "mean_modified_suffix_activity_count": mean_numeric(
                    group["modified_suffix_activity_count"]
                ),
                "mean_time_shift_count": mean_numeric(
                    group["time_shift_count"]
                ),
                "mean_order_deviation": mean_numeric(
                    group["order_deviation"]
                ),
                "mean_edit_distance_score": mean_numeric(
                    group["edit_distance_score"]
                ),
                "locked_prefix_violation_count": int(
                    group["locked_prefix_changed"]
                    .fillna(False)
                    .sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--legacy-static",
        action="store_true",
        help=(
            "Use the archived legacy static replay rule and write to "
            "results/rq1_full_benchmark/."
        ),
    )
    parser.add_argument(
        "--corrected-static",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.legacy_static and args.corrected_static:
        parser.error(
            "--legacy-static and --corrected-static cannot be combined."
        )

    global STATIC_EXECUTION_MODE
    global OUTPUT_DIRECTORY, RAW_RUNS_FILE, PLANNER_SCENARIO_FILE
    global OVERALL_PLANNER_FILE, FAILURE_CASES_FILE, RUN_CONFIG_FILE
    STATIC_EXECUTION_MODE = (
        "legacy_static_execution"
        if args.legacy_static
        else "planned_skip_and_continue_execution"
    )
    OUTPUT_DIRECTORY = (
        LEGACY_OUTPUT_DIRECTORY
        if args.legacy_static
        else CANONICAL_OUTPUT_DIRECTORY
    )
    RAW_RUNS_FILE = OUTPUT_DIRECTORY / "rq1_raw_runs.csv"
    PLANNER_SCENARIO_FILE = OUTPUT_DIRECTORY / "rq1_planner_scenario_summary.csv"
    OVERALL_PLANNER_FILE = OUTPUT_DIRECTORY / "rq1_overall_planner_summary.csv"
    FAILURE_CASES_FILE = OUTPUT_DIRECTORY / "rq1_failure_cases.csv"
    RUN_CONFIG_FILE = OUTPUT_DIRECTORY / "rq1_run_config.json"

    if args.smoke:
        isolate_smoke_outputs()

    started_at = time.perf_counter()

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    data = load_planning_data(
        poi_file=POI_FILE,
        matrix_file=MATRIX_FILE,
    )

    constraints = PlanningConstraints()

    routes_by_seed = load_initial_routes()
    initial_schedules = build_initial_schedules(
        data, routes_by_seed
    )
    scenario_lookup = build_scenario_lookup()

    initial_route_utility_by_seed = {
        seed: route_utility(data, route)
        for seed, route in routes_by_seed.items()
    }

    static_distance_route = build_static_distance_route(
        data=data, start_poi_id=1
    )
    static_distance_schedule = decode_route(
        data=data,
        visit_sequence=static_distance_route,
        start_poi_id=1,
        start_time_min=constraints.start_time_min,
        trip_end_min=constraints.trip_end_min,
        day_code=constraints.day_code,
        cost_budget_rm=constraints.cost_budget_rm,
        start_poi_is_visit=True,
        required_mandatory_ids=data.mandatory_ids,
    )
    static_distance_utility = route_utility(
        data, static_distance_route
    )

    if args.smoke:
        instance_ids = ("S1_nominal",)
        seeds = (1001,)
    else:
        instance_ids = ALL_INSTANCE_IDS
        seeds = SEEDS

    records: list[dict] = []

    total_units = len(instance_ids) * (
        1 + len(seeds) * 4
    )
    completed_units = 0

    for instance_id in instance_ids:
        scenario_instance = scenario_lookup[instance_id]

        # ----------------------------------------------------
        # Static Distance (deterministic, seed-independent)
        # ----------------------------------------------------

        records.append(
            static_planner_record(
                planner=PLANNER_STATIC_DISTANCE,
                data=data,
                scenario_instance=scenario_instance,
                seed=None,
                initial_route=static_distance_route,
                initial_schedule=(
                    static_distance_schedule
                ),
                initial_route_utility=(
                    static_distance_utility
                ),
                constraints=constraints,
                deterministic=True,
                static_execution_mode=STATIC_EXECUTION_MODE,
            )
        )
        completed_units += 1

        for seed in seeds:
            initial_route = routes_by_seed[seed]
            initial_schedule = initial_schedules[seed]

            # ------------------------------------------------
            # Static Utility (frozen seed-matched initial route)
            # ------------------------------------------------

            records.append(
                static_planner_record(
                    planner=PLANNER_STATIC_UTILITY,
                    data=data,
                    scenario_instance=(
                        scenario_instance
                    ),
                    seed=seed,
                    initial_route=initial_route,
                    initial_schedule=initial_schedule,
                    initial_route_utility=(
                        initial_route_utility_by_seed[
                            seed
                        ]
                    ),
                    constraints=constraints,
                    deterministic=True,
                    static_execution_mode=STATIC_EXECUTION_MODE,
                )
            )
            completed_units += 1

            # ------------------------------------------------
            # Greedy Repair
            # ------------------------------------------------

            records.append(
                greedy_planner_record(
                    data=data,
                    scenario_instance=(
                        scenario_instance
                    ),
                    seed=seed,
                    initial_route=initial_route,
                    initial_schedule=initial_schedule,
                    initial_route_utility=(
                        initial_route_utility_by_seed[
                            seed
                        ]
                    ),
                    constraints=constraints,
                )
            )
            completed_units += 1

            # ------------------------------------------------
            # Adaptive Without Thermal and TripMates Lite
            # ------------------------------------------------

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
                constraints=constraints,
            )

            records.append(
                ga_planner_record(
                    planner=PLANNER_ADAPTIVE,
                    data=data,
                    problem=problem,
                    scenario_instance=(
                        scenario_instance
                    ),
                    seed=seed,
                    initial_route=initial_route,
                    initial_schedule=initial_schedule,
                    initial_route_utility=(
                        initial_route_utility_by_seed[
                            seed
                        ]
                    ),
                )
            )
            completed_units += 1

            records.append(
                ga_planner_record(
                    planner=PLANNER_TRIPMATES,
                    data=data,
                    problem=problem,
                    scenario_instance=(
                        scenario_instance
                    ),
                    seed=seed,
                    initial_route=initial_route,
                    initial_schedule=initial_schedule,
                    initial_route_utility=(
                        initial_route_utility_by_seed[
                            seed
                        ]
                    ),
                )
            )
            completed_units += 1

            print(
                f"[{completed_units:>4}/{total_units}] "
                f"{instance_id} seed={seed}"
            )

    raw = pd.DataFrame(records)

    raw.to_csv(
        RAW_RUNS_FILE, index=False, float_format="%.6f"
    )

    planner_scenario = build_planner_scenario_summary(raw)
    planner_scenario.to_csv(
        PLANNER_SCENARIO_FILE,
        index=False,
        float_format="%.6f",
    )

    overall = build_overall_planner_summary(raw)
    overall.to_csv(
        OVERALL_PLANNER_FILE,
        index=False,
        float_format="%.6f",
    )

    failure_cases = raw[~raw["feasible"]].copy()
    failure_cases.to_csv(
        FAILURE_CASES_FILE,
        index=False,
        float_format="%.6f",
    )

    total_runtime = time.perf_counter() - started_at

    run_config = {
        "experiment": "RQ1 full 18-instance benchmark",
        "mode": (
            "smoke_test" if args.smoke else "research_benchmark"
        ),
        "timestamp_utc": utc_timestamp(),

        "planners": list(PLANNERS),
        "instance_ids": list(instance_ids),
        "official_instance_count": len(
            ALL_INSTANCE_IDS
        ),
        "seeds": list(seeds),

        "protocol": {
            "static_distance": (
                "Deterministic, seed-independent distance-first "
                "route; uses the configured static execution rule "
                "after disruption."
            ),
            "static_utility": (
                "Seed-matched frozen Static Utility GA route; "
                "uses the configured static execution rule after "
                "disruption."
            ),
            "greedy_repair": (
                "Deterministic given the seed-matched route; no "
                "WES, thermal or GA input."
            ),
            "adaptive_without_thermal": (
                "Stochastic GA repair, matched seeds."
            ),
            "tripmates_lite": (
                "Stochastic thermal-aware GA repair, matched "
                "seeds, frozen selection policy."
            ),
        },

        "candidate_window_limit": 10,
        "static_execution_mode": STATIC_EXECUTION_MODE,
        "ga_configuration": asdict(REPLANNING_GA_CONFIG),

        "stability_weights": (
            StabilityWeightConfig().to_dict()
        ),
        "stability_metric_module": "stability_metrics.py",

        "dataset": file_fingerprint(POI_FILE),
        "matrix": file_fingerprint(MATRIX_FILE),
        "initial_routes": file_fingerprint(
            Path(
                "results/static_utility/"
                "initial_routes_by_seed.json"
            )
        ),

        "seed_interpretation": (
            "Seeds are repeated stochastic optimisation runs for "
            "stability and paired comparison, not independent "
            "real-world disruptions. Static Utility and Greedy "
            "Repair use the seed only to select the seed-matched "
            "initial route."
        ),

        "run_count": int(len(raw)),
        "failure_count": int(len(failure_cases)),
        "total_runtime_seconds": round(total_runtime, 3),
        "runtime_metadata": runtime_metadata(),
    }

    write_json(RUN_CONFIG_FILE, run_config)

    print()
    print("=" * 100)
    print("RQ1 FULL BENCHMARK SUMMARY")
    print("=" * 100)
    print("Mode:", run_config["mode"])
    print("Runs:", len(raw))
    print("Failures:", len(failure_cases))
    print("Total runtime:", f"{total_runtime:.1f} s")
    print()
    print(overall.to_string(index=False))
    print()
    print("Raw runs:", RAW_RUNS_FILE.resolve())
    print("Overall summary:", OVERALL_PLANNER_FILE.resolve())
    print(
        "Planner-by-scenario summary:",
        PLANNER_SCENARIO_FILE.resolve(),
    )
    print("Failure cases:", FAILURE_CASES_FILE.resolve())
    print("Run config:", RUN_CONFIG_FILE.resolve())


if __name__ == "__main__":
    main()
