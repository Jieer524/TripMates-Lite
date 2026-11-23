"""Run a matched single-route AILS-inspired comparator without changing RQ1 inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import pandas as pd

from adaptive_ga_core import prepare_adaptive_problem
from benchmark_common import (
    ALL_INSTANCE_IDS, MATRIX_FILE, POI_FILE, SEEDS, build_initial_schedules,
    build_metric_record, build_scenario_lookup, file_fingerprint,
    load_initial_routes, route_utility, runtime_metadata, utc_timestamp,
)
from exposure_core import calculate_exposure_metrics
from ils_repair_baseline import ILSConfig, run_ils_repair
from planning_core import PlanningConstraints, load_planning_data
from run_rq1_full_benchmark import build_overall_planner_summary, build_planner_scenario_summary
from scenario_core import simulate_replanning_state


ROOT = Path(__file__).resolve().parent
PRIMARY = ROOT / "results/rq1_full_benchmark_minimum_deletion_primary"
OUTPUT = ROOT / "results/rq1_ils_external_comparison"
PLANNER = "Single-route AILS adaptation"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    target = OUTPUT / "smoke" if args.smoke else OUTPUT
    target.mkdir(parents=True, exist_ok=True)
    config = ILSConfig()
    data = load_planning_data(poi_file=POI_FILE, matrix_file=MATRIX_FILE)
    constraints = PlanningConstraints()
    routes = load_initial_routes()
    schedules = build_initial_schedules(data, routes)
    scenarios = build_scenario_lookup()
    instances = ("S1_nominal",) if args.smoke else ALL_INSTANCE_IDS
    seeds = (1001,) if args.smoke else SEEDS
    rows: list[dict] = []
    for instance_id in instances:
        scenario = scenarios[instance_id]
        for seed in seeds:
            initial_route = routes[seed]
            initial_schedule = schedules[seed]
            state = simulate_replanning_state(
                data=data, initial_schedule=initial_schedule, scenario_instance=scenario,
            )
            problem = prepare_adaptive_problem(
                data=data, initial_schedule=initial_schedule, state=state,
                scenario_instance=scenario, constraints=constraints,
            )
            started = time.perf_counter()
            result = run_ils_repair(data=data, problem=problem, seed=seed, config=config)
            runtime = time.perf_counter() - started
            exposure = calculate_exposure_metrics(
                data=data, initial_schedule=initial_schedule, state=state,
                repaired_schedule=result.schedule,
                thermal_multiplier=scenario.scenario.thermal_multiplier,
            )
            row = build_metric_record(
                planner=PLANNER, scenario_instance=scenario, seed=seed,
                initial_route=initial_route, initial_schedule=initial_schedule,
                initial_route_utility=route_utility(data, initial_route),
                state=state, repaired_route=list(result.route),
                achieved_utility=problem.locked_utility + sum(data.pois[p].utility_score for p in result.route),
                completed_poi_count=problem.locked_completed_count + len(result.route),
                total_travel_min=problem.locked_travel_min + result.schedule.total_travel_min,
                locked_travel_min=problem.locked_travel_min,
                repaired_schedule=result.schedule, exposure=exposure,
                feasible=True, failure_reason=None,
                mandatory_complete=set(state.remaining_mandatory_ids) <= set(result.route),
                runtime_seconds=runtime, deterministic=False,
                evaluated_chromosomes=result.evaluations,
                unique_decoded_routes=result.evaluations,
                repair_record=result.repair_record,
                extra={"ils_adaptation": True},
            )
            rows.append(row)
            print(f"{len(rows)}/{len(instances)*len(seeds)} {instance_id} {seed} {runtime:.3f}s", flush=True)

    new = pd.DataFrame(rows)
    if args.smoke:
        combined = new
    else:
        primary = pd.read_csv(PRIMARY / "rq1_raw_runs.csv")
        assert len(primary) == 738
        for field in ("instance_id", "seed"):
            actual = set(new[field].dropna())
            expected = set(primary[primary["planner"] == "Adaptive Without Thermal"][field].dropna())
            assert actual == expected, (field, actual ^ expected)
        combined = pd.concat([primary, new], ignore_index=True)
        assert len(combined) == 918
    new.to_csv(target / "ils_raw_runs.csv", index=False, float_format="%.6f")
    combined.to_csv(target / "rq1_with_ils_raw_runs.csv", index=False, float_format="%.6f")
    build_overall_planner_summary(combined).to_csv(
        target / "rq1_with_ils_overall_summary.csv", index=False, float_format="%.6f"
    )
    build_planner_scenario_summary(combined).to_csv(
        target / "rq1_with_ils_instance_summary.csv", index=False, float_format="%.6f"
    )
    configuration = {
        "experiment": "Matched single-route AILS-inspired external-method comparison",
        "method_status": "adapted algorithm family, not an original AILS implementation",
        "literature": "Hapsari et al. 2019, DOI 10.1007/s40092-019-0315-9",
        "adaptations": ["single post-trigger route", "locked realised prefix", "shared candidate window and decoder", "mandatory visits and admission budget"],
        "algorithm": {"construction": "best feasible optional insertion by utility, visits, travel", "local_search": "best-improvement pairwise exchange and segment reversal", "shake": "remove up to two optional visits then reconstruct", "acceptance": "strict incumbent improvement"},
        "config": vars(config), "instance_ids": list(instances), "seeds": list(seeds),
        "generated_utc": utc_timestamp(), "runtime": runtime_metadata(),
        "primary_input": file_fingerprint(PRIMARY / "rq1_raw_runs.csv") if not args.smoke else None,
        "poi_input": file_fingerprint(POI_FILE), "matrix_input": file_fingerprint(MATRIX_FILE),
        "initial_routes_input": file_fingerprint(ROOT / "results/static_utility/initial_routes_by_seed.json"),
        "row_counts": {"new": len(new), "combined": len(combined)},
    }
    (target / "run_config.json").write_text(json.dumps(configuration, indent=2, default=str), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
