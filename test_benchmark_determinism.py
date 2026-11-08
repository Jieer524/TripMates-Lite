"""
Benchmark-determinism regression test.

Guards the frozen thermal-ablation result against silent change from
performance work (Task 6).  It recomputes the seed-matched adaptive and
thermal replanning for a sample of the frozen 90-pair ablation and
asserts that the selected routes and key metrics still match the frozen
raw CSV exactly.

It also checks that the memoised travel-adjustment helper returns the
same value as the original, uncached construction.

Usage:
    python test_benchmark_determinism.py          # quick sample
    python test_benchmark_determinism.py --full    # all 90 pairs
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from adaptive_ga_core import (
    REPLANNING_GA_CONFIG,
    prepare_adaptive_problem,
    run_adaptive_without_thermal_ga,
)
from exposure_core import calculate_exposure_metrics
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
    get_travel_adjustments,
    simulate_replanning_state,
)
from tripmates_lite_ga import run_tripmates_lite_ga


POI_FILE = Path("poi_data_calculated_final.csv")
MATRIX_FILE = Path("walking_time_matrix_final.csv")
FROZEN_RAW_FILE = Path(
    "results/weather_ablation/"
    "weather_ablation_raw_90_pairs.csv"
)
INITIAL_ROUTES_FILE = Path(
    "results/static_utility/"
    "initial_routes_by_seed.json"
)

QUICK_SEEDS = (1001, 1005, 1010)

ROUTE_TOLERANCE = 1e-6


def frozen_pairs(
    frame: pd.DataFrame,
    *,
    full: bool,
) -> pd.DataFrame:
    if full:
        return frame
    return frame[frame["seed"].isin(QUICK_SEEDS)]


def main() -> None:
    full = "--full" in sys.argv

    data = load_planning_data(
        poi_file=POI_FILE,
        matrix_file=MATRIX_FILE,
    )

    routes_by_seed = json.loads(
        INITIAL_ROUTES_FILE.read_text(encoding="utf-8")
    )

    frozen = pd.read_csv(FROZEN_RAW_FILE)

    scenario_lookup = {
        instance.instance_id: instance
        for instance in build_scenario_instances()
    }

    # ------------------------------------------------------------
    # Travel-adjustment cache equivalence
    # ------------------------------------------------------------

    distinct_scenarios = {
        instance.scenario.scenario_id: instance.scenario
        for instance in scenario_lookup.values()
    }

    for scenario in distinct_scenarios.values():
        cached = get_travel_adjustments(data, scenario)
        fresh = scenario.build_travel_adjustments(data)

        if cached != fresh:
            raise RuntimeError(
                "Cached travel adjustments differ from a fresh build "
                f"for scenario {scenario.scenario_id}."
            )

    sample = frozen_pairs(frozen, full=full)

    checked = 0

    for seed in sorted(sample["seed"].unique()):
        initial_route = [
            int(poi_id)
            for poi_id in routes_by_seed[str(int(seed))]
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
            required_mandatory_ids=data.mandatory_ids,
        )

        if not initial_schedule.feasible:
            raise RuntimeError(
                f"Seed {seed}: initial route is infeasible."
            )

        seed_rows = sample[sample["seed"] == seed]

        for _, row in seed_rows.iterrows():
            instance_id = str(row["instance_id"])
            scenario_instance = scenario_lookup[instance_id]

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

            adaptive_result = run_adaptive_without_thermal_ga(
                data=data,
                problem=problem,
                seed=int(seed),
                config=REPLANNING_GA_CONFIG,
            )
            adaptive_solution = adaptive_result.selected_solution

            tripmates_result = run_tripmates_lite_ga(
                data=data,
                initial_schedule=initial_schedule,
                problem=problem,
                seed=int(seed),
                config=REPLANNING_GA_CONFIG,
            )
            tripmates_solution = tripmates_result.selected_solution

            adaptive_exposure = calculate_exposure_metrics(
                data=data,
                initial_schedule=initial_schedule,
                state=state,
                repaired_schedule=adaptive_solution.schedule,
                thermal_multiplier=(
                    scenario_instance.scenario.thermal_multiplier
                ),
            )

            expected_adaptive_route = json.loads(row["adaptive_route"])
            expected_tripmates_route = json.loads(row["tripmates_route"])

            failures: list[str] = []

            if list(adaptive_solution.route) != expected_adaptive_route:
                failures.append("adaptive route")

            if list(tripmates_solution.route) != expected_tripmates_route:
                failures.append("tripmates route")

            if abs(
                adaptive_exposure.weather_weighted_exposure_score
                - float(row["adaptive_wes"])
            ) > ROUTE_TOLERANCE:
                failures.append("adaptive WES")

            if abs(
                tripmates_solution.exposure.weather_weighted_exposure_score
                - float(row["tripmates_wes"])
            ) > ROUTE_TOLERANCE:
                failures.append("tripmates WES")

            if abs(
                adaptive_solution.achieved_utility
                - float(row["adaptive_utility"])
            ) > ROUTE_TOLERANCE:
                failures.append("adaptive utility")

            if abs(
                adaptive_solution.total_travel_min
                - float(row["adaptive_travel_min"])
            ) > ROUTE_TOLERANCE:
                failures.append("adaptive travel")

            if failures:
                raise RuntimeError(
                    f"Frozen result changed for seed {seed}, "
                    f"{instance_id}: {failures}."
                )

            checked += 1

    print(
        "PASS: benchmark determinism preserved for "
        f"{checked} frozen ablation pairs "
        f"({'full' if full else 'quick sample'})."
    )


if __name__ == "__main__":
    main()
