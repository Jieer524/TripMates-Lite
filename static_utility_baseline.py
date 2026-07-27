"""
Run the Static Utility baseline using paired seeds 1001–1010.

The generated initial route for each seed will later be reused by:

- Adaptive Without Thermal
- TripMates Lite
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd

from ga_core import (
    GAConfig,
    GARunResult,
    run_static_utility_ga,
)

from planning_core import (
    load_planning_data,
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

OUTPUT_DIRECTORY = Path(
    "results/static_utility"
)

SEEDS = tuple(
    range(1001, 1011)
)

GA_CONFIG = GAConfig(
    population_size=80,
    maximum_generations=120,

    tournament_size=3,

    crossover_probability=0.90,
    mutation_probability=0.20,

    elitism_count=2,
    stagnation_limit=15,
)


# ================================================================
# Output helpers
# ================================================================

def schedule_to_dataframe(
    run_result: GARunResult,
) -> pd.DataFrame:
    entries = (
        run_result
        .selected_solution
        .schedule
        .entries
    )

    dataframe = pd.DataFrame(
        [
            asdict(entry)
            for entry in entries
        ]
    )

    if not dataframe.empty:
        dataframe[
            "previous_poi_id"
        ] = (
            dataframe[
                "previous_poi_id"
            ]
            .astype("Int64")
        )

    return dataframe


def create_summary_row(
    data,
    run_result: GARunResult,
) -> dict:
    solution = (
        run_result
        .selected_solution
    )

    schedule = solution.schedule

    return {
        "planner": "Static Utility",

        "seed": run_result.seed,

        "route": json.dumps(
            list(solution.route)
        ),

        "route_names": json.dumps(
            [
                data.pois[
                    poi_id
                ].name
                for poi_id in (
                    solution.route
                )
            ],
            ensure_ascii=False,
        ),

        "poi_count": (
            solution
            .completed_poi_count
        ),

        "achieved_utility": round(
            solution
            .achieved_utility,
            6,
        ),

        "total_travel_min": round(
            schedule
            .total_travel_min,
            6,
        ),

        "total_waiting_min": round(
            schedule
            .total_waiting_min,
            6,
        ),

        "total_visit_min": round(
            schedule
            .total_visit_min,
            6,
        ),

        "total_time_used_min": round(
            schedule
            .time_used_min,
            6,
        ),

        "finish_time": (
            schedule.finish_clock
        ),

        "total_cost_rm": round(
            schedule
            .total_cost_rm,
            2,
        ),

        "mandatory_complete": (
            len(
                schedule
                .missing_mandatory_ids
            )
            == 0
        ),

        "feasible": (
            schedule.feasible
        ),

        "generations_executed": (
            run_result
            .generations_executed
        ),

        "convergence_generation": (
            run_result
            .convergence_generation
        ),

        "stopped_by_stagnation": (
            run_result
            .stopped_by_stagnation
        ),

        "evaluated_chromosomes": (
            run_result
            .feasible_solution_count
        ),

        "unique_decoded_routes": (
            run_result
            .unique_route_count
        ),
    }


# ================================================================
# Main
# ================================================================

def main() -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_planning_data(
        poi_file=POI_FILE,
        matrix_file=MATRIX_FILE,
    )

    summary_rows: list[dict] = []

    initial_routes_by_seed: dict[
        str,
        list[int],
    ] = {}

    for seed in SEEDS:
        print()
        print("=" * 90)
        print(
            f"STATIC UTILITY — SEED {seed}"
        )
        print("=" * 90)

        run_result = (
            run_static_utility_ga(
                data=data,
                seed=seed,
                config=GA_CONFIG,
            )
        )

        solution = (
            run_result
            .selected_solution
        )

        schedule = (
            solution.schedule
        )

        if not schedule.feasible:
            raise RuntimeError(
                f"Seed {seed} produced "
                "an infeasible final route."
            )

        if (
            schedule
            .missing_mandatory_ids
        ):
            raise RuntimeError(
                f"Seed {seed} omitted "
                "mandatory POIs."
            )

        route = list(
            solution.route
        )

        initial_routes_by_seed[
            str(seed)
        ] = route

        print(
            "Route:",
            " -> ".join(
                str(poi_id)
                for poi_id in route
            ),
        )

        print(
            "POI count:",
            solution
            .completed_poi_count,
        )

        print(
            "Achieved utility:",
            f"{solution.achieved_utility:.6f}",
        )

        print(
            "Travel time:",
            f"{schedule.total_travel_min:.2f}",
            "min",
        )

        print(
            "Total time used:",
            f"{schedule.time_used_min:.2f}",
            "min",
        )

        print(
            "Finish time:",
            schedule.finish_clock,
        )

        print(
            "Total cost:",
            f"RM{schedule.total_cost_rm:.2f}",
        )

        print(
            "Generations executed:",
            run_result
            .generations_executed,
        )

        print(
            "Convergence generation:",
            run_result
            .convergence_generation,
        )

        print(
            "Stopped by stagnation:",
            run_result
            .stopped_by_stagnation,
        )

        schedule_dataframe = (
            schedule_to_dataframe(
                run_result
            )
        )

        schedule_file = (
            OUTPUT_DIRECTORY
            / (
                f"seed_{seed}"
                "_schedule.csv"
            )
        )

        schedule_dataframe.to_csv(
            schedule_file,
            index=False,
            float_format="%.6f",
        )

        summary_rows.append(
            create_summary_row(
                data=data,
                run_result=run_result,
            )
        )

    summary_dataframe = pd.DataFrame(
        summary_rows
    )

    summary_file = (
        OUTPUT_DIRECTORY
        / "static_utility_seed_summary.csv"
    )

    summary_dataframe.to_csv(
        summary_file,
        index=False,
    )

    route_file = (
        OUTPUT_DIRECTORY
        / "initial_routes_by_seed.json"
    )

    route_file.write_text(
        json.dumps(
            initial_routes_by_seed,
            indent=2,
        ),
        encoding="utf-8",
    )

    configuration_file = (
        OUTPUT_DIRECTORY
        / "static_utility_ga_config.json"
    )

    configuration_file.write_text(
        json.dumps(
            asdict(GA_CONFIG),
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 90)
    print("STATIC UTILITY COMPLETED")
    print("=" * 90)

    print(
        "Summary file:",
        summary_file.resolve(),
    )

    print(
        "Initial routes file:",
        route_file.resolve(),
    )

    print(
        "GA configuration file:",
        configuration_file.resolve(),
    )

    print()
    print(
        summary_dataframe[
            [
                "seed",
                "poi_count",
                "achieved_utility",
                "total_travel_min",
                "total_time_used_min",
                "total_cost_rm",
                "generations_executed",
                "convergence_generation",
            ]
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()