from __future__ import annotations

from ga_core import GAConfig, run_static_utility_ga
from planning_core import load_planning_data


POI_FILE = "poi_data_calculated_final.csv"
MATRIX_FILE = "walking_time_matrix_final.csv"

TEST_SEED = 1001

GA_CONFIG = GAConfig(
    population_size=80,
    maximum_generations=120,
    tournament_size=3,
    crossover_probability=0.90,
    mutation_probability=0.20,
    elitism_count=2,
    stagnation_limit=15,
)


def extract_result(run_result) -> dict:
    solution = run_result.selected_solution
    schedule = solution.schedule

    return {
        "route": solution.route,
        "utility": round(
            solution.achieved_utility,
            6,
        ),
        "poi_count": (
            solution.completed_poi_count
        ),
        "travel_min": round(
            schedule.total_travel_min,
            6,
        ),
        "time_used_min": round(
            schedule.time_used_min,
            6,
        ),
        "cost_rm": round(
            schedule.total_cost_rm,
            2,
        ),
        "generations_executed": (
            run_result.generations_executed
        ),
        "convergence_generation": (
            run_result.convergence_generation
        ),
        "stopped_by_stagnation": (
            run_result.stopped_by_stagnation
        ),
    }


def main() -> None:
    data = load_planning_data(
        poi_file=POI_FILE,
        matrix_file=MATRIX_FILE,
    )

    first_run = run_static_utility_ga(
        data=data,
        seed=TEST_SEED,
        config=GA_CONFIG,
    )

    second_run = run_static_utility_ga(
        data=data,
        seed=TEST_SEED,
        config=GA_CONFIG,
    )

    first_result = extract_result(first_run)
    second_result = extract_result(second_run)

    print("=" * 70)
    print("STATIC UTILITY SAME-SEED REPRODUCIBILITY TEST")
    print("=" * 70)

    print("\nFirst run:")
    for key, value in first_result.items():
        print(f"  {key}: {value}")

    print("\nSecond run:")
    for key, value in second_result.items():
        print(f"  {key}: {value}")

    identical = first_result == second_result

    print()
    print("Identical results:", identical)

    if not identical:
        print("\nDifferences:")

        for key in first_result:
            if first_result[key] != second_result[key]:
                print(
                    f"  {key}: "
                    f"{first_result[key]} != "
                    f"{second_result[key]}"
                )

        raise SystemExit(
            "Same-seed reproducibility test failed."
        )

    print(
        "\nPASS: The same files, configuration "
        "and seed produced identical results."
    )


if __name__ == "__main__":
    main()