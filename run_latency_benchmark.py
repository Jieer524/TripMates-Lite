"""
RQ3 latency benchmark.

Measures replanning latency for:

- Greedy Repair
- Adaptive Without Thermal
- TripMates Lite

Protocol
--------
For every applicable disruption scenario:
1. one untimed warm-up;
2. ``--repetitions`` timed repetitions (default 30) using the fixed
   seed 1001;
3. timing with ``time.perf_counter()``;
4. file-writing and chart-generation are excluded from the timing.

Two measured scopes are reported separately and never mixed:

- ``engine_only``          : the planner entry-point call, with the
  planning data already loaded and the shared disruption state
  pre-built.
- ``end_to_end_request``   : shared disruption-state construction plus
  the planner call, with the planning data already loaded (i.e. an
  in-process request after dataset load).

The dataset-loading cost is intentionally excluded from both scopes and
reported once separately as ``dataset_load_seconds``.

Two planner settings are distinguished:

- ``research_benchmark``: the frozen research GA configuration.
- ``quick_preview``     : a reduced interactive configuration, labelled
  explicitly and never presented as a research benchmark.

Greedy Repair has no GA parameters; it is measured once and labelled
``deterministic_no_ga``.

Outputs (results/latency_benchmark/):
- latency_raw_measurements.csv
- latency_summary.csv
- latency_run_config.json

Usage:
    python run_latency_benchmark.py                 # full
    python run_latency_benchmark.py --smoke          # smoke only
"""

from __future__ import annotations

from dataclasses import asdict
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_ga_core import (
    REPLANNING_GA_CONFIG,
    prepare_adaptive_problem,
    run_adaptive_without_thermal_ga,
)
from benchmark_common import (
    ALL_INSTANCE_IDS,
    MATRIX_FILE,
    POI_FILE,
    SEEDS,
    build_initial_schedules,
    build_scenario_lookup,
    file_fingerprint,
    load_initial_routes,
    runtime_metadata,
    utc_timestamp,
    write_json,
)
from ga_core import GAConfig
from greedy_repair import run_greedy_repair
from planning_core import (
    PlanningConstraints,
    load_planning_data,
)
from scenario_core import simulate_replanning_state
from tripmates_lite_ga import run_tripmates_lite_ga


OUTPUT_DIRECTORY = Path("results/latency_benchmark")

RAW_FILE = OUTPUT_DIRECTORY / "latency_raw_measurements.csv"
SUMMARY_FILE = OUTPUT_DIRECTORY / "latency_summary.csv"
RUN_CONFIG_FILE = OUTPUT_DIRECTORY / "latency_run_config.json"


def isolate_smoke_outputs() -> None:
    """Redirect a ``--smoke`` run into an isolated sub-folder.

    Smoke runs must never overwrite a completed full benchmark, so a
    ``--smoke`` invocation writes every artefact under
    ``<output-directory>/smoke/`` instead of the shared full-run paths.
    """
    global OUTPUT_DIRECTORY, RAW_FILE, SUMMARY_FILE, RUN_CONFIG_FILE

    smoke_directory = OUTPUT_DIRECTORY / "smoke"

    RAW_FILE = smoke_directory / RAW_FILE.name
    SUMMARY_FILE = smoke_directory / SUMMARY_FILE.name
    RUN_CONFIG_FILE = smoke_directory / RUN_CONFIG_FILE.name

    OUTPUT_DIRECTORY = smoke_directory
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)


PLANNER_GREEDY = "Greedy Repair"
PLANNER_ADAPTIVE = "Adaptive Without Thermal"
PLANNER_TRIPMATES = "TripMates Lite"

FIXED_SEED = 1001
DEFAULT_REPETITIONS = 30

RESEARCH_SETTING = "research_benchmark"
QUICK_SETTING = "quick_preview"
GREEDY_SETTING = "deterministic_no_ga"

QUICK_PREVIEW_GA_CONFIG = GAConfig(
    population_size=12,
    maximum_generations=8,
    tournament_size=3,
    crossover_probability=0.90,
    mutation_probability=0.20,
    elitism_count=2,
    stagnation_limit=3,
)


def execute_planner(
    *,
    planner: str,
    data,
    initial_schedule,
    problem,
    scenario_instance,
    constraints,
    config,
):
    if planner == PLANNER_GREEDY:
        return run_greedy_repair(
            data=data,
            initial_schedule=initial_schedule,
            state=problem.state,
            scenario_instance=scenario_instance,
            constraints=constraints,
            seed=FIXED_SEED,
        )

    if planner == PLANNER_ADAPTIVE:
        return run_adaptive_without_thermal_ga(
            data=data,
            problem=problem,
            seed=FIXED_SEED,
            config=config,
        )

    return run_tripmates_lite_ga(
        data=data,
        initial_schedule=initial_schedule,
        problem=problem,
        seed=FIXED_SEED,
        config=config,
    )


def measure(
    *,
    planner: str,
    setting: str,
    data,
    scenario_instance,
    initial_schedule,
    constraints,
    config,
    repetitions: int,
) -> tuple[list[dict], list[dict]]:
    """Return (raw rows, summary rows) for one planner/scenario."""

    raw_rows: list[dict] = []

    def build_state_and_problem():
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
        return problem

    # --------------------------------------------------------
    # Untimed warm-up
    # --------------------------------------------------------

    warm_problem = build_state_and_problem()
    execute_planner(
        planner=planner,
        data=data,
        initial_schedule=initial_schedule,
        problem=warm_problem,
        scenario_instance=scenario_instance,
        constraints=constraints,
        config=config,
    )

    engine_times: list[float] = []
    end_to_end_times: list[float] = []
    preparation_times: list[float] = []
    failures = 0

    for repetition in range(repetitions):
        request_start = time.perf_counter()

        problem = build_state_and_problem()

        engine_start = time.perf_counter()

        try:
            result = execute_planner(
                planner=planner,
                data=data,
                initial_schedule=initial_schedule,
                problem=problem,
                scenario_instance=scenario_instance,
                constraints=constraints,
                config=config,
            )
            failed = _is_failure(planner, result)
        except (RuntimeError, ValueError, KeyError):
            failed = True

        engine_end = time.perf_counter()

        if failed:
            failures += 1

        engine_seconds = engine_end - engine_start
        preparation_seconds = engine_start - request_start
        end_to_end_seconds = engine_end - request_start

        engine_times.append(engine_seconds)
        preparation_times.append(preparation_seconds)
        end_to_end_times.append(end_to_end_seconds)

        raw_rows.append(
            {
                "planner": planner,
                "setting": setting,
                "instance_id": scenario_instance.instance_id,
                "seed": FIXED_SEED,
                "repetition": repetition,
                "failed": failed,
                "engine_only_seconds": round(
                    engine_seconds, 9
                ),
                "preparation_seconds": round(
                    preparation_seconds, 9
                ),
                "end_to_end_request_seconds": round(
                    end_to_end_seconds, 9
                ),
            }
        )

    summary_rows: list[dict] = []

    for scope, values in (
        ("engine_only", engine_times),
        (
            "end_to_end_request",
            end_to_end_times,
        ),
        ("shared_preparation", preparation_times),
    ):
        array = np.asarray(values, dtype=float)

        summary_rows.append(
            {
                "planner": planner,
                "setting": setting,
                "scope": scope,
                "instance_id": (
                    scenario_instance.instance_id
                ),
                "repetition_count": int(len(array)),
                "median_seconds": float(
                    np.median(array)
                ),
                "p95_seconds": float(
                    np.percentile(array, 95)
                ),
                "mean_seconds": float(array.mean()),
                "std_seconds": float(array.std(ddof=0)),
                "min_seconds": float(array.min()),
                "max_seconds": float(array.max()),
                "failure_count": int(failures),
                "measured_scope": scope,
            }
        )

    return raw_rows, summary_rows


def _is_failure(planner: str, result) -> bool:
    if planner == PLANNER_GREEDY:
        return result.status not in {"success", "no_action"}
    return False


def build_overall_summary(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    grouped = summary.groupby(
        ["planner", "setting", "scope"], sort=True
    )

    for (planner, setting, scope), group in grouped:
        total_reps = int(
            group["repetition_count"].sum()
        )
        rows.append(
            {
                "planner": planner,
                "setting": setting,
                "scope": scope,
                "instance_id": "ALL",
                "repetition_count": total_reps,
                "median_seconds": float(
                    group["median_seconds"].median()
                ),
                "p95_seconds": float(
                    group["p95_seconds"].max()
                ),
                "mean_seconds": float(
                    group["mean_seconds"].mean()
                ),
                "std_seconds": float(
                    group["std_seconds"].mean()
                ),
                "min_seconds": float(
                    group["min_seconds"].min()
                ),
                "max_seconds": float(
                    group["max_seconds"].max()
                ),
                "failure_count": int(
                    group["failure_count"].sum()
                ),
                "measured_scope": scope,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--repetitions", type=int, default=None
    )
    args = parser.parse_args()

    if args.smoke:
        isolate_smoke_outputs()

    repetitions = (
        args.repetitions
        if args.repetitions is not None
        else (3 if args.smoke else DEFAULT_REPETITIONS)
    )

    if args.smoke:
        instance_ids = ("S1_nominal",)
        settings = {
            PLANNER_GREEDY: [
                (GREEDY_SETTING, None)
            ],
            PLANNER_ADAPTIVE: [
                (RESEARCH_SETTING, REPLANNING_GA_CONFIG),
                (QUICK_SETTING, QUICK_PREVIEW_GA_CONFIG),
            ],
            PLANNER_TRIPMATES: [
                (RESEARCH_SETTING, REPLANNING_GA_CONFIG),
                (QUICK_SETTING, QUICK_PREVIEW_GA_CONFIG),
            ],
        }
    else:
        instance_ids = ALL_INSTANCE_IDS
        settings = {
            PLANNER_GREEDY: [
                (GREEDY_SETTING, None)
            ],
            PLANNER_ADAPTIVE: [
                (RESEARCH_SETTING, REPLANNING_GA_CONFIG),
                (QUICK_SETTING, QUICK_PREVIEW_GA_CONFIG),
            ],
            PLANNER_TRIPMATES: [
                (RESEARCH_SETTING, REPLANNING_GA_CONFIG),
                (QUICK_SETTING, QUICK_PREVIEW_GA_CONFIG),
            ],
        }

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    load_start = time.perf_counter()
    data = load_planning_data(
        poi_file=POI_FILE,
        matrix_file=MATRIX_FILE,
    )
    dataset_load_seconds = time.perf_counter() - load_start

    constraints = PlanningConstraints()

    routes_by_seed = load_initial_routes()
    initial_schedules = build_initial_schedules(
        data, routes_by_seed
    )
    scenario_lookup = build_scenario_lookup()

    initial_schedule = initial_schedules[FIXED_SEED]

    raw_rows: list[dict] = []
    summary_rows: list[dict] = []

    for instance_id in instance_ids:
        scenario_instance = scenario_lookup[instance_id]

        for planner, planner_settings in settings.items():
            for setting, config in planner_settings:
                print(
                    f"Measuring {planner} [{setting}] "
                    f"on {instance_id} ..."
                )

                run_raw, run_summary = measure(
                    planner=planner,
                    setting=setting,
                    data=data,
                    scenario_instance=scenario_instance,
                    initial_schedule=initial_schedule,
                    constraints=constraints,
                    config=config,
                    repetitions=repetitions,
                )

                raw_rows.extend(run_raw)
                summary_rows.extend(run_summary)

    raw = pd.DataFrame(raw_rows)
    summary = pd.DataFrame(summary_rows)

    overall = build_overall_summary(summary)
    summary = pd.concat(
        [summary, overall], ignore_index=True
    )

    raw.to_csv(
        RAW_FILE, index=False, float_format="%.9f"
    )
    summary.to_csv(
        SUMMARY_FILE, index=False, float_format="%.9f"
    )

    run_config = {
        "experiment": "RQ3 latency benchmark",
        "mode": (
            "smoke_test" if args.smoke else "research_benchmark"
        ),
        "timestamp_utc": utc_timestamp(),

        "planners": list(settings.keys()),
        "instance_ids": list(instance_ids),
        "official_instance_count": len(
            ALL_INSTANCE_IDS
        ),
        "fixed_seed": FIXED_SEED,
        "repetitions": repetitions,
        "warmup_count": 1,
        "timer": "time.perf_counter",

        "measured_scopes": {
            "engine_only": (
                "Planner entry-point call, planning data loaded, "
                "shared disruption state pre-built."
            ),
            "end_to_end_request": (
                "Shared disruption-state construction plus the "
                "planner call; planning data already loaded."
            ),
            "shared_preparation": (
                "simulate_replanning_state + "
                "prepare_adaptive_problem only."
            ),
        },

        "dataset_loading_policy": (
            "Dataset loading is excluded from both timed scopes "
            "and reported separately as dataset_load_seconds."
        ),
        "dataset_load_seconds": round(
            dataset_load_seconds, 6
        ),

        "settings": {
            "research_benchmark": asdict(
                REPLANNING_GA_CONFIG
            ),
            "quick_preview": asdict(
                QUICK_PREVIEW_GA_CONFIG
            ),
            "quick_preview_note": (
                "Reduced interactive configuration; never "
                "presented as a research benchmark."
            ),
            "greedy_deterministic_no_ga": (
                "Greedy Repair has no GA parameters."
            ),
        },

        "greedy_preparation_note": (
            "Greedy Repair builds its own shared candidate window "
            "inside its entry-point call, so its engine_only time "
            "includes that preparation; the GA planners receive a "
            "pre-built problem. The shared_preparation scope makes "
            "this explicit."
        ),

        "dataset": file_fingerprint(POI_FILE),
        "matrix": file_fingerprint(MATRIX_FILE),
        "initial_routes": file_fingerprint(
            Path(
                "results/static_utility/"
                "initial_routes_by_seed.json"
            )
        ),

        "runtime_metadata": runtime_metadata(),
    }

    write_json(RUN_CONFIG_FILE, run_config)

    print()
    print("=" * 100)
    print("RQ3 LATENCY SUMMARY (overall, engine_only)")
    print("=" * 100)

    engine_overall = summary[
        (summary["scope"] == "engine_only")
        & (summary["instance_id"] == "ALL")
    ]

    print(
        engine_overall[
            [
                "planner",
                "setting",
                "median_seconds",
                "p95_seconds",
                "mean_seconds",
                "std_seconds",
                "failure_count",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        "Dataset load seconds:",
        f"{dataset_load_seconds:.4f}",
    )
    print("Raw measurements:", RAW_FILE.resolve())
    print("Summary:", SUMMARY_FILE.resolve())
    print("Run config:", RUN_CONFIG_FILE.resolve())


if __name__ == "__main__":
    main()
