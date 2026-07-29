"""
Generate and inspect the 18 replanning states for Static Utility
seed 1001.

This test does not run the adaptive GA.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

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
    simulate_replanning_state,
)


POI_FILE = Path(
    "poi_data_calculated_final.csv"
)

MATRIX_FILE = Path(
    "walking_time_matrix_final.csv"
)

INITIAL_ROUTES_FILE = Path(
    "results/static_utility/"
    "initial_routes_by_seed.json"
)

OUTPUT_DIRECTORY = Path(
    "results/scenario_states"
)

TEST_SEED = 1001


def main() -> None:
    data = load_planning_data(
        poi_file=POI_FILE,
        matrix_file=MATRIX_FILE,
    )

    routes_by_seed = json.loads(
        INITIAL_ROUTES_FILE.read_text(
            encoding="utf-8"
        )
    )

    route = routes_by_seed[
        str(TEST_SEED)
    ]

    initial_schedule = decode_route(
        data=data,
        visit_sequence=route,

        start_poi_id=1,
        start_time_min=TRIP_START_MIN,
        trip_end_min=TRIP_END_MIN,

        day_code=EXPERIMENT_DAY,
        cost_budget_rm=COST_BUDGET_RM,

        start_poi_is_visit=True,

        required_mandatory_ids=(
            data.mandatory_ids
        ),
    )

    if not initial_schedule.feasible:
        raise RuntimeError(
            "The stored Static Utility route "
            "is infeasible."
        )

    rows: list[dict] = []

    for instance in (
        build_scenario_instances()
    ):
        state = (
            simulate_replanning_state(
                data=data,
                initial_schedule=(
                    initial_schedule
                ),
                scenario_instance=(
                    instance
                ),
            )
        )

        # Basic state invariants
        if (
            state.replan_start_min
            < state.trigger_absolute_min
        ):
            raise RuntimeError(
                f"{state.instance_id}: "
                "replanning starts before "
                "the trigger."
            )

        if len(
            state.completed_poi_ids
        ) != len(
            set(state.completed_poi_ids)
        ):
            raise RuntimeError(
                f"{state.instance_id}: "
                "duplicate completed POIs."
            )

        if not set(
            state.completed_poi_ids
        ).issubset(route):
            raise RuntimeError(
                f"{state.instance_id}: "
                "completed POIs are not part "
                "of the initial route."
            )

        row = state.to_dict()

        # Store list values clearly in the CSV.
        for field_name in (
            "completed_poi_ids",
            "original_remaining_poi_ids",
            "remaining_mandatory_ids",
            "unavailable_mandatory_ids",
            "unavailable_poi_ids",
        ):
            row[field_name] = json.dumps(
                list(row[field_name])
            )

        rows.append(row)

    if len(rows) != 18:
        raise RuntimeError(
            f"Expected 18 scenario states, "
            f"but generated {len(rows)}."
        )

    output_dataframe = pd.DataFrame(
        rows
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        OUTPUT_DIRECTORY
        / (
            f"seed_{TEST_SEED}"
            "_replanning_states.csv"
        )
    )

    output_dataframe.to_csv(
        output_file,
        index=False,
    )

    display_columns = [
        "instance_id",
        "trigger_clock",
        "committed_component",
        "current_poi_id",
        "current_poi_requires_visit",
        "replan_start_clock",
        "completed_poi_ids",
        "original_remaining_poi_ids",
        "remaining_mandatory_ids",
        "unavailable_poi_ids",
        "global_travel_multiplier",
        "thermal_multiplier",
    ]

    print("=" * 120)
    print(
        f"SCENARIO STATES - STATIC UTILITY "
        f"SEED {TEST_SEED}"
    )
    print("=" * 120)

    print(
        output_dataframe[
            display_columns
        ].to_string(index=False)
    )

    print()
    print(
        "Scenario-state file:",
        output_file.resolve(),
    )


if __name__ == "__main__":
    main()