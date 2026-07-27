"""
Static Distance baseline for TripMates Lite.

Selection priority:
1. Lowest walking time from the current POI
2. Higher utility when walking times are equal
3. Lower POI ID as the final tie-breaker

The planner does not simply test whether the next POI is feasible.
It also checks whether all remaining mandatory POIs can still be
completed after selecting that candidate.
"""

from __future__ import annotations

from dataclasses import asdict
from itertools import permutations
import json
from pathlib import Path

import pandas as pd

from planning_core import (
    COST_BUDGET_RM,
    EXPERIMENT_DAY,
    TRIP_END_MIN,
    TRIP_START_MIN,
    PlanningData,
    ScheduleResult,
    decode_route,
    load_planning_data,
    print_schedule,
)


# ================================================================
# File configuration
# ================================================================

POI_FILE = Path("poi_data_calculated_final.csv")
MATRIX_FILE = Path("walking_time_matrix_final.csv")

OUTPUT_DIRECTORY = Path("results")

OUTPUT_ROUTE_FILE = (
    OUTPUT_DIRECTORY
    / "static_distance_normal_route.csv"
)

OUTPUT_SUMMARY_FILE = (
    OUTPUT_DIRECTORY
    / "static_distance_normal_summary.json"
)


# ================================================================
# Mandatory feasibility checks
# ================================================================

def route_can_complete_mandatory_pois(
    data: PlanningData,
    route_prefix: list[int],
) -> bool:
    """
    Determine whether at least one ordering of the remaining
    mandatory POIs is feasible after the current route prefix.

    Because the mandatory set is small, all remaining mandatory
    permutations can be checked exactly.
    """

    required_mandatory_ids = set(
        data.mandatory_ids
    )

    visited_ids = set(route_prefix)

    remaining_mandatory_ids = sorted(
        required_mandatory_ids
        - visited_ids
    )

    # All mandatory POIs are already present.
    if not remaining_mandatory_ids:
        result = decode_route(
            data=data,
            visit_sequence=route_prefix,
            start_poi_id=1,
            start_time_min=TRIP_START_MIN,
            trip_end_min=TRIP_END_MIN,
            day_code=EXPERIMENT_DAY,
            cost_budget_rm=COST_BUDGET_RM,
            start_poi_is_visit=True,
            required_mandatory_ids=(
                required_mandatory_ids
            ),
        )

        return result.feasible

    # Test every possible order of the remaining mandatory POIs.
    for mandatory_order in permutations(
        remaining_mandatory_ids
    ):
        test_route = (
            route_prefix
            + list(mandatory_order)
        )

        result = decode_route(
            data=data,
            visit_sequence=test_route,
            start_poi_id=1,
            start_time_min=TRIP_START_MIN,
            trip_end_min=TRIP_END_MIN,
            day_code=EXPERIMENT_DAY,
            cost_budget_rm=COST_BUDGET_RM,
            start_poi_is_visit=True,
            required_mandatory_ids=(
                required_mandatory_ids
            ),
        )

        if result.feasible:
            return True

    return False


def candidate_is_feasible(
    data: PlanningData,
    current_route: list[int],
    candidate_poi_id: int,
) -> bool:
    """
    Check whether a candidate can be appended while preserving
    the feasibility of all remaining mandatory POIs.
    """

    if candidate_poi_id in current_route:
        return False

    candidate_route = (
        current_route
        + [candidate_poi_id]
    )

    return route_can_complete_mandatory_pois(
        data=data,
        route_prefix=candidate_route,
    )


# ================================================================
# Static Distance planner
# ================================================================

def build_static_distance_route(
    data: PlanningData,
    *,
    start_poi_id: int = 1,
) -> list[int]:
    """
    Construct a deterministic distance-first itinerary.

    At each step:
    1. Test every unvisited POI.
    2. Exclude candidates that would make the route or remaining
       mandatory POIs infeasible.
    3. Select the candidate with the lowest walking time.
    4. Break an exact distance tie using higher utility.
    5. Break any remaining tie using lower POI ID.
    """

    if start_poi_id not in data.pois:
        raise KeyError(
            f"Unknown starting POI: {start_poi_id}"
        )

    route = [start_poi_id]

    unvisited_ids = set(data.pois) - {
        start_poi_id
    }

    while unvisited_ids:
        current_poi_id = route[-1]

        feasible_candidates: list[
            tuple[float, float, int]
        ] = []

        for candidate_poi_id in sorted(
            unvisited_ids
        ):
            if not candidate_is_feasible(
                data=data,
                current_route=route,
                candidate_poi_id=(
                    candidate_poi_id
                ),
            ):
                continue

            travel_min = data.get_travel_time(
                current_poi_id,
                candidate_poi_id,
            )

            utility = (
                data
                .pois[candidate_poi_id]
                .utility_score
            )

            # Python selects the smallest tuple.
            #
            # travel_min:
            #     smaller is preferred
            #
            # -utility:
            #     higher utility becomes a smaller
            #     negative value
            #
            # candidate_poi_id:
            #     lower ID is preferred
            feasible_candidates.append(
                (
                    travel_min,
                    -utility,
                    candidate_poi_id,
                )
            )

        if not feasible_candidates:
            break

        selected = min(
            feasible_candidates
        )

        selected_poi_id = selected[2]

        route.append(selected_poi_id)
        unvisited_ids.remove(
            selected_poi_id
        )

    missing_mandatory_ids = (
        set(data.mandatory_ids)
        - set(route)
    )

    if missing_mandatory_ids:
        raise RuntimeError(
            "Static Distance failed to include "
            "mandatory POIs: "
            f"{sorted(missing_mandatory_ids)}"
        )

    return route


# ================================================================
# Output generation
# ================================================================

def calculate_total_utility(
    data: PlanningData,
    route: list[int],
) -> float:
    return sum(
        data.pois[poi_id].utility_score
        for poi_id in route
    )


def schedule_to_dataframe(
    result: ScheduleResult,
) -> pd.DataFrame:
    rows = []

    for entry in result.entries:
        row = asdict(entry)

        # Keep POI IDs as nullable integers so that the first
        # previous_poi_id is blank rather than displayed as NaN.
        rows.append(row)

    dataframe = pd.DataFrame(rows)

    if not dataframe.empty:
        dataframe["previous_poi_id"] = (
            dataframe["previous_poi_id"]
            .astype("Int64")
        )

    return dataframe


def save_static_distance_results(
    data: PlanningData,
    route: list[int],
    result: ScheduleResult,
) -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    schedule_dataframe = (
        schedule_to_dataframe(result)
    )

    schedule_dataframe.to_csv(
        OUTPUT_ROUTE_FILE,
        index=False,
        float_format="%.6f",
    )

    route_names = [
        data.pois[poi_id].name
        for poi_id in route
    ]

    total_utility = (
        calculate_total_utility(
            data=data,
            route=route,
        )
    )

    summary = {
        "planner": "Static Distance",
        "condition": "normal",
        "deterministic": True,

        "start_poi_id": 1,

        "route": route,
        "route_names": route_names,

        "poi_count": len(route),

        "mandatory_poi_ids": sorted(
            data.mandatory_ids
        ),

        "mandatory_complete": (
            set(data.mandatory_ids)
            <= set(route)
        ),

        "feasible": result.feasible,
        "failure_reason": (
            result.failure_reason
        ),

        "finish_time": (
            result.finish_clock
        ),

        "total_travel_min": round(
            result.total_travel_min,
            6,
        ),

        "total_waiting_min": round(
            result.total_waiting_min,
            6,
        ),

        "total_visit_min": round(
            result.total_visit_min,
            6,
        ),

        "total_time_used_min": round(
            result.time_used_min,
            6,
        ),

        "total_cost_rm": round(
            result.total_cost_rm,
            2,
        ),

        "total_utility": round(
            total_utility,
            6,
        ),
    }

    OUTPUT_SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ================================================================
# Main execution
# ================================================================

def main() -> None:
    data = load_planning_data(
        poi_file=POI_FILE,
        matrix_file=MATRIX_FILE,
    )

    route = build_static_distance_route(
        data=data,
        start_poi_id=1,
    )

    result = decode_route(
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

    if not result.feasible:
        raise RuntimeError(
            "The generated Static Distance route "
            "failed final feasibility validation:\n"
            f"{result.failure_reason}"
        )

    print()
    print("STATIC DISTANCE BASELINE")
    print("=" * 90)

    print(
        "Route:",
        " -> ".join(
            str(poi_id)
            for poi_id in route
        ),
    )

    print(
        "POI names:",
        " -> ".join(
            data.pois[poi_id].name
            for poi_id in route
        ),
    )

    print(
        "Total utility:",
        f"{calculate_total_utility(data, route):.6f}",
    )

    print_schedule(result)

    save_static_distance_results(
        data=data,
        route=route,
        result=result,
    )

    print()
    print(
        "Route file:",
        OUTPUT_ROUTE_FILE.resolve(),
    )

    print(
        "Summary file:",
        OUTPUT_SUMMARY_FILE.resolve(),
    )


if __name__ == "__main__":
    main()