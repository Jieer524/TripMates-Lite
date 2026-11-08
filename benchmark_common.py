"""
Shared, read-only helpers for TripMates Lite benchmark runners.

This module contains only orchestration plumbing: frozen input paths,
scenario/seed enumeration, reproducibility metadata, run-configuration
writing and the common metric schema.  It contains no planning or
exposure logic of its own; every planner is invoked through its
existing interface.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

from dataclasses import dataclass

from exposure_core import ExposureMetrics
from planning_core import (
    COST_BUDGET_RM,
    EXPERIMENT_DAY,
    TRIP_END_MIN,
    TRIP_START_MIN,
    PlanningConstraints,
    PlanningData,
    ScheduleResult,
    decode_route,
)
from scenario_core import (
    ScenarioInstance,
    build_scenario_instances,
)
from stability_metrics import (
    StabilityMetrics,
    StabilityWeightConfig,
    compute_stability_metrics,
)


ROOT = Path(__file__).resolve().parent

POI_FILE = ROOT / "poi_data_calculated_final.csv"
MATRIX_FILE = ROOT / "walking_time_matrix_final.csv"
INITIAL_ROUTES_FILE = (
    ROOT
    / "results/static_utility/initial_routes_by_seed.json"
)

SEEDS = tuple(range(1001, 1011))

WEATHER_INSTANCE_IDS = (
    "S1_early",
    "S1_nominal",
    "S1_late",
    "S5_early",
    "S5_nominal",
    "S5_late",
    "S6_early",
    "S6_nominal",
    "S6_late",
)

# The repository's official scenario set: six frozen disruption
# families, each with early / nominal / late trigger variants.
ALL_INSTANCE_IDS = tuple(
    instance.instance_id
    for instance in build_scenario_instances()
)

C2_ALPHAS = (1.00, 0.95, 0.90, 0.85)

TOLERANCE = 1e-9


# ================================================================
# Reproducibility metadata
# ================================================================

def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_fingerprint(path: Path) -> dict:
    resolved = Path(path)
    digest = hashlib.sha256(
        resolved.read_bytes()
    ).hexdigest()

    return {
        "file": resolved.name,
        "sha256": digest,
        "bytes": resolved.stat().st_size,
    }


def runtime_metadata() -> dict:
    import numpy

    metadata = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "numpy_version": numpy.__version__,
    }

    try:
        import scipy

        metadata["scipy_version"] = scipy.__version__
    except ImportError:  # pragma: no cover
        metadata["scipy_version"] = None

    metadata["pandas_version"] = pd.__version__

    return metadata


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ================================================================
# Frozen inputs
# ================================================================

def load_initial_routes() -> dict[int, list[int]]:
    raw = json.loads(
        INITIAL_ROUTES_FILE.read_text(encoding="utf-8")
    )

    return {
        int(seed): [int(poi_id) for poi_id in route]
        for seed, route in raw.items()
    }


def build_initial_schedules(
    data: PlanningData,
    routes_by_seed: dict[int, list[int]],
) -> dict[int, ScheduleResult]:
    schedules: dict[int, ScheduleResult] = {}

    for seed, route in routes_by_seed.items():
        schedule = decode_route(
            data=data,
            visit_sequence=route,
            start_poi_id=1,
            start_time_min=TRIP_START_MIN,
            trip_end_min=TRIP_END_MIN,
            day_code=EXPERIMENT_DAY,
            cost_budget_rm=COST_BUDGET_RM,
            start_poi_is_visit=True,
            required_mandatory_ids=data.mandatory_ids,
        )

        if not schedule.feasible:
            raise RuntimeError(
                f"Seed {seed}: the frozen initial route is infeasible."
            )

        schedules[seed] = schedule

    return schedules


def build_scenario_lookup() -> dict[str, ScenarioInstance]:
    return {
        instance.instance_id: instance
        for instance in build_scenario_instances()
    }


# ================================================================
# Encoding / numeric helpers
# ================================================================

def encode_ids(values: Iterable[int]) -> str:
    return json.dumps([int(value) for value in values])


def safe_ratio(
    numerator: float,
    denominator: float,
) -> float | None:
    if abs(denominator) <= TOLERANCE:
        return None
    return numerator / denominator


def route_utility(
    data: PlanningData,
    route: Iterable[int],
) -> float:
    return sum(
        data.pois[poi_id].utility_score
        for poi_id in route
    )


# ================================================================
# Static (non-adaptive) post-disruption evaluation
# ================================================================

@dataclass(frozen=True)
class StaticSuffixEvaluation:
    """Result of executing the remaining original sequence unchanged.

    Static planners never reorder, substitute or optimise after a
    disruption.  They execute the remaining original sequence and
    report infeasibility when the disrupted plan cannot be completed.
    """

    feasible: bool
    failure_reason: str | None
    repaired_route: tuple[int, ...]
    schedule: ScheduleResult | None
    executed_poi_count: int


def evaluate_static_suffix(
    *,
    data: PlanningData,
    state,
    scenario_instance: ScenarioInstance,
    constraints: PlanningConstraints,
) -> StaticSuffixEvaluation:
    from adaptive_ga_core import decode_replanned_route

    remaining = tuple(
        state.original_remaining_poi_ids
    )

    result = decode_replanned_route(
        data=data,
        state=state,
        scenario_instance=scenario_instance,
        route=remaining,
        constraints=constraints,
    )

    if result.feasible:
        return StaticSuffixEvaluation(
            feasible=True,
            failure_reason=None,
            repaired_route=remaining,
            schedule=result,
            executed_poi_count=len(
                result.completed_sequence
            ),
        )

    return StaticSuffixEvaluation(
        feasible=False,
        failure_reason=result.failure_reason,
        repaired_route=tuple(result.completed_sequence),
        schedule=None,
        executed_poi_count=len(
            result.completed_sequence
        ),
    )


# ================================================================
# Common metric schema
# ================================================================

def build_metric_record(
    *,
    planner: str,
    scenario_instance: ScenarioInstance,
    seed: int | None,
    initial_route: list[int],
    initial_schedule: ScheduleResult,
    initial_route_utility: float,
    state,
    repaired_route: list[int] | None,
    achieved_utility: float | None,
    completed_poi_count: int | None,
    total_travel_min: float | None,
    locked_travel_min: float | None,
    repaired_schedule: ScheduleResult | None,
    exposure: ExposureMetrics | None,
    feasible: bool,
    failure_reason: str | None,
    mandatory_complete: bool | None,
    runtime_seconds: float,
    deterministic: bool,
    generations_executed: int | None = None,
    convergence_generation: int | None = None,
    evaluated_chromosomes: int | None = None,
    unique_decoded_routes: int | None = None,
    pareto_front_size: int | None = None,
    repair_record=None,
    stability_weights: StabilityWeightConfig | None = None,
    extra: dict | None = None,
) -> dict:
    """Return one normalised RQ1 run record.

    Besides the outcome metrics, the record carries a deterministic
    itinerary-stability block (see ``stability_metrics``): preserved /
    removed / added POIs, changed activities, time shifts, order
    deviation, the weighted edit distance and the locked-prefix verdict.
    The block is computed here, once, so every runner and every planner
    reports the same definition.
    """

    completed_prefix = list(state.completed_poi_ids)

    final_route = (
        completed_prefix + list(repaired_route)
        if repaired_route is not None
        else completed_prefix
    )

    original_remaining = list(
        state.original_remaining_poi_ids
    )

    record = {
        "planner": planner,
        "scenario_id": scenario_instance.scenario.scenario_id,
        "scenario_name": scenario_instance.scenario.name,
        "instance_id": scenario_instance.instance_id,
        "trigger_variant": scenario_instance.trigger_variant,
        "seed": seed,
        "deterministic": deterministic,

        "initial_route": encode_ids(initial_route),
        "completed_prefix": encode_ids(completed_prefix),
        "original_remaining_route": encode_ids(
            original_remaining
        ),
        "final_route": encode_ids(final_route),
        "repaired_route": (
            encode_ids(repaired_route)
            if repaired_route is not None
            else ""
        ),
        "original_suffix": encode_ids(original_remaining),
        "replanned_suffix": (
            encode_ids(repaired_route)
            if repaired_route is not None
            else ""
        ),

        "feasible": feasible,
        "failure_reason": failure_reason or "",
        "mandatory_complete": mandatory_complete,

        "itinerary_completion_ratio": (
            safe_ratio(
                completed_poi_count or 0,
                len(initial_route),
            )
        ),

        "original_poi_retention": (
            safe_ratio(
                len(
                    set(final_route)
                    & set(initial_route)
                ),
                len(initial_route),
            )
        ),

        "achieved_utility": achieved_utility,
        "initial_route_utility": initial_route_utility,
        "utility_retention_ratio": (
            None
            if achieved_utility is None
            else safe_ratio(
                achieved_utility,
                initial_route_utility,
            )
        ),
        "completed_poi_count": completed_poi_count,

        "total_travel_min": total_travel_min,
        "locked_travel_min": locked_travel_min,
        "repaired_travel_min": (
            None
            if repaired_schedule is None
            else repaired_schedule.total_travel_min
        ),
        "travel_overhead_min": (
            None
            if total_travel_min is None
            else round(
                total_travel_min
                - initial_schedule.total_travel_min,
                6,
            )
        ),
        "repaired_waiting_min": (
            None
            if repaired_schedule is None
            else repaired_schedule.total_waiting_min
        ),
        "total_cost_rm": (
            None
            if repaired_schedule is None
            else repaired_schedule.total_cost_rm
        ),

        "outdoor_walking_min": (
            None
            if exposure is None
            else exposure.outdoor_walking_min
        ),
        "outdoor_visit_min": (
            None
            if exposure is None
            else exposure.outdoor_visit_min
        ),
        "total_outdoor_exposure_min": (
            None
            if exposure is None
            else exposure.total_outdoor_exposure_min
        ),
        "wes": (
            None
            if exposure is None
            else exposure.weather_weighted_exposure_score
        ),
        "weather_source": (
            None if exposure is None else exposure.weather_source
        ),
        "weather_fallback_reason": (
            None if exposure is None else exposure.weather_fallback_reason
        ),
        "total_outdoor_exposure_min": (
            None if exposure is None else exposure.total_outdoor_exposure_min
        ),
        "high_heat_outdoor_exposure_min": (
            None if exposure is None else exposure.high_heat_outdoor_exposure_min
        ),
        "maximum_continuous_outdoor_exposure_min": (
            None if exposure is None else exposure.maximum_continuous_outdoor_exposure_min
        ),
        "exposure_during_travel_min": (
            None if exposure is None else exposure.exposure_during_travel_min
        ),
        "exposure_during_visits_min": (
            None if exposure is None else exposure.exposure_during_visits_min
        ),
        "exposure_during_waiting_min": (
            None if exposure is None else exposure.exposure_during_waiting_min
        ),

        "route_changed": (
            repaired_route is not None
            and list(repaired_route)
            != original_remaining
        ),
        "removed_poi_count": (
            None
            if repaired_route is None
            else len(
                set(original_remaining)
                - set(repaired_route)
            )
        ),
        "added_poi_count": (
            None
            if repaired_route is None
            else len(
                set(repaired_route)
                - set(original_remaining)
            )
        ),
        "retained_poi_count": (
            None
            if repaired_route is None
            else len(
                set(original_remaining)
                & set(repaired_route)
            )
        ),

        "generations_executed": generations_executed,
        "convergence_generation": convergence_generation,
        "evaluated_chromosomes": evaluated_chromosomes,
        "unique_decoded_routes": unique_decoded_routes,
        "pareto_front_size": pareto_front_size,

        "runtime_seconds": round(runtime_seconds, 6),

        # ----------------------------------------------------------
        # Bounded-repair audit fields
        # ----------------------------------------------------------

        "bounded_repair_limit": (
            None
            if repair_record is None
            else repair_record.bounded_repair_limit
        ),
        "actual_modification_count": (
            None
            if repair_record is None
            else repair_record.actual_modification_count
        ),
        "locked_prefix": encode_ids(
            state.completed_poi_ids
        ),
        "current_activity": state.committed_component,
        "modified_suffix": (
            encode_ids(repaired_route)
            if repaired_route is not None
            else ""
        ),
        "retained_activity_count": (
            None
            if repair_record is None
            else len(repair_record.retained_activities)
        ),
        "removed_activity_count": (
            None
            if repair_record is None
            else len(repair_record.removed_activities)
        ),
        "replacement_count": (
            None
            if repair_record is None
            else len(repair_record.replacements)
        ),
        "removed_pois_with_reasons": (
            []
            if repair_record is None
            else [
                d.to_dict()
                for d in repair_record.removed_activities
            ]
        ),
        "replacement_pois_with_reasons": (
            []
            if repair_record is None
            else [
                d.to_dict()
                for d in repair_record.replacements
            ]
        ),
        "repair_failure_status": (
            None
            if repair_record is None
            else repair_record.failure_status
        ),
        "repair_failure_reason": (
            None
            if repair_record is None
            else repair_record.failure_reason
        ),
    }

    # ----------------------------------------------------------
    # Itinerary-stability block
    #
    # One shared definition for every planner.  The comparison is
    # always original suffix -> repaired suffix from the same
    # post-trigger state, and the locked prefix is verified rather
    # than assumed.
    # ----------------------------------------------------------

    stability: StabilityMetrics = compute_stability_metrics(
        original_suffix=original_remaining,
        replanned_suffix=(
            () if repaired_route is None else tuple(repaired_route)
        ),
        locked_prefix=completed_prefix,
        original_schedule=initial_schedule,
        replanned_schedule=repaired_schedule,
        repair_record=repair_record,
        weights=stability_weights,
        final_route=final_route,
        replanning_feasible=feasible,
    )

    record.update(stability.to_dict())

    if extra:
        record.update(extra)

    return record
