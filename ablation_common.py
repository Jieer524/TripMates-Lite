"""
Shared infrastructure for the TripMates Lite ablation and sensitivity
studies.

Five controlled experiments are built on top of this module:

1. ``run_stability_ablation.py``          - selection-key ablation
2. ``run_full_replanning_ablation.py``    - bounded vs full replanning
3. ``run_severity_sensitivity.py``        - light / medium / severe
4. ``run_density_sensitivity.py``         - relaxed / moderate / dense
5. ``run_weather_profile_sensitivity.py`` - normal / hot / very hot / missing

Everything that must be *identical* across those studies lives here:

- one ``run_cell`` that invokes every planner on exactly the same
  inputs and returns ``benchmark_common.build_metric_record`` rows;
- explicit failure records (a failed run is a row, never a dropped row);
- factor definitions with the exact numeric parameter values;
- study output paths that isolate ``--smoke`` automatically;
- the paired-statistics helpers used by the analysis script.

Design rules
------------
- No planner is re-implemented.  Static, Greedy and the two GA planners
  are called through the same record builders the RQ1 benchmark uses,
  so a metric means the same thing in every study.
- Every factor level is a frozen dataclass instance, so the manifest can
  be dumped from the objects that actually drove the run.
- ``--smoke`` never shares a directory with a full run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import time
from typing import Iterable, Sequence

import pandas as pd

from adaptive_ga_core import (
    BoundedRepairConfig,
    REPLANNING_GA_CONFIG,
    prepare_adaptive_problem,
    run_adaptive_without_thermal_ga,
)
from benchmark_common import (
    MATRIX_FILE,
    POI_FILE,
    SEEDS,
    build_initial_schedules,
    build_metric_record,
    build_scenario_lookup,
    file_fingerprint,
    load_initial_routes,
    route_utility,
    runtime_metadata,
    utc_timestamp,
    write_json,
)
from ga_core import build_mandatory_backbone
from planning_core import (
    PlanningConstraints,
    PlanningData,
    ScheduleResult,
    decode_route,
    format_clock,
    load_planning_data,
)
from scenario_core import (
    ScenarioDefinition,
    ScenarioInstance,
    build_scenario_instances,
    simulate_replanning_state,
)
from stability_metrics import StabilityWeightConfig
from tripmates_lite_ga import run_tripmates_lite_ga
from weather_service import (
    WeatherProfile,
    WeatherServiceAdapter,
    predefined_weather_profile,
)

# The record builders are imported rather than re-implemented: a metric
# must mean exactly the same thing here as in the RQ1 benchmark.
from run_rq1_full_benchmark import (
    PLANNER_ADAPTIVE,
    PLANNER_GREEDY,
    PLANNER_STATIC_DISTANCE,
    PLANNER_STATIC_UTILITY,
    PLANNER_TRIPMATES,
    PLANNERS as ALL_PLANNERS,
    ga_planner_record,
    greedy_planner_record,
    static_planner_record,
)
from static_distance_baseline import build_static_distance_route


# ================================================================
# Study identifiers and output layout
# ================================================================

STUDY_STABILITY = "stability_ablation"
STUDY_FULL_REPLANNING = "full_replanning_ablation"
STUDY_SEVERITY = "disruption_severity_sensitivity"
STUDY_DENSITY = "itinerary_density_sensitivity"
STUDY_WEATHER = "weather_profile_sensitivity"

STUDIES = (
    STUDY_STABILITY,
    STUDY_FULL_REPLANNING,
    STUDY_SEVERITY,
    STUDY_DENSITY,
    STUDY_WEATHER,
)

RESULTS_ROOT = Path("results/ablation_studies")

# Every study in this package is an exploratory sensitivity analysis.
# The confirmatory evidence remains the frozen RQ1 benchmark and the
# frozen weather ablation; nothing here may be reported as confirmatory.
RESULT_CLASS_EXPLORATORY = "exploratory_sensitivity"


def study_directory(
    study: str,
    *,
    smoke: bool,
) -> Path:
    """Return the output directory for a study.

    A smoke run always writes to ``<study>/smoke`` and therefore can
    never overwrite a completed full run.
    """

    if study not in STUDIES:
        raise ValueError(f"Unknown study: {study!r}")

    base = RESULTS_ROOT / study

    return base / "smoke" if smoke else base


# The six nominal trigger variants are the default study grid: one
# instance per frozen disruption family, always the nominal offset, so
# a study that varies some other factor does not silently also vary the
# trigger timing.
NOMINAL_INSTANCE_IDS: tuple[str, ...] = tuple(
    instance.instance_id
    for instance in build_scenario_instances()
    if instance.trigger_variant == "nominal"
)


def study_files(study: str, *, smoke: bool) -> dict[str, Path]:
    directory = study_directory(study, smoke=smoke)

    return {
        "directory": directory,
        "raw_runs": directory / "raw_runs.csv",
        "planner_summary": directory / "planner_summary.csv",
        "failure_cases": directory / "failure_cases.csv",
        "run_config": directory / "run_config.json",
        "paired_tests": directory / "analysis_paired_tests.csv",
        "analysis_summary": directory / "analysis_summary.json",
    }


# ================================================================
# Factor definitions
# ================================================================

@dataclass(frozen=True)
class SeverityLevel:
    """One level of one disruption-severity family.

    The numeric values below are the complete specification of the
    factor; they are written verbatim into the study manifest.
    """

    family: str
    level: str
    label: str
    trigger_offset_min: int
    thermal_multiplier: float = 1.0
    global_travel_multiplier: float = 1.0
    closed_poi_ids: tuple[int, ...] = ()
    rationale: str = ""

    @property
    def cell_id(self) -> str:
        return f"{self.family}_{self.level}"

    def to_scenario(self) -> ScenarioInstance:
        definition = ScenarioDefinition(
            scenario_id=self.cell_id,
            name=f"{self.family} / {self.level}",
            nominal_trigger_offset_min=self.trigger_offset_min,
            thermal_multiplier=self.thermal_multiplier,
            global_travel_multiplier=self.global_travel_multiplier,
            closed_poi_ids=frozenset(self.closed_poi_ids),
        )

        return ScenarioInstance(
            scenario=definition,
            trigger_variant="nominal",
            trigger_offset_min=self.trigger_offset_min,
            trigger_absolute_min=(
                PlanningConstraints().start_time_min
                + self.trigger_offset_min
            ),
        )

    def to_dict(self) -> dict:
        return {
            "family": self.family,
            "level": self.level,
            "cell_id": self.cell_id,
            "label": self.label,
            "trigger_offset_min": self.trigger_offset_min,
            "thermal_multiplier": self.thermal_multiplier,
            "global_travel_multiplier": (
                self.global_travel_multiplier
            ),
            "closed_poi_ids": list(self.closed_poi_ids),
            "rationale": self.rationale,
        }


# Trigger offsets are taken from the frozen scenario family the level
# is derived from, so only the severity parameter changes between
# levels of the same family.
SEVERITY_LEVELS: tuple[SeverityLevel, ...] = (
    # ---- travel delay (base: S2 network delay, nominal offset 90) ----
    SeverityLevel(
        family="travel_delay",
        level="light",
        label="+15% walking time",
        trigger_offset_min=90,
        global_travel_multiplier=1.15,
        rationale=(
            "Mild network slowdown; below the frozen S2 value of 1.50."
        ),
    ),
    SeverityLevel(
        family="travel_delay",
        level="medium",
        label="+50% walking time",
        trigger_offset_min=90,
        global_travel_multiplier=1.50,
        rationale="The frozen S2 network-wide delay multiplier.",
    ),
    SeverityLevel(
        family="travel_delay",
        level="severe",
        label="+100% walking time",
        trigger_offset_min=90,
        global_travel_multiplier=2.00,
        rationale=(
            "Walking time doubles; still a plausible pedestrian-network "
            "failure, above anything in the frozen set."
        ),
    ),

    # ---- closure (base: neutral, nominal offset 150) ----
    SeverityLevel(
        family="closure",
        level="light",
        label="one optional POI closed",
        trigger_offset_min=150,
        closed_poi_ids=(33,),
        rationale=(
            "POI 33 is an optional high-utility stop present in every "
            "frozen initial route, so the repair must substitute it."
        ),
    ),
    SeverityLevel(
        family="closure",
        level="medium",
        label="one mandatory POI closed",
        trigger_offset_min=150,
        closed_poi_ids=(13,),
        rationale=(
            "POI 13 is mandatory, so the closure cannot be absorbed by "
            "dropping an optional stop."
        ),
    ),
    SeverityLevel(
        family="closure",
        level="severe",
        label="one mandatory and one optional POI closed",
        trigger_offset_min=150,
        closed_poi_ids=(13, 33),
        rationale=(
            "Removes the mandatory landmark and the highest-utility "
            "optional stop at the same time."
        ),
    ),

    # ---- thermal escalation (base: S5 heatwave, nominal offset 180) ----
    SeverityLevel(
        family="thermal",
        level="light",
        label="thermal multiplier 1.30",
        trigger_offset_min=180,
        thermal_multiplier=1.30,
        rationale="Warm but below both frozen thermal scenarios.",
    ),
    SeverityLevel(
        family="thermal",
        level="medium",
        label="thermal multiplier 1.80",
        trigger_offset_min=180,
        thermal_multiplier=1.80,
        rationale="The frozen S1 heavy-rain thermal multiplier.",
    ),
    SeverityLevel(
        family="thermal",
        level="severe",
        label="thermal multiplier 2.40",
        trigger_offset_min=180,
        thermal_multiplier=2.40,
        rationale=(
            "Above the frozen S5 heatwave value of 2.00; an extreme-heat "
            "stress test, not an observed condition."
        ),
    ),

    # ---- combined disruption escalation ----
    SeverityLevel(
        family="combined",
        level="light",
        label="mild delay + thermal + optional closure",
        trigger_offset_min=150,
        thermal_multiplier=1.30,
        global_travel_multiplier=1.15,
        closed_poi_ids=(33,),
        rationale="Combined low-intensity stressors; each uses the light level above.",
    ),
    SeverityLevel(
        family="combined",
        level="medium",
        label="moderate delay + thermal + mandatory closure",
        trigger_offset_min=150,
        thermal_multiplier=1.80,
        global_travel_multiplier=1.50,
        closed_poi_ids=(13,),
        rationale="Combined medium levels using the frozen S1/S2 conventions.",
    ),
    SeverityLevel(
        family="combined",
        level="severe",
        label="severe delay + thermal + multiple closure",
        trigger_offset_min=150,
        thermal_multiplier=2.40,
        global_travel_multiplier=2.00,
        closed_poi_ids=(13, 33),
        rationale="Combined severe stress test; synthetic exploratory condition.",
    ),
)


def severity_levels(family: str) -> tuple[SeverityLevel, ...]:
    levels = tuple(
        level
        for level in SEVERITY_LEVELS
        if level.family == family
    )

    if not levels:
        raise ValueError(f"Unknown severity family: {family!r}")

    return levels


# ================================================================
# Weather profiles
# ================================================================

@dataclass(frozen=True)
class WeatherLevel:
    """One weather-profile condition.

    ``simulate_missing_segment`` drives the documented three-level
    fallback in ``weather_service.WeatherServiceAdapter``: the live
    lookup fails, no cache exists, and the predefined profile is used.
    A missing segment therefore never becomes zero exposure.
    """

    level: str
    label: str
    category: str
    thermal_multiplier: float
    simulate_missing_segment: bool = False
    rationale: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


WEATHER_LEVELS: tuple[WeatherLevel, ...] = (
    WeatherLevel(
        level="normal",
        label="normal conditions",
        category="normal",
        thermal_multiplier=1.0,
        rationale=(
            "Thermal objective inactive; TripMates Lite must reproduce "
            "Adaptive Without Thermal exactly."
        ),
    ),
    WeatherLevel(
        level="hot",
        label="hot conditions",
        category="hot",
        thermal_multiplier=1.8,
        rationale="The frozen S1 heavy-rain thermal multiplier.",
    ),
    WeatherLevel(
        level="very_hot",
        label="very hot conditions",
        category="very_hot",
        thermal_multiplier=2.4,
        rationale=(
            "Above the frozen S5 heatwave value of 2.00; an extreme-heat "
            "stress test."
        ),
    ),
    WeatherLevel(
        level="missing_segment",
        label="missing weather segment",
        category="hot",
        thermal_multiplier=1.8,
        simulate_missing_segment=True,
        rationale=(
            "The live lookup fails and no cache exists, so the documented "
            "predefined fallback applies. The resolved multiplier equals "
            "the hot profile (1.8) and is recorded per run; missing "
            "weather is never treated as zero exposure."
        ),
    ),
    WeatherLevel(
        level="afternoon_escalation",
        label="afternoon thermal escalation",
        category="hot",
        thermal_multiplier=2.0,
        rationale=(
            "Fixed benchmark profile applied to post-12:00 activity; "
            "it is not a live diurnal forecast."
        ),
    ),
)


def weather_level(level: str) -> WeatherLevel:
    for candidate in WEATHER_LEVELS:
        if candidate.level == level:
            return candidate

    raise ValueError(f"Unknown weather level: {level!r}")


def _failing_fetcher(_url: str, _timeout: float) -> dict:
    raise OSError("missing weather segment for this part of the itinerary")


def resolve_weather_profile(
    level: WeatherLevel,
) -> tuple[WeatherProfile, dict]:
    """Return the resolved profile and its provenance record.

    The provenance block is written into every run row so a reader can
    tell a live profile from a cached one, a predefined one, and a
    predefined fallback triggered by missing data.
    """

    if level.simulate_missing_segment:
        # A throw-away cache path guarantees the adapter cannot fall back
        # to a cached profile: the documented predefined fallback is the
        # only remaining path.
        with tempfile.TemporaryDirectory() as temporary_directory:
            adapter = WeatherServiceAdapter(
                cache_file=(
                    Path(temporary_directory) / "weather_cache.json"
                ),
                fetcher=_failing_fetcher,
            )

            profile = adapter.get_profile(
                latitude=3.1390,
                longitude=101.6869,
                applicable_at="2026-11-04T09:00:00+08:00",
                fallback_profile=predefined_weather_profile(
                    category=level.category,
                    thermal_multiplier=level.thermal_multiplier,
                    applicable_at="2026-11-04T09:00:00+08:00",
                ),
            )
    else:
        profile = predefined_weather_profile(
            category=level.category,
            thermal_multiplier=level.thermal_multiplier,
            applicable_at="2026-11-04T09:00:00+08:00",
        )

    provenance = {
        "weather_level": level.level,
        "weather_source_type": profile.source_type,
        "weather_category": profile.category,
        "weather_thermal_multiplier": profile.thermal_multiplier,
        "weather_fallback_reason": profile.fallback_reason or "",
        "weather_treated_as_zero_exposure": (
            profile.thermal_multiplier <= 0.0
        ),
    }

    return profile, provenance


# ================================================================
# Full-replanning state
# ================================================================

def build_full_replanning_state(
    *,
    data: PlanningData,
    state,
    initial_schedule: ScheduleResult,
) -> "object":
    """Release the locked prefix for the full-replanning ablation.

    The returned state keeps the disruption (closures, travel
    adjustments, trigger record) but commits nothing: the traveller is
    back at the origin at the original start time with the full budget.
    The planner therefore re-optimises the entire day under the
    post-trigger conditions.

    This is an ablation upper bound, not an implementable policy: a
    traveller cannot un-visit a completed attraction.  It exists to
    quantify what the locked prefix costs.
    """

    trip_start_min = float(
        initial_schedule.finish_min
        - initial_schedule.time_used_min
    )

    mandatory_ids = set(data.mandatory_ids)
    unavailable_ids = set(state.unavailable_poi_ids)

    return replace(
        state,
        committed_component="full_replanning",
        current_poi_id=initial_schedule.start_poi_id,
        current_poi_requires_visit=True,
        replan_start_min=trip_start_min,
        replan_start_clock=format_clock(trip_start_min),
        completed_poi_ids=(),
        original_remaining_poi_ids=tuple(
            entry.poi_id for entry in initial_schedule.entries
        ),
        remaining_mandatory_ids=tuple(
            sorted(mandatory_ids - unavailable_ids)
        ),
        unavailable_mandatory_ids=tuple(
            sorted(mandatory_ids & unavailable_ids)
        ),
        cost_spent_rm=0.0,
        utility_achieved=0.0,
        route_finished_before_trigger=False,
    )


# ================================================================
# Itinerary density
# ================================================================

DENSITY_LEVELS = ("relaxed", "moderate", "dense")

DENSITY_MAX_STOPS = 16


def _decode(
    data: PlanningData,
    route: Sequence[int],
    constraints: PlanningConstraints,
) -> ScheduleResult:
    return decode_route(
        data=data,
        visit_sequence=list(route),
        start_poi_id=constraints.start_poi_id,
        start_time_min=constraints.start_time_min,
        trip_end_min=constraints.trip_end_min,
        day_code=constraints.day_code,
        cost_budget_rm=constraints.cost_budget_rm,
        start_poi_is_visit=True,
        required_mandatory_ids=data.mandatory_ids,
    )


def _extend_route(
    data: PlanningData,
    route: Sequence[int],
    constraints: PlanningConstraints,
    *,
    maximum_stops: int = DENSITY_MAX_STOPS,
) -> tuple[list[int], ScheduleResult]:
    """Greedily add the best-value feasible POI until nothing fits.

    Deterministic: candidates are tried in ascending POI-ID order and
    ties are broken by lower POI ID and then by earlier insertion
    position.
    """

    current = list(route)
    current_schedule = _decode(data, current, constraints)

    if not current_schedule.feasible:
        raise RuntimeError(
            "The moderate itinerary must be feasible before extension."
        )

    while len(current) < maximum_stops:
        best: tuple[float, int, int, ScheduleResult] | None = None

        for poi_id in sorted(set(data.pois) - set(current)):
            for position in range(len(current) + 1):
                candidate = (
                    current[:position]
                    + [poi_id]
                    + current[position:]
                )

                schedule = _decode(data, candidate, constraints)

                if not schedule.feasible:
                    continue

                extra_minutes = max(
                    schedule.time_used_min
                    - current_schedule.time_used_min,
                    1e-6,
                )

                score = (
                    data.pois[poi_id].utility_score / extra_minutes
                )

                if best is None:
                    best = (score, poi_id, position, schedule)
                    continue

                best_score, best_poi, best_position, _ = best

                if (
                    score > best_score + 1e-12
                    or (
                        abs(score - best_score) <= 1e-12
                        and (poi_id, position)
                        < (best_poi, best_position)
                    )
                ):
                    best = (score, poi_id, position, schedule)

        if best is None:
            break

        _, poi_id, position, schedule = best

        current = (
            current[:position] + [poi_id] + current[position:]
        )
        current_schedule = schedule

    return current, current_schedule


def build_density_levels(
    *,
    data: PlanningData,
    base_route: Sequence[int],
    constraints: PlanningConstraints,
) -> dict[str, dict]:
    """Build the three initial-itinerary density conditions.

    - ``relaxed``: only the mandatory POIs (the sparsest feasible plan).
    - ``moderate``: the frozen seed-matched Static Utility route.
    - ``dense``: that route greedily extended until no insertion fits.

    The POI catalogue and the walking matrix are untouched; only the
    initial itinerary changes.
    """

    relaxed_route = list(
        build_mandatory_backbone(
            data=data,
            constraints=constraints,
        )
    )
    relaxed_schedule = _decode(data, relaxed_route, constraints)

    if not relaxed_schedule.feasible:
        raise RuntimeError(
            "The mandatory-only itinerary is infeasible on this dataset."
        )

    moderate_route = list(base_route)
    moderate_schedule = _decode(
        data, moderate_route, constraints
    )

    if not moderate_schedule.feasible:
        raise RuntimeError(
            "The frozen initial route is infeasible on this dataset."
        )

    dense_route, dense_schedule = _extend_route(
        data, moderate_route, constraints
    )

    levels = {
        "relaxed": (relaxed_route, relaxed_schedule),
        "moderate": (moderate_route, moderate_schedule),
        "dense": (dense_route, dense_schedule),
    }

    result: dict[str, dict] = {}

    for name, (route, schedule) in levels.items():
        day_length = (
            constraints.trip_end_min - constraints.start_time_min
        )

        result[name] = {
            "density_level": name,
            "route": route,
            "schedule": schedule,
            "stop_count": len(route),
            "visit_min": schedule.total_visit_min,
            "travel_min": schedule.total_travel_min,
            "time_used_min": schedule.time_used_min,
            "time_utilisation": (
                schedule.time_used_min / day_length
                if day_length > 0
                else None
            ),
            "cost_rm": schedule.total_cost_rm,
            "utility": sum(
                data.pois[poi_id].utility_score for poi_id in route
            ),
        }

    return result


# ================================================================
# Planner execution on identical inputs
# ================================================================

@dataclass(frozen=True)
class StudyInputs:
    """The frozen inputs every study needs, loaded exactly once.

    Loading these per study (rather than per run) guarantees that the
    POI catalogue, walking matrix, initial routes, initial schedules and
    scenario instances are literally the same objects for every planner
    and every factor level inside a study.
    """

    data: PlanningData
    constraints: PlanningConstraints
    routes_by_seed: dict[int, list[int]]
    schedules_by_seed: dict[int, ScheduleResult]
    scenario_lookup: dict[str, ScenarioInstance]
    static_distance: tuple[list[int], ScheduleResult, float]

    def route(self, seed: int) -> list[int]:
        return self.routes_by_seed[seed]

    def schedule(self, seed: int) -> ScheduleResult:
        return self.schedules_by_seed[seed]

    def utility(self, seed: int) -> float:
        return route_utility(self.data, self.routes_by_seed[seed])

    def instance(self, instance_id: str) -> ScenarioInstance:
        return self.scenario_lookup[instance_id]


def load_study_inputs() -> StudyInputs:
    data = load_planning_data(
        poi_file=POI_FILE,
        matrix_file=MATRIX_FILE,
    )

    constraints = PlanningConstraints()
    routes_by_seed = load_initial_routes()
    schedules_by_seed = build_initial_schedules(
        data, routes_by_seed
    )

    static_route, static_schedule, static_utility = (
        _static_distance_inputs(data, constraints)
    )

    return StudyInputs(
        data=data,
        constraints=constraints,
        routes_by_seed=routes_by_seed,
        schedules_by_seed=schedules_by_seed,
        scenario_lookup=build_scenario_lookup(),
        static_distance=(
            static_route,
            static_schedule,
            static_utility,
        ),
    )


def tag_records(
    records: Iterable[dict],
    **factors,
) -> list[dict]:
    """Attach the factor levels of the cell to every record it produced.

    The columns are added after the fact so that a study never has to
    re-implement a record builder in order to label a run.
    """

    tagged = list(records)

    for record in tagged:
        record.update(factors)

    return tagged


def failure_record(
    *,
    planner: str,
    scenario_instance: ScenarioInstance,
    seed: int | None,
    initial_route: Sequence[int],
    initial_schedule: ScheduleResult,
    initial_route_utility: float,
    state,
    failure_reason: str,
    runtime_seconds: float,
    deterministic: bool,
) -> dict:
    """Record a failed run as a row, never as a dropped row."""

    return build_metric_record(
        planner=planner,
        scenario_instance=scenario_instance,
        seed=seed,
        initial_route=list(initial_route),
        initial_schedule=initial_schedule,
        initial_route_utility=initial_route_utility,
        state=state,
        repaired_route=None,
        achieved_utility=None,
        completed_poi_count=None,
        total_travel_min=None,
        locked_travel_min=None,
        repaired_schedule=None,
        exposure=None,
        feasible=False,
        failure_reason=failure_reason,
        mandatory_complete=False,
        runtime_seconds=runtime_seconds,
        deterministic=deterministic,
    )


def _route_utility(data: PlanningData, route: Iterable[int]) -> float:
    return sum(
        data.pois[poi_id].utility_score for poi_id in route
    )


def _static_distance_inputs(
    data: PlanningData,
    constraints: PlanningConstraints,
) -> tuple[list[int], ScheduleResult, float]:
    route = build_static_distance_route(
        data=data, start_poi_id=constraints.start_poi_id
    )

    schedule = _decode(data, route, constraints)

    if not schedule.feasible:
        raise RuntimeError(
            "The static-distance route is infeasible on this dataset."
        )

    return route, schedule, _route_utility(data, route)


def run_cell(
    *,
    data: PlanningData,
    constraints: PlanningConstraints,
    initial_route: Sequence[int],
    initial_schedule: ScheduleResult,
    scenario_instance: ScenarioInstance,
    seed: int,
    planners: Sequence[str] = ALL_PLANNERS,
    state_override=None,
    static_distance_inputs=None,
    bounded_repair_config: BoundedRepairConfig | None = None,
) -> list[dict]:
    """Run each planner once on identical inputs and return records.

    ``state_override`` is only meaningful for the two adaptive
    planners: it lets an ablation swap the replanning state (for
    example to release the locked prefix) without touching the
    optimizer.  Static baselines and Greedy Repair derive their own
    state and therefore reject an override.
    """

    records: list[dict] = []

    initial_route = list(initial_route)
    initial_utility = _route_utility(data, initial_route)

    for planner in planners:
        if planner == PLANNER_STATIC_DISTANCE:
            if state_override is not None:
                raise ValueError(
                    "Static Distance does not accept a state override."
                )

            if static_distance_inputs is None:
                static_distance_inputs = _static_distance_inputs(
                    data, constraints
                )

            route, schedule, utility = static_distance_inputs

            records.append(
                static_planner_record(
                    planner=planner,
                    data=data,
                    scenario_instance=scenario_instance,
                    seed=None,
                    initial_route=route,
                    initial_schedule=schedule,
                    initial_route_utility=utility,
                    constraints=constraints,
                    deterministic=True,
                )
            )
            continue

        if planner == PLANNER_STATIC_UTILITY:
            if state_override is not None:
                raise ValueError(
                    "Static Utility does not accept a state override."
                )

            records.append(
                static_planner_record(
                    planner=planner,
                    data=data,
                    scenario_instance=scenario_instance,
                    seed=seed,
                    initial_route=initial_route,
                    initial_schedule=initial_schedule,
                    initial_route_utility=initial_utility,
                    constraints=constraints,
                    deterministic=True,
                )
            )
            continue

        if planner == PLANNER_GREEDY:
            if state_override is not None:
                raise ValueError(
                    "Greedy Repair does not accept a state override."
                )

            records.append(
                greedy_planner_record(
                    data=data,
                    scenario_instance=scenario_instance,
                    seed=seed,
                    initial_route=initial_route,
                    initial_schedule=initial_schedule,
                    initial_route_utility=initial_utility,
                    constraints=constraints,
                )
            )
            continue

        if planner not in (
            PLANNER_ADAPTIVE,
            PLANNER_TRIPMATES,
        ):
            raise ValueError(f"Unknown planner: {planner!r}")

        records.append(
            _adaptive_record(
                planner=planner,
                data=data,
                constraints=constraints,
                initial_route=initial_route,
                initial_schedule=initial_schedule,
                initial_route_utility=initial_utility,
                scenario_instance=scenario_instance,
                seed=seed,
                state_override=state_override,
                bounded_repair_config=bounded_repair_config,
            )
        )

    return records


def _adaptive_record(
    *,
    planner: str,
    data: PlanningData,
    constraints: PlanningConstraints,
    initial_route: list[int],
    initial_schedule: ScheduleResult,
    initial_route_utility: float,
    scenario_instance: ScenarioInstance,
    seed: int,
    state_override,
    bounded_repair_config: BoundedRepairConfig | None = None,
) -> dict:
    started_at = time.perf_counter()

    if state_override is None:
        state = simulate_replanning_state(
            data=data,
            initial_schedule=initial_schedule,
            scenario_instance=scenario_instance,
        )
    else:
        state = state_override

    try:
        problem = prepare_adaptive_problem(
            data=data,
            initial_schedule=initial_schedule,
            state=state,
            scenario_instance=scenario_instance,
            constraints=constraints,
            bounded_repair_config=bounded_repair_config,
        )
    except (RuntimeError, ValueError, KeyError) as error:
        return failure_record(
            planner=planner,
            scenario_instance=scenario_instance,
            seed=seed,
            initial_route=initial_route,
            initial_schedule=initial_schedule,
            initial_route_utility=initial_route_utility,
            state=state,
            failure_reason=f"problem preparation failed: {error}",
            runtime_seconds=time.perf_counter() - started_at,
            deterministic=False,
        )

    try:
        return ga_planner_record(
            planner=planner,
            data=data,
            problem=problem,
            scenario_instance=scenario_instance,
            seed=seed,
            initial_route=initial_route,
            initial_schedule=initial_schedule,
            initial_route_utility=initial_route_utility,
        )
    except (RuntimeError, ValueError, KeyError) as error:
        return failure_record(
            planner=planner,
            scenario_instance=scenario_instance,
            seed=seed,
            initial_route=initial_route,
            initial_schedule=initial_schedule,
            initial_route_utility=initial_route_utility,
            state=state,
            failure_reason=f"planner failed: {error}",
            runtime_seconds=time.perf_counter() - started_at,
            deterministic=False,
        )


def run_adaptive_planner_raw(
    *,
    planner: str,
    data: PlanningData,
    problem,
    seed: int,
):
    """Return the raw planner result object (not a metric record).

    Used by the two ablations that need the final Pareto front rather
    than only the selected route.
    """

    if planner == PLANNER_ADAPTIVE:
        return run_adaptive_without_thermal_ga(
            data=data,
            problem=problem,
            seed=seed,
            config=REPLANNING_GA_CONFIG,
        )

    if planner == PLANNER_TRIPMATES:
        raise ValueError(
            "TripMates Lite needs the initial schedule; use "
            "run_tripmates_planner_raw instead."
        )

    raise ValueError(f"Unknown adaptive planner: {planner!r}")


def run_tripmates_planner_raw(
    *,
    data: PlanningData,
    initial_schedule: ScheduleResult,
    problem,
    seed: int,
):
    return run_tripmates_lite_ga(
        data=data,
        initial_schedule=initial_schedule,
        problem=problem,
        seed=seed,
        config=REPLANNING_GA_CONFIG,
    )


# ================================================================
# Aggregation
# ================================================================

SUMMARY_METRICS = (
    "feasibility_rate",
    "failure_rate",
    "mean_achieved_utility",
    "mean_retained_poi_count",
    "mean_retained_poi_percentage",
    "mean_wes",
    "mean_total_travel_min",
    "mean_changed_activity_count",
    "mean_edit_distance_score",
    "mean_runtime_seconds",
    "p95_runtime_seconds",
)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(dtype=float)

    return pd.to_numeric(frame[column], errors="coerce")


def _mean(series: pd.Series) -> float | None:
    if series.empty:
        return None

    value = series.mean()

    return None if pd.isna(value) else float(value)


def build_planner_summary(
    frame: pd.DataFrame,
    *,
    factor_columns: Sequence[str],
) -> pd.DataFrame:
    """Aggregate raw runs by factor levels and planner.

    ``retained_poi_percentage`` is reported as the mean of
    ``original_poi_retention`` (the share of the original itinerary
    still present), which is defined even when a repair fails.
    """

    if frame.empty:
        return pd.DataFrame()

    group_columns = list(factor_columns) + ["planner"]

    rows: list[dict] = []

    for keys, group in frame.groupby(
        group_columns, sort=True, dropna=False
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {
            column: value
            for column, value in zip(group_columns, keys)
        }

        run_count = int(len(group))
        feasible = group["feasible"].astype(bool)
        feasible_count = int(feasible.sum())

        row["run_count"] = run_count
        row["feasible_count"] = feasible_count
        row["failure_count"] = run_count - feasible_count
        row["feasibility_rate"] = (
            feasible_count / run_count if run_count else None
        )
        row["failure_rate"] = (
            (run_count - feasible_count) / run_count
            if run_count
            else None
        )

        row["mean_achieved_utility"] = _mean(
            _numeric(group, "achieved_utility")
        )
        row["median_achieved_utility"] = (
            _numeric(group, "achieved_utility").median()
            if "achieved_utility" in group
            else None
        )
        row["mean_retained_poi_count"] = _mean(
            _numeric(group, "retained_poi_count")
        )
        row["mean_retained_poi_percentage"] = _mean(
            _numeric(group, "original_poi_retention")
        )
        row["mean_wes"] = _mean(_numeric(group, "wes"))
        row["mean_total_travel_min"] = _mean(
            _numeric(group, "total_travel_min")
        )
        row["mean_changed_activity_count"] = _mean(
            _numeric(group, "changed_activity_count")
        )
        row["mean_edit_distance_score"] = _mean(
            _numeric(group, "edit_distance_score")
        )
        row["mean_runtime_seconds"] = _mean(
            _numeric(group, "runtime_seconds")
        )

        runtime = _numeric(group, "runtime_seconds").dropna()
        row["p95_runtime_seconds"] = (
            float(runtime.quantile(0.95))
            if not runtime.empty
            else None
        )

        rows.append(row)

    return pd.DataFrame(rows)


# ================================================================
# Paired statistics
# ================================================================

def rank_biserial_effect(differences: pd.Series) -> float:
    """Matched-pairs rank-biserial correlation in [-1, 1]."""

    nonzero = differences[differences.abs() > 1e-12]

    if nonzero.empty:
        return 0.0

    ranks = nonzero.abs().rank(method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    total = positive + negative

    if total == 0.0:
        return 0.0

    return (positive - negative) / total


def wilcoxon_pair(differences: pd.Series) -> tuple[float, float, str]:
    """Two-sided Wilcoxon signed-rank test with an explicit status."""

    from scipy.stats import wilcoxon

    values = differences.astype(float)

    if values.empty:
        return 0.0, 1.0, "no_pairs"

    nonzero = int((values.abs() > 1e-12).sum())

    if nonzero == 0:
        return 0.0, 1.0, "all_paired_differences_zero"

    result = wilcoxon(
        values,
        zero_method="wilcox",
        alternative="two-sided",
        method="auto",
    )

    return float(result.statistic), float(result.pvalue), "computed"


def benjamini_hochberg(
    p_values: Sequence[float],
) -> list[float]:
    """Benjamini-Hochberg FDR adjustment, order-preserving.

    The adjustment is monotone: ``q_i`` is never smaller than the
    adjusted value of any larger raw p-value.
    """

    values = [float(value) for value in p_values]
    count = len(values)

    if count == 0:
        return []

    order = sorted(
        range(count), key=lambda index: values[index]
    )

    adjusted = [0.0] * count
    running = 1.0

    for rank_index in range(count - 1, -1, -1):
        index = order[rank_index]
        scaled = (
            values[index] * count / (rank_index + 1)
        )

        running = min(running, scaled)
        adjusted[index] = min(running, 1.0)

    return adjusted


# ================================================================
# Manifest
# ================================================================

def build_manifest(
    *,
    study: str,
    smoke: bool,
    factors: dict,
    seeds: Sequence[int],
    planners: Sequence[str],
    run_count: int,
    failure_count: int,
    total_runtime_seconds: float,
    extra: dict | None = None,
) -> dict:
    """Assemble the machine-readable run configuration.

    Every manifest carries the exact factor levels, the input hashes and
    an explicit statement that the study is exploratory.
    """

    manifest = {
        "study": study,
        "experiment": f"{study} (TripMates Lite)",
        "mode": "smoke_test" if smoke else "full_run",
        "result_class": RESULT_CLASS_EXPLORATORY,
        "confirmatory": False,
        "confirmatory_note": (
            "This study is an exploratory sensitivity analysis. "
            "Confirmatory evidence is the frozen RQ1 benchmark "
            "(results/rq1_full_benchmark) and the frozen weather "
            "ablation (results/weather_ablation). No claim of "
            "superiority may be based on this directory alone."
        ),
        "timestamp_utc": utc_timestamp(),
        "seeds": list(seeds),
        "planners": list(planners),
        "factor_specification": factors,
        "run_count": int(run_count),
        "failure_count": int(failure_count),
        "total_runtime_seconds": round(total_runtime_seconds, 3),
        "candidate_window_limit": 10,
        "ga_configuration": asdict(REPLANNING_GA_CONFIG),
        "stability_weights": StabilityWeightConfig().to_dict(),
        "stability_metric_module": "stability_metrics.py",
        "metric_schema": "benchmark_common.build_metric_record",
        "inputs": {
            "poi_catalogue": file_fingerprint(
                Path("poi_data_calculated_final.csv")
            ),
            "walking_matrix": file_fingerprint(
                Path("walking_time_matrix_final.csv")
            ),
            "initial_routes": file_fingerprint(
                Path(
                    "results/static_utility/"
                    "initial_routes_by_seed.json"
                )
            ),
        },
        "runtime_metadata": runtime_metadata(),
    }

    if extra:
        manifest.update(extra)

    return manifest


def write_study_outputs(
    *,
    study: str,
    smoke: bool,
    frame: pd.DataFrame,
    factor_columns: Sequence[str],
    manifest: dict,
) -> dict[str, Path]:
    """Write the four standard artefacts for a study."""

    files = study_files(study, smoke=smoke)
    files["directory"].mkdir(parents=True, exist_ok=True)

    frame.to_csv(
        files["raw_runs"], index=False, float_format="%.6f"
    )

    summary = build_planner_summary(
        frame, factor_columns=factor_columns
    )
    summary.to_csv(
        files["planner_summary"],
        index=False,
        float_format="%.6f",
    )

    failures = frame[~frame["feasible"].astype(bool)].copy()
    failures.to_csv(
        files["failure_cases"],
        index=False,
        float_format="%.6f",
    )

    write_json(files["run_config"], manifest)

    return files


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def seeds_for_run(smoke: bool) -> tuple[int, ...]:
    return (SEEDS[0],) if smoke else tuple(SEEDS)


def print_study_footer(
    *,
    files: dict[str, Path],
    manifest: dict,
) -> None:
    print()
    print("=" * 100)
    print(manifest["study"])
    print("=" * 100)
    print("Mode:", manifest["mode"])
    print("Runs:", manifest["run_count"])
    print("Failures:", manifest["failure_count"])
    print(
        "Total runtime:",
        f"{manifest['total_runtime_seconds']:.1f} s",
    )
    print()
    print("Raw runs:", files["raw_runs"].resolve())
    print("Planner summary:", files["planner_summary"].resolve())
    print("Failure cases:", files["failure_cases"].resolve())
    print("Run config:", files["run_config"].resolve())
