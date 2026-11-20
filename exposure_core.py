"""Transparent thermal-exposure proxy calculations.

WES is a planning proxy, not a medical, physiological, or health-risk
model. For POI ``j``, ``indoor_ratio_j`` is the proportion of the visit
expected to occur indoors. ``shelter_ratio_j`` is the conditional
proportion of the remaining outdoor visit fraction that is sheltered or
shaded; it is not a second whole-visit enclosure proportion. The exposed
fraction is therefore ``x_j = (1 - indoor_ratio_j) * (1 - shelter_ratio_j)``.
Walking exposure
is walking minutes; visit exposure is ``visit_minutes * x_j``. Waiting is
excluded by default and is recorded separately; callers may opt in to
``waiting_minutes * waiting_exposed_fraction`` explicitly. For an
interval ``[a,b]`` and trigger ``t``, weighted exposure is
``overlap([a,b], before t) + M * overlap([a,b], after t)``. Thus
``WES = pre_trigger_exposure + M * post_trigger_exposure``.
"""

from __future__ import annotations

from dataclasses import dataclass

from planning_core import (
    PlanningData,
    ScheduleResult,
)

from scenario_core import ReplanningState


@dataclass(frozen=True)
class PrefixExposure:
    outdoor_walking_min: float
    outdoor_visit_min: float
    weather_weighted_exposure_score: float


@dataclass(frozen=True)
class ExposureMetrics:
    prefix_walking_min: float
    prefix_visit_min: float

    repaired_walking_min: float
    repaired_visit_min: float

    outdoor_walking_min: float
    outdoor_visit_min: float
    total_outdoor_exposure_min: float

    weather_weighted_exposure_score: float
    thermal_multiplier: float
    waiting_exposure_min: float = 0.0
    high_heat_outdoor_exposure_min: float = 0.0
    maximum_continuous_outdoor_exposure_min: float = 0.0
    exposure_during_travel_min: float = 0.0
    exposure_during_visits_min: float = 0.0
    exposure_during_waiting_min: float = 0.0
    weather_source: str = "scenario"
    weather_fallback_reason: str | None = None


def calculate_exposed_fraction(
    indoor_ratio: float,
    shelter_ratio: float,
) -> float:
    """
    Calculate the sequential exposed-fraction proxy.

    ``indoor_ratio`` is a whole-visit indoor proportion. ``shelter_ratio``
    is conditional on the remaining outdoor fraction: it represents the
    share of that outdoor time covered or shaded by outdoor shelter. It
    must not be interpreted as a second whole-visit enclosure proportion.

    X_j = (1 - I_j)(1 - S_j)
    """

    if not 0.0 <= indoor_ratio <= 1.0:
        raise ValueError(
            "Indoor ratio must be between 0 and 1."
        )

    if not 0.0 <= shelter_ratio <= 1.0:
        raise ValueError(
            "Shelter ratio must be between 0 and 1."
        )

    return (
        (1.0 - indoor_ratio)
        * (1.0 - shelter_ratio)
    )


def interval_overlap(
    interval_start: float,
    interval_end: float,
    window_start: float,
    window_end: float,
) -> float:
    """Return the overlap between two time intervals."""

    overlap_start = max(
        interval_start,
        window_start,
    )

    overlap_end = min(
        interval_end,
        window_end,
    )

    return max(
        0.0,
        overlap_end - overlap_start,
    )


def calculate_weighted_interval(
    interval_start: float,
    interval_end: float,
    execution_start: float,
    execution_end: float,
    trigger_min: float,
    thermal_multiplier: float,
) -> float:
    """
    Apply multiplier 1 before the trigger and M_s afterwards.

    This prevents a disruption from affecting exposure that
    occurred before its trigger.
    """

    if thermal_multiplier < 1.0:
        raise ValueError(
            "Thermal multiplier cannot be below 1."
        )

    clipped_start = max(
        interval_start,
        execution_start,
    )

    clipped_end = min(
        interval_end,
        execution_end,
    )

    if clipped_end <= clipped_start:
        return 0.0

    before_trigger = interval_overlap(
        interval_start=clipped_start,
        interval_end=clipped_end,
        window_start=execution_start,
        window_end=min(
            trigger_min,
            execution_end,
        ),
    )

    after_trigger = interval_overlap(
        interval_start=clipped_start,
        interval_end=clipped_end,
        window_start=max(
            trigger_min,
            execution_start,
        ),
        window_end=execution_end,
    )

    return (
        before_trigger
        + thermal_multiplier
        * after_trigger
    )


def calculate_prefix_exposure(
    data: PlanningData,
    initial_schedule: ScheduleResult,
    state: ReplanningState,
    thermal_multiplier: float,
) -> PrefixExposure:
    """
    Calculate exposure from trip start until replanning starts.

    This includes:

    - completed walking legs;
    - a committed walking leg;
    - completed visits;
    - a committed visit completed before replanning.
    """

    if not initial_schedule.feasible:
        raise ValueError(
            "Initial schedule must be feasible."
        )

    execution_start = (
        initial_schedule.finish_min
        - initial_schedule.time_used_min
    )

    execution_end = (
        state.replan_start_min
    )

    trigger_min = (
        state.trigger_absolute_min
    )

    outdoor_walking = 0.0
    outdoor_visit = 0.0
    weighted_score = 0.0

    previous_visit_end = (
        execution_start
    )

    for entry in initial_schedule.entries:
        # --------------------------------------------------------
        # Walking segment
        # --------------------------------------------------------

        travel_start = (
            previous_visit_end
        )

        travel_end = (
            entry.arrival_min
        )

        executed_walking = interval_overlap(
            interval_start=travel_start,
            interval_end=travel_end,
            window_start=execution_start,
            window_end=execution_end,
        )

        weighted_walking = (
            calculate_weighted_interval(
                interval_start=travel_start,
                interval_end=travel_end,
                execution_start=(
                    execution_start
                ),
                execution_end=(
                    execution_end
                ),
                trigger_min=trigger_min,
                thermal_multiplier=(
                    thermal_multiplier
                ),
            )
        )

        outdoor_walking += (
            executed_walking
        )

        weighted_score += (
            weighted_walking
        )

        # --------------------------------------------------------
        # POI visit segment
        # --------------------------------------------------------

        poi = data.pois[
            entry.poi_id
        ]

        exposed_fraction = (
            calculate_exposed_fraction(
                indoor_ratio=(
                    poi.indoor_ratio
                ),
                shelter_ratio=(
                    poi.shelter_ratio
                ),
            )
        )

        executed_visit_duration = (
            interval_overlap(
                interval_start=(
                    entry.visit_start_min
                ),
                interval_end=(
                    entry.visit_end_min
                ),
                window_start=(
                    execution_start
                ),
                window_end=(
                    execution_end
                ),
            )
        )

        weighted_visit_duration = (
            calculate_weighted_interval(
                interval_start=(
                    entry.visit_start_min
                ),
                interval_end=(
                    entry.visit_end_min
                ),
                execution_start=(
                    execution_start
                ),
                execution_end=(
                    execution_end
                ),
                trigger_min=(
                    trigger_min
                ),
                thermal_multiplier=(
                    thermal_multiplier
                ),
            )
        )

        outdoor_visit += (
            executed_visit_duration
            * exposed_fraction
        )

        weighted_score += (
            weighted_visit_duration
            * exposed_fraction
        )

        previous_visit_end = (
            entry.visit_end_min
        )

    return PrefixExposure(
        outdoor_walking_min=(
            outdoor_walking
        ),

        outdoor_visit_min=(
            outdoor_visit
        ),

        weather_weighted_exposure_score=(
            weighted_score
        ),
    )


def calculate_repaired_exposure(
    data: PlanningData,
    repaired_schedule: ScheduleResult,
) -> tuple[float, float]:
    """
    Calculate raw walking and visit exposure for the repaired route.

    The scenario multiplier is applied later because every repaired
    activity occurs after the trigger.
    """

    if not repaired_schedule.feasible:
        raise ValueError(
            "Repaired schedule must be feasible."
        )

    outdoor_walking = (
        repaired_schedule.total_travel_min
    )

    outdoor_visit = 0.0

    for entry in repaired_schedule.entries:
        poi = data.pois[
            entry.poi_id
        ]

        exposed_fraction = (
            calculate_exposed_fraction(
                indoor_ratio=(
                    poi.indoor_ratio
                ),
                shelter_ratio=(
                    poi.shelter_ratio
                ),
            )
        )

        outdoor_visit += (
            entry.visit_duration_min
            * exposed_fraction
        )

    return (
        outdoor_walking,
        outdoor_visit,
    )


def _maximum_continuous_exposure(
    data: PlanningData,
    schedule: ScheduleResult,
    *,
    include_waiting: bool = False,
) -> float:
    """Return the longest contiguous exposed activity run in minutes.

    This is a schedule-continuity proxy. Gaps, indoor visits, and excluded
    waiting break the run; it does not infer a physiological threshold.
    """
    best = current = 0.0
    previous_end = None
    for entry in schedule.entries:
        travel = max(0.0, entry.travel_min)
        waiting = entry.waiting_min if include_waiting else 0.0
        poi = data.pois[entry.poi_id]
        visit = entry.visit_duration_min * calculate_exposed_fraction(
            poi.indoor_ratio, poi.shelter_ratio
        )
        segments = (travel, waiting, visit)
        for duration in segments:
            if duration > 0:
                current += duration
                best = max(best, current)
            else:
                current = 0.0
        previous_end = entry.visit_end_min
    return best


def calculate_exposure_metrics(
    data: PlanningData,
    initial_schedule: ScheduleResult,
    state: ReplanningState,
    repaired_schedule: ScheduleResult,
    thermal_multiplier: float,
    *,
    include_waiting: bool = False,
    waiting_exposed_fraction: float = 0.0,
    weather_source: str = "scenario",
    weather_fallback_reason: str | None = None,
) -> ExposureMetrics:
    """
    Calculate:

    OWM  = outdoor walking minutes
    OVM  = outdoor POI visit minutes
    TOEM = OWM + OVM
    WES  = weather-weighted exposure score
    """

    prefix = calculate_prefix_exposure(
        data=data,
        initial_schedule=(
            initial_schedule
        ),
        state=state,
        thermal_multiplier=(
            thermal_multiplier
        ),
    )

    (
        repaired_walking,
        repaired_visit,
    ) = calculate_repaired_exposure(
        data=data,
        repaired_schedule=(
            repaired_schedule
        ),
    )

    outdoor_walking = (
        prefix.outdoor_walking_min
        + repaired_walking
    )

    outdoor_visit = (
        prefix.outdoor_visit_min
        + repaired_visit
    )

    total_outdoor_exposure = (
        outdoor_walking
        + outdoor_visit
    )

    # Every repaired component begins after the trigger.
    repaired_weighted_exposure = (
        thermal_multiplier
        * (
            repaired_walking
            + repaired_visit
        )
    )

    weather_weighted_score = (
        prefix
        .weather_weighted_exposure_score
        + repaired_weighted_exposure
    )

    waiting_exposure = 0.0
    if include_waiting:
        if not 0.0 <= waiting_exposed_fraction <= 1.0:
            raise ValueError("waiting_exposed_fraction must be between 0 and 1")
        waiting_exposure = (
            initial_schedule.total_waiting_min
            + repaired_schedule.total_waiting_min
        ) * waiting_exposed_fraction

    return ExposureMetrics(
        prefix_walking_min=(
            prefix.outdoor_walking_min
        ),

        prefix_visit_min=(
            prefix.outdoor_visit_min
        ),

        repaired_walking_min=(
            repaired_walking
        ),

        repaired_visit_min=(
            repaired_visit
        ),

        outdoor_walking_min=(
            outdoor_walking
        ),

        outdoor_visit_min=(
            outdoor_visit
        ),

        total_outdoor_exposure_min=(
            total_outdoor_exposure
        ),

        weather_weighted_exposure_score=(
            weather_weighted_score
        ),

        thermal_multiplier=(
            thermal_multiplier
        ),
        waiting_exposure_min=waiting_exposure,
        high_heat_outdoor_exposure_min=(
            total_outdoor_exposure if thermal_multiplier >= 1.8 else 0.0
        ),
        maximum_continuous_outdoor_exposure_min=(
            _maximum_continuous_exposure(data, repaired_schedule, include_waiting=include_waiting)
        ),
        exposure_during_travel_min=outdoor_walking,
        exposure_during_visits_min=outdoor_visit,
        exposure_during_waiting_min=waiting_exposure,
        weather_source=weather_source,
        weather_fallback_reason=weather_fallback_reason,
    )
