"""Application-service boundary for validated initial itinerary planning."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Literal

from adaptive_ga_core import (
    BoundedRepairConfig,
    REPLANNING_GA_CONFIG,
    AdaptiveGARunResult,
    prepare_adaptive_problem,
    run_adaptive_without_thermal_ga,
)
from exposure_core import ExposureMetrics, calculate_exposure_metrics
from ga_core import GAConfig, GARunResult, run_static_utility_ga
from greedy_repair import GreedyRepairRunResult, run_greedy_repair
from planning_core import (
    PlanningConstraints,
    PlanningData,
    ScheduleResult,
    load_planning_data,
)
from scenario_core import (
    ReplanningState,
    ScenarioInstance,
    build_scenario_instances,
    simulate_replanning_state,
)
from stability_metrics import (
    StabilityMetrics,
    StabilityWeightConfig,
    compute_stability_from_state,
)
from tripmates_lite_ga import TripMatesGARunResult, run_tripmates_lite_ga


DAY_CODES = (
    "MON",
    "TUE",
    "WED",
    "THU",
    "FRI",
    "SAT",
    "SUN",
)

PREFERRED_UTILITY_MULTIPLIER = 1.25


@dataclass(frozen=True)
class TripRequest:
    trip_date: date
    start_poi_id: int
    start_time_min: float
    end_time_min: float
    cost_budget_rm: float
    mandatory_poi_ids: tuple[int, ...] = ()
    preferred_poi_ids: tuple[int, ...] = ()
    seed: int = 1001


@dataclass(frozen=True)
class ValidationIssue:
    field: str
    message: str


class RequestValidationError(ValueError):
    def __init__(self, issues: list[ValidationIssue]) -> None:
        self.issues = tuple(issues)
        super().__init__("; ".join(issue.message for issue in issues))


@dataclass(frozen=True)
class PlanningInstance:
    request: TripRequest
    data: PlanningData
    constraints: PlanningConstraints
    effective_mandatory_poi_ids: tuple[int, ...]


@dataclass(frozen=True)
class InitialPlanOutcome:
    status: Literal["success", "infeasible"]
    message: str
    instance: PlanningInstance
    ga_result: GARunResult | None = None

    @property
    def schedule(self) -> ScheduleResult | None:
        if self.ga_result is None:
            return None
        return self.ga_result.selected_solution.schedule


@dataclass(frozen=True)
class ReplanningOutcome:
    planner_id: Literal[
        "adaptive_without_thermal",
        "tripmates_lite",
        "greedy_repair",
    ]
    status: Literal["success", "no_action", "infeasible"]
    message: str
    state: ReplanningState
    repaired_schedule: ScheduleResult | None
    exposure: ExposureMetrics | None
    explanation: str
    replanning_latency_ms: float
    ga_result: (
        AdaptiveGARunResult
        | TripMatesGARunResult
        | GreedyRepairRunResult
        | None
    ) = None
    stability: StabilityMetrics | None = None


def build_replanning_explanation(
    *,
    data: PlanningData,
    state: ReplanningState,
    repaired_schedule: ScheduleResult,
    planner_id: str,
    thermal_objective_active: bool,
) -> str:
    """Produce the deterministic explanation required by the thesis."""

    component_text = {
        "walking_leg": "an in-progress walking leg",
        "visit": "an in-progress visit",
        "waiting": "a waiting interval",
        "none": "the next unfinished activity",
    }.get(state.committed_component, "the unfinished itinerary")

    original_remaining = set(state.original_remaining_poi_ids)
    repaired_ids = set(repaired_schedule.completed_sequence)
    removed_ids = sorted(original_remaining - repaired_ids)
    added_ids = sorted(repaired_ids - original_remaining)

    removed_text = (
        ", ".join(data.pois[poi_id].name for poi_id in removed_ids)
        if removed_ids
        else "no scheduled POI"
    )
    added_text = (
        ", ".join(data.pois[poi_id].name for poi_id in added_ids)
        if added_ids
        else "no substitute POI"
    )

    if thermal_objective_active:
        benefit = (
            "The selected route prioritised lower weather-weighted "
            "outdoor exposure while retaining the required route quality."
        )
    elif state.unavailable_poi_ids:
        benefit = (
            "The selected route preserves feasible mandatory visits "
            "after removing unavailable POIs."
        )
    else:
        benefit = (
            "The selected route preserves feasible mandatory visits "
            "within the remaining time and admission budget."
        )

    return (
        f"{state.scenario_name} occurred at {state.trigger_clock} during "
        f"{component_text}. The replanner removed or skipped {removed_text} "
        f"and introduced {added_text}. {benefit}"
    )


class ItineraryService:
    """Coordinates request validation, initial planning and results."""

    def __init__(self, data: PlanningData) -> None:
        self.data = data

    @classmethod
    def from_files(
        cls,
        *,
        poi_file: str | Path,
        matrix_file: str | Path,
    ) -> "ItineraryService":
        return cls(
            load_planning_data(
                poi_file=poi_file,
                matrix_file=matrix_file,
            )
        )

    def build_instance(self, request: TripRequest) -> PlanningInstance:
        issues = self.validate_request(request)

        if issues:
            raise RequestValidationError(issues)

        day_code = DAY_CODES[request.trip_date.weekday()]
        effective_mandatory_ids = tuple(
            sorted(
                set(request.mandatory_poi_ids)
                | {request.start_poi_id}
            )
        )

        constraints = PlanningConstraints(
            start_poi_id=request.start_poi_id,
            start_time_min=request.start_time_min,
            trip_end_min=request.end_time_min,
            day_code=day_code,
            cost_budget_rm=request.cost_budget_rm,
        )

        request_data = self.data.with_mandatory_pois(
            effective_mandatory_ids
        ).with_preferred_pois(
            request.preferred_poi_ids,
            utility_multiplier=PREFERRED_UTILITY_MULTIPLIER,
        )

        return PlanningInstance(
            request=request,
            data=request_data,
            constraints=constraints,
            effective_mandatory_poi_ids=effective_mandatory_ids,
        )

    def plan_initial_itinerary(
        self,
        instance: PlanningInstance,
        *,
        config: GAConfig | None = None,
    ) -> InitialPlanOutcome:
        try:
            result = run_static_utility_ga(
                data=instance.data,
                seed=instance.request.seed,
                config=config,
                constraints=instance.constraints,
            )
        except (RuntimeError, ValueError, KeyError) as error:
            return InitialPlanOutcome(
                status="infeasible",
                message=str(error),
                instance=instance,
            )

        return InitialPlanOutcome(
            status="success",
            message="Initial itinerary generated.",
            instance=instance,
            ga_result=result,
        )

    def scenario_instances_for(
        self,
        instance: PlanningInstance,
    ) -> tuple[ScenarioInstance, ...]:
        return build_scenario_instances(
            trip_start_min=int(instance.constraints.start_time_min)
        )

    def replan(
        self,
        *,
        instance: PlanningInstance,
        initial_outcome: InitialPlanOutcome,
        scenario_instance: ScenarioInstance,
        planner_id: Literal[
            "adaptive_without_thermal",
            "tripmates_lite",
            "greedy_repair",
        ],
        config: GAConfig | None = None,
        bounded_repair_config: BoundedRepairConfig | None = None,
        stability_weights: StabilityWeightConfig | None = None,
    ) -> ReplanningOutcome:
        initial_schedule = initial_outcome.schedule

        if initial_schedule is None or not initial_schedule.feasible:
            raise ValueError(
                "Replanning requires a feasible initial itinerary."
            )

        active_bounded_repair = (
            bounded_repair_config or BoundedRepairConfig()
        )

        state = simulate_replanning_state(
            data=instance.data,
            initial_schedule=initial_schedule,
            scenario_instance=scenario_instance,
        )

        def stability_for(
            *,
            repaired_route=(),
            repaired_schedule=None,
            repair_record=None,
            feasible: bool = True,
        ) -> StabilityMetrics:
            """Compare the original and replanned suffixes from this state.

            The locked prefix is verified, never corrected: a violation
            is reported through ``hard_stability_violation``.
            """

            return compute_stability_from_state(
                state=state,
                replanned_suffix=repaired_route,
                original_schedule=initial_schedule,
                replanned_schedule=repaired_schedule,
                repair_record=repair_record,
                weights=stability_weights,
                replanning_feasible=feasible,
            )

        if state.route_finished_before_trigger:
            return ReplanningOutcome(
                planner_id=planner_id,
                status="no_action",
                message="The itinerary finished before the disruption.",
                state=state,
                repaired_schedule=None,
                exposure=None,
                explanation=(
                    "No replanning was required because the itinerary "
                    "was already complete."
                ),
                replanning_latency_ms=0.0,
                stability=stability_for(),
            )

        active_config = config or REPLANNING_GA_CONFIG
        started_at = perf_counter()

        try:
            if planner_id == "greedy_repair":
                ga_result = run_greedy_repair(
                    data=instance.data,
                    initial_schedule=initial_schedule,
                    state=state,
                    scenario_instance=scenario_instance,
                    constraints=instance.constraints,
                    seed=instance.request.seed,
                    bounded_repair_config=(
                        active_bounded_repair
                    ),
                )

                if (
                    ga_result.status != "success"
                    or ga_result.selected_solution is None
                ):
                    return ReplanningOutcome(
                        planner_id=planner_id,
                        status="infeasible",
                        message=(
                            ga_result.failure_reason
                            or "Greedy Repair could not complete "
                            "the mandatory POIs."
                        ),
                        state=state,
                        repaired_schedule=None,
                        exposure=None,
                        explanation=(
                            "No feasible repaired itinerary could be "
                            "generated under the active constraints."
                        ),
                        replanning_latency_ms=(
                            perf_counter() - started_at
                        )
                        * 1000,
                        ga_result=ga_result,
                        stability=stability_for(
                            repair_record=(
                                ga_result.repair_record
                            ),
                            feasible=False,
                        ),
                    )

                repaired_schedule = (
                    ga_result.selected_solution.schedule
                )
                thermal_active = False
                exposure = calculate_exposure_metrics(
                    data=instance.data,
                    initial_schedule=initial_schedule,
                    state=state,
                    repaired_schedule=repaired_schedule,
                    thermal_multiplier=(
                        scenario_instance.scenario.thermal_multiplier
                    ),
                )
            else:
                problem = prepare_adaptive_problem(
                    data=instance.data,
                    initial_schedule=initial_schedule,
                    state=state,
                    scenario_instance=scenario_instance,
                    constraints=instance.constraints,
                    bounded_repair_config=(
                        active_bounded_repair
                    ),
                )

                if planner_id == "adaptive_without_thermal":
                    ga_result = run_adaptive_without_thermal_ga(
                        data=instance.data,
                        problem=problem,
                        seed=instance.request.seed,
                        config=active_config,
                    )
                    repaired_schedule = (
                        ga_result.selected_solution.schedule
                    )
                    thermal_active = False
                    exposure = calculate_exposure_metrics(
                        data=instance.data,
                        initial_schedule=initial_schedule,
                        state=state,
                        repaired_schedule=repaired_schedule,
                        thermal_multiplier=(
                            scenario_instance.scenario.thermal_multiplier
                        ),
                    )
                else:
                    ga_result = run_tripmates_lite_ga(
                        data=instance.data,
                        initial_schedule=initial_schedule,
                        problem=problem,
                        seed=instance.request.seed,
                        config=active_config,
                    )
                    repaired_schedule = (
                        ga_result.selected_solution.schedule
                    )
                    thermal_active = (
                        ga_result.thermal_objective_active
                    )
                    exposure = ga_result.selected_solution.exposure
        except (RuntimeError, ValueError, KeyError) as error:
            return ReplanningOutcome(
                planner_id=planner_id,
                status="infeasible",
                message=str(error),
                state=state,
                repaired_schedule=None,
                exposure=None,
                explanation=(
                    "No feasible repaired itinerary could be generated "
                    "under the active constraints."
                ),
                replanning_latency_ms=(perf_counter() - started_at) * 1000,
                stability=stability_for(feasible=False),
            )

        return ReplanningOutcome(
            planner_id=planner_id,
            status="success",
            message="Itinerary replanned.",
            state=state,
            repaired_schedule=repaired_schedule,
            exposure=exposure,
            explanation=build_replanning_explanation(
                data=instance.data,
                state=state,
                repaired_schedule=repaired_schedule,
                planner_id=planner_id,
                thermal_objective_active=thermal_active,
            ),
            replanning_latency_ms=(perf_counter() - started_at) * 1000,
            ga_result=ga_result,
            stability=stability_for(
                repaired_route=repaired_schedule.completed_sequence,
                repaired_schedule=repaired_schedule,
                repair_record=ga_result.repair_record,
            ),
        )

    def validate_request(
        self,
        request: TripRequest,
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        if not isinstance(request.trip_date, date):
            issues.append(
                ValidationIssue(
                    "trip_date",
                    "Trip date must be a calendar date.",
                )
            )
            return issues

        if request.start_poi_id not in self.data.pois:
            issues.append(
                ValidationIssue(
                    "start_poi_id",
                    f"Unknown starting POI: {request.start_poi_id}.",
                )
            )

        if request.start_time_min < 0:
            issues.append(
                ValidationIssue(
                    "start_time_min",
                    "Trip start time cannot be negative.",
                )
            )

        if request.end_time_min <= request.start_time_min:
            issues.append(
                ValidationIssue(
                    "end_time_min",
                    "Trip end time must be after the start time.",
                )
            )

        if request.cost_budget_rm < 0:
            issues.append(
                ValidationIssue(
                    "cost_budget_rm",
                    "Compulsory-admission budget cannot be negative.",
                )
            )

        selected_ids = (
            set(request.mandatory_poi_ids)
            | set(request.preferred_poi_ids)
        )
        unknown_ids = selected_ids - set(self.data.pois)

        if unknown_ids:
            issues.append(
                ValidationIssue(
                    "selected_pois",
                    "Unknown selected POI IDs: "
                    f"{sorted(unknown_ids)}.",
                )
            )

        if len(request.mandatory_poi_ids) != len(
            set(request.mandatory_poi_ids)
        ):
            issues.append(
                ValidationIssue(
                    "mandatory_poi_ids",
                    "Mandatory POIs must not contain duplicates.",
                )
            )

        if len(request.preferred_poi_ids) != len(
            set(request.preferred_poi_ids)
        ):
            issues.append(
                ValidationIssue(
                    "preferred_poi_ids",
                    "Preferred POIs must not contain duplicates.",
                )
            )

        if issues:
            return issues

        day_code = DAY_CODES[request.trip_date.weekday()]
        required_ids = set(request.mandatory_poi_ids) | {
            request.start_poi_id
        }
        unavailable = [
            poi_id
            for poi_id in sorted(required_ids)
            if self.data.pois[poi_id].get_opening_window(day_code) is None
        ]

        if unavailable:
            issues.append(
                ValidationIssue(
                    "mandatory_poi_ids",
                    "Selected mandatory POIs are unavailable on "
                    f"{day_code}: {unavailable}.",
                )
            )

        return issues
