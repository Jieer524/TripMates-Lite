"""
Deterministic Greedy Repair baseline for TripMates Lite.

Purpose
-------
Provide a non-GA adaptive repair baseline.  After a disruption it
repairs only the unfinished suffix while obeying exactly the same hard
feasibility constraints as the adaptive planners:

- the same candidate window and candidate-window policy;
- the same POI data, walking matrix and travel adjustments;
- the same opening-hour, budget, time-horizon, duplicate and
  matrix-connectivity rules;
- the same locked completed/committed prefix handling;
- outstanding mandatory POIs preserved whenever feasible.

It deliberately uses none of the following:

- WES or any thermal ranking;
- the weather/thermal multiplier;
- random selection, mutation, crossover, population search or any GA
  state.

Deterministic priority rule (documented)
----------------------------------------
1. preserve feasibility of all outstanding mandatory POIs (a gate);
2. maximise utility gained per added travel-plus-visit minute;
3. lower incremental travel time;
4. higher utility;
5. lower POI ID.

The planner stops gracefully when no further candidate is feasible and
returns a structured failure status if outstanding mandatory POIs
cannot be completed.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import math

from adaptive_ga_core import (
    BoundedRepairConfig,
    RepairDecision,
    RepairRecord,
    build_failure_repair_record,
    build_repair_record,
    decode_replanned_route,
    prepare_adaptive_problem,
)

from ga_core import EvaluatedSolution

from planning_core import (
    PlanningConstraints,
    PlanningData,
)

from scenario_core import (
    ReplanningState,
    ScenarioInstance,
    get_travel_adjustments,
)


GREEDY_REPAIR_NAME = "Greedy Repair"


@dataclass(frozen=True)
class GreedyRepairStep:
    step_index: int
    selected_poi_id: int

    incremental_travel_min: float
    visit_min: float
    utility_score: float
    utility_per_minute: float

    mandatory: bool

    feasible_candidate_ids: tuple[int, ...]


@dataclass(frozen=True)
class GreedyRepairRunResult:
    seed: int | None
    status: str
    failure_reason: str | None

    selected_solution: EvaluatedSolution | None

    candidate_window: tuple[int, ...]
    mandatory_backbone: tuple[int, ...]

    locked_completed_count: int
    locked_utility: float
    locked_travel_min: float

    steps: tuple[GreedyRepairStep, ...]

    repair_record: RepairRecord | None = None


def mandatory_completion_feasible(
    *,
    data: PlanningData,
    state: ReplanningState,
    scenario_instance: ScenarioInstance,
    route: list[int],
    outstanding_mandatory: set[int],
    constraints: PlanningConstraints,
) -> bool:
    """Return True when all outstanding mandatory POIs can still be
    completed after ``route``, in at least one order.

    The remaining mandatory set is small, so every order is tested
    exactly.
    """

    if not outstanding_mandatory:
        return decode_replanned_route(
            data=data,
            state=state,
            scenario_instance=scenario_instance,
            route=tuple(route),
            constraints=constraints,
        ).feasible

    for ordering in permutations(
        sorted(outstanding_mandatory)
    ):
        test_route = list(route) + list(ordering)

        result = decode_replanned_route(
            data=data,
            state=state,
            scenario_instance=scenario_instance,
            route=tuple(test_route),
            constraints=constraints,
        )

        if result.feasible:
            return True

    return False


def run_greedy_repair(
    data: PlanningData,
    *,
    initial_schedule,
    state: ReplanningState,
    scenario_instance: ScenarioInstance,
    constraints: PlanningConstraints | None = None,
    seed: int | None = None,
    bounded_repair_config: BoundedRepairConfig | None = None,
) -> GreedyRepairRunResult:
    """Repair the unfinished suffix with a deterministic greedy rule.

    The bounded-repair limit is enforced via
    ``bounded_repair_config`` (default ``BoundedRepairConfig()``).
    """

    active_bounded_repair = (
        bounded_repair_config or BoundedRepairConfig()
    )

    active_constraints = (
        constraints or PlanningConstraints()
    )

    # ------------------------------------------------------------
    # No repair required when the itinerary already finished.
    # ------------------------------------------------------------

    if state.route_finished_before_trigger:
        no_action_record = RepairRecord(
            locked_prefix=state.completed_poi_ids,
            current_activity=state.committed_component,
            modified_suffix=(),
            retained_activities=(),
            removed_activities=(),
            replacements=(),
            failure_status="no_action",
            failure_reason=None,
            bounded_repair_limit=(
                active_bounded_repair
                .max_modifiable_suffix_activities
            ),
            actual_modification_count=0,
        )
        return GreedyRepairRunResult(
            seed=seed,
            status="no_action",
            failure_reason=None,
            selected_solution=None,
            candidate_window=(),
            mandatory_backbone=(),
            locked_completed_count=len(
                state.completed_poi_ids
            ),
            locked_utility=state.utility_achieved,
            locked_travel_min=0.0,
            steps=(),
            repair_record=no_action_record,
        )

    # ------------------------------------------------------------
    # Shared candidate window / mandatory backbone.
    #
    # prepare_adaptive_problem raises when no feasible post-trigger
    # ordering of the remaining mandatory POIs exists.  That is a
    # genuine infeasibility and is reported structurally.
    # ------------------------------------------------------------

    try:
        problem = prepare_adaptive_problem(
            data=data,
            initial_schedule=initial_schedule,
            state=state,
            scenario_instance=scenario_instance,
            constraints=active_constraints,
            bounded_repair_config=(
                active_bounded_repair
            ),
        )
    except (RuntimeError, ValueError, KeyError) as error:
        failure_record = build_failure_repair_record(
            state=state,
            bounded_repair_limit=(
                active_bounded_repair
                .max_modifiable_suffix_activities
            ),
            failure_reason=str(error),
        )
        return GreedyRepairRunResult(
            seed=seed,
            status="infeasible",
            failure_reason=str(error),
            selected_solution=None,
            candidate_window=(),
            mandatory_backbone=(),
            locked_completed_count=len(
                state.completed_poi_ids
            ),
            locked_utility=state.utility_achieved,
            locked_travel_min=0.0,
            steps=(),
            repair_record=failure_record,
        )

    candidate_window = tuple(
        problem.candidate_window.all_candidate_ids
    )

    adjustments = get_travel_adjustments(
        data,
        scenario_instance.scenario,
    )

    remaining_mandatory = set(
        state.remaining_mandatory_ids
    )

    route: list[int] = []
    steps: list[GreedyRepairStep] = []

    while True:
        current_poi_id = (
            route[-1] if route else state.current_poi_id
        )

        feasible_candidates: list[
            tuple[int, float, float, float]
        ] = []

        for candidate_poi_id in candidate_window:
            if candidate_poi_id in route:
                continue

            test_route = route + [candidate_poi_id]

            outstanding = (
                remaining_mandatory - set(test_route)
            )

            if not mandatory_completion_feasible(
                data=data,
                state=state,
                scenario_instance=scenario_instance,
                route=test_route,
                outstanding_mandatory=outstanding,
                constraints=active_constraints,
            ):
                continue

            if candidate_poi_id == current_poi_id:
                travel_min = 0.0
            else:
                base_travel_min = data.get_travel_time(
                    current_poi_id,
                    candidate_poi_id,
                )

                travel_min = adjustments.apply(
                    current_poi_id,
                    candidate_poi_id,
                    base_travel_min,
                )

            visit_min = (
                data.pois[candidate_poi_id].visit_duration_min
            )

            utility = (
                data.pois[candidate_poi_id].utility_score
            )

            denominator = travel_min + visit_min

            ratio = (
                utility / denominator
                if denominator > 1e-12
                else math.inf
            )

            feasible_candidates.append(
                (
                    candidate_poi_id,
                    ratio,
                    travel_min,
                    utility,
                )
            )

        if not feasible_candidates:
            break

        # Deterministic priority:
        #   -ratio  -> maximise utility per added minute
        #   travel  -> lower incremental travel
        #   -utility -> higher utility
        #   id      -> lower POI ID
        selected = min(
            feasible_candidates,
            key=lambda item: (
                -item[1],
                item[2],
                -item[3],
                item[0],
            ),
        )

        selected_poi_id = selected[0]
        route.append(selected_poi_id)

        steps.append(
            GreedyRepairStep(
                step_index=len(steps),
                selected_poi_id=selected_poi_id,
                incremental_travel_min=selected[2],
                visit_min=(
                    data.pois[
                        selected_poi_id
                    ].visit_duration_min
                ),
                utility_score=selected[3],
                utility_per_minute=selected[1],
                mandatory=(
                    selected_poi_id
                    in remaining_mandatory
                ),
                feasible_candidate_ids=tuple(
                    item[0]
                    for item in feasible_candidates
                ),
            )
        )

    # ------------------------------------------------------------
    # Final decode and mandatory completeness check.
    # ------------------------------------------------------------

    final_result = decode_replanned_route(
        data=data,
        state=state,
        scenario_instance=scenario_instance,
        route=tuple(route),
        constraints=active_constraints,
    )

    if not final_result.feasible:
        infeasible_record = build_failure_repair_record(
            state=state,
            bounded_repair_limit=(
                problem
                .bounded_repair_config
                .max_modifiable_suffix_activities
            ),
            failure_reason=(
                final_result.failure_reason
                or "The greedy repaired route is infeasible."
            ),
        )
        return GreedyRepairRunResult(
            seed=seed,
            status="infeasible",
            failure_reason=(
                final_result.failure_reason
                or "The greedy repaired route is infeasible."
            ),
            selected_solution=None,
            candidate_window=candidate_window,
            mandatory_backbone=(
                problem.mandatory_backbone
            ),
            locked_completed_count=(
                problem.locked_completed_count
            ),
            locked_utility=problem.locked_utility,
            locked_travel_min=(
                problem.locked_travel_min
            ),
            steps=tuple(steps),
            repair_record=infeasible_record,
        )

    missing_mandatory = (
        remaining_mandatory - set(route)
    )

    if missing_mandatory:
        missing_reason = (
            "Outstanding mandatory POIs could not be "
            "completed: "
            f"{sorted(missing_mandatory)}"
        )
        missing_record = build_failure_repair_record(
            state=state,
            bounded_repair_limit=(
                problem
                .bounded_repair_config
                .max_modifiable_suffix_activities
            ),
            failure_reason=missing_reason,
        )
        return GreedyRepairRunResult(
            seed=seed,
            status="infeasible",
            failure_reason=missing_reason,
            selected_solution=None,
            candidate_window=candidate_window,
            mandatory_backbone=(
                problem.mandatory_backbone
            ),
            locked_completed_count=(
                problem.locked_completed_count
            ),
            locked_utility=problem.locked_utility,
            locked_travel_min=(
                problem.locked_travel_min
            ),
            steps=tuple(steps),
            repair_record=missing_record,
        )

    repaired_utility = sum(
        data.pois[poi_id].utility_score
        for poi_id in route
    )

    solution = EvaluatedSolution(
        chromosome=tuple(route),
        route=tuple(route),
        schedule=final_result,
        achieved_utility=(
            problem.locked_utility + repaired_utility
        ),
        completed_poi_count=(
            problem.locked_completed_count + len(route)
        ),
        total_travel_min=(
            problem.locked_travel_min
            + final_result.total_travel_min
        ),
    )

    success_record = build_repair_record(
        state=state,
        problem=problem,
        repaired_route=tuple(route),
    )

    return GreedyRepairRunResult(
        seed=seed,
        status="success",
        failure_reason=None,
        selected_solution=solution,
        candidate_window=candidate_window,
        mandatory_backbone=problem.mandatory_backbone,
        locked_completed_count=(
            problem.locked_completed_count
        ),
        locked_utility=problem.locked_utility,
        locked_travel_min=problem.locked_travel_min,
        steps=tuple(steps),
        repair_record=success_record,
    )
