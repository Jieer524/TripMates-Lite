"""Exact candidate-window fronts for the thermal-selection experiment.

The adaptive planner represents a repair by a permutation of the optional
members of its shared candidate window.  ``decode_adaptive_chromosome`` then
inserts those candidates, in order, around the mandatory backbone.  This
module enumerates that finite decoder domain exactly.  It is deliberately not
a new route decoder: every transition calls the same constrained decoder used
by the GA, so operating hours, budget, availability, mandatory visits, and
the bounded suffix limit remain identical.

The traversal caches equivalent ``(route, remaining-candidates)`` states.
This is dynamic programming over the permutation tree: once two prefix
permutations reach the same route with the same remaining candidates, their
unexplored suffixes are identical and need be visited once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from adaptive_ga_core import (
    AdaptiveProblem,
    decode_replanned_route,
)
from exposure_core import calculate_exposure_metrics
from ga_core import Individual, rank_population, select_final_solution
from planning_core import PlanningData, ScheduleResult
from tripmates_lite_ga import (
    ThermalEvaluatedSolution,
    ThermalSelectionPolicy,
    ThermalSelectionOutcome,
    evaluate_thermal_selection,
    rank_thermal_population,
)


@dataclass(frozen=True)
class ExactFrontResult:
    """The complete decoder-domain population and its thermal front."""

    all_solutions: tuple[ThermalEvaluatedSolution, ...]
    first_front: tuple[ThermalEvaluatedSolution, ...]
    selection: ThermalSelectionOutcome
    state_count: int
    transition_count: int


def select_exact_non_thermal(
    solutions: Iterable[ThermalEvaluatedSolution],
) -> ThermalEvaluatedSolution:
    """Apply the project's non-thermal selector to the exact population."""

    population = [Individual(solution=solution) for solution in solutions]
    if not population:
        raise ValueError("Cannot select from an empty exact population.")
    rank_population(population)
    return select_final_solution(population)


def _insert_candidate(
    *,
    data: PlanningData,
    problem: AdaptiveProblem,
    current_route: tuple[int, ...],
    current_schedule: ScheduleResult,
    candidate_poi_id: int,
) -> tuple[tuple[int, ...], ScheduleResult]:
    """Apply the GA's deterministic least-added-travel insertion rule."""

    feasible: list[tuple[float, float, tuple[int, ...], ScheduleResult]] = []
    for position in range(len(current_route) + 1):
        trial_route = (
            current_route[:position]
            + (candidate_poi_id,)
            + current_route[position:]
        )
        trial_schedule = decode_replanned_route(
            data=data,
            state=problem.state,
            scenario_instance=problem.scenario_instance,
            route=trial_route,
            constraints=problem.constraints,
        )
        if trial_schedule.feasible:
            feasible.append(
                (
                    trial_schedule.total_travel_min
                    - current_schedule.total_travel_min,
                    trial_schedule.finish_min,
                    trial_route,
                    trial_schedule,
                )
            )

    if not feasible:
        return current_route, current_schedule

    return min(feasible, key=lambda item: item[:3])[2:]


def enumerate_exact_front(
    *,
    data: PlanningData,
    initial_schedule: ScheduleResult,
    problem: AdaptiveProblem,
    thermal_multiplier: float,
    non_thermal_reference: ThermalEvaluatedSolution,
    selection_policy: ThermalSelectionPolicy,
) -> ExactFrontResult:
    """Enumerate every bounded repair reachable from the candidate window.

    The returned non-thermal reference is intentionally external: both exact
    selectors are measured against the same selected non-thermal route.  This
    makes the comparison a selector ablation rather than a change in inputs.
    """

    backbone = tuple(problem.mandatory_backbone)
    initial_repaired = decode_replanned_route(
        data=data,
        state=problem.state,
        scenario_instance=problem.scenario_instance,
        route=backbone,
        constraints=problem.constraints,
    )
    if not initial_repaired.feasible:
        raise RuntimeError("The exact mandatory backbone is infeasible.")

    optional_ids = tuple(problem.candidate_window.optional_ids)
    maximum_length = problem.bounded_repair_config.max_modifiable_suffix_activities
    transition_cache: dict[tuple[tuple[int, ...], int], tuple[tuple[int, ...], ScheduleResult]] = {}
    visited: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    terminal_routes: dict[tuple[int, ...], ScheduleResult] = {}

    def transition(route: tuple[int, ...], schedule: ScheduleResult, candidate: int):
        key = (route, candidate)
        cached = transition_cache.get(key)
        if cached is None:
            cached = _insert_candidate(
                data=data,
                problem=problem,
                current_route=route,
                current_schedule=schedule,
                candidate_poi_id=candidate,
            )
            transition_cache[key] = cached
        return cached

    def visit(
        route: tuple[int, ...],
        schedule: ScheduleResult,
        remaining: tuple[int, ...],
    ) -> None:
        key = (route, remaining)
        if key in visited:
            return
        visited.add(key)
        if not remaining or len(route) >= maximum_length:
            terminal_routes[route] = schedule
            return
        for candidate in remaining:
            next_route, next_schedule = transition(route, schedule, candidate)
            next_remaining = tuple(poi for poi in remaining if poi != candidate)
            visit(next_route, next_schedule, next_remaining)

    visit(backbone, initial_repaired, tuple(sorted(optional_ids)))

    solutions: list[ThermalEvaluatedSolution] = []
    for route, schedule in sorted(terminal_routes.items()):
        utility = problem.locked_utility + sum(data.pois[poi].utility_score for poi in route)
        exposure = calculate_exposure_metrics(
            data=data,
            initial_schedule=initial_schedule,
            state=problem.state,
            repaired_schedule=schedule,
            thermal_multiplier=thermal_multiplier,
        )
        solutions.append(
            ThermalEvaluatedSolution(
                chromosome=tuple(optional_ids),
                route=route,
                schedule=schedule,
                achieved_utility=utility,
                completed_poi_count=problem.locked_completed_count + len(route),
                total_travel_min=problem.locked_travel_min + schedule.total_travel_min,
                exposure=exposure,
            )
        )

    population = [Individual(solution=solution) for solution in solutions]
    rank_thermal_population(population)
    selection = evaluate_thermal_selection(
        population, non_thermal_reference, selection_policy)
    return ExactFrontResult(
        all_solutions=tuple(solutions),
        first_front=selection.first_front,
        selection=selection,
        state_count=len(visited),
        transition_count=len(transition_cache),
    )
