"""Single-route ILS adaptation of Hapsari et al. (2019), DOI 10.1007/s40092-019-0315-9.

This is an adapted comparator, not a reproduction of their multi-route AILS.
It uses insertion, pairwise exchange/reversal, and shaking, while the shared
TripMates decoder enforces the post-trigger single-route repair contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import random

from adaptive_ga_core import AdaptiveProblem, build_repair_record, decode_replanned_route
from planning_core import PlanningData, ScheduleResult


@dataclass(frozen=True)
class ILSConfig:
    iterations: int = 20
    shake_count: int = 2


@dataclass(frozen=True)
class ILSResult:
    route: tuple[int, ...]
    schedule: ScheduleResult
    evaluations: int
    repair_record: object


def run_ils_repair(
    *, data: PlanningData, problem: AdaptiveProblem, seed: int,
    config: ILSConfig = ILSConfig(),
) -> ILSResult:
    """Search one bounded repaired suffix with a seeded ILS adaptation."""
    rng = random.Random(seed)
    cache: dict[tuple[int, ...], ScheduleResult | None] = {}
    allowed = set(problem.candidate_window.optional_ids)
    mandatory = set(problem.mandatory_backbone)
    maximum = problem.bounded_repair_config.max_modifiable_suffix_activities

    def evaluate(route: tuple[int, ...]) -> ScheduleResult | None:
        if route not in cache:
            if len(route) > maximum or len(route) != len(set(route)) or not mandatory <= set(route):
                cache[route] = None
            elif not set(route) <= allowed | mandatory:
                cache[route] = None
            else:
                schedule = decode_replanned_route(
                    data=data, state=problem.state,
                    scenario_instance=problem.scenario_instance,
                    route=route, constraints=problem.constraints,
                )
                cache[route] = schedule if schedule.feasible else None
        return cache[route]

    def quality(route: tuple[int, ...], schedule: ScheduleResult):
        utility = problem.locked_utility + sum(data.pois[p].utility_score for p in route)
        return (round(utility, 9), problem.locked_completed_count + len(route),
                -round(problem.locked_travel_min + schedule.total_travel_min, 9))

    def better(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
        ls, rs = evaluate(left), evaluate(right)
        if ls is None:
            return False
        if rs is None:
            return True
        return (quality(left, ls), tuple(-x for x in left)) > (quality(right, rs), tuple(-x for x in right))

    def insert(route: tuple[int, ...]) -> tuple[int, ...]:
        while len(route) < maximum:
            chosen = route
            for poi in sorted(allowed - set(route)):
                for position in range(len(route) + 1):
                    trial = route[:position] + (poi,) + route[position:]
                    if better(trial, chosen):
                        chosen = trial
            if chosen == route:
                break
            route = chosen
        return route

    def local_search(route: tuple[int, ...]) -> tuple[int, ...]:
        while True:
            chosen = route
            for left in range(len(route)):
                for right in range(left + 1, len(route)):
                    swapped = list(route)
                    swapped[left], swapped[right] = swapped[right], swapped[left]
                    reversed_segment = route[:left] + tuple(reversed(route[left:right + 1])) + route[right + 1:]
                    for trial in (tuple(swapped), reversed_segment):
                        if better(trial, chosen):
                            chosen = trial
            if chosen == route:
                return route
            route = chosen

    current = tuple(problem.mandatory_backbone)
    if evaluate(current) is None:
        raise RuntimeError("No feasible mandatory backbone for ILS")
    best = local_search(insert(current))
    for _ in range(config.iterations):
        removable = [p for p in best if p not in mandatory]
        if not removable:
            break
        removed = set(rng.sample(removable, min(config.shake_count, len(removable))))
        shaken = tuple(p for p in best if p not in removed)
        candidate = local_search(insert(shaken))
        if better(candidate, best):
            best = candidate

    schedule = evaluate(best)
    assert schedule is not None
    return ILSResult(
        route=best, schedule=schedule, evaluations=len(cache),
        repair_record=build_repair_record(problem.state, problem, best),
    )
