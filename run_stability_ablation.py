"""Selection-policy ablation: does a stability term change the route?

Research question
-----------------
The published TripMates Lite pipeline *reports* itinerary-stability
metrics (``stability_metrics.py``) but no planner consumes them: the
final route is chosen by the frozen selection rule, whose key is travel
time (non-thermal) or WES (thermal) and which contains no stability
term at all.  The "no-stability-penalty ablation" is therefore not a
removal — the penalty does not exist — but a controlled *addition*:

    what would change if the same GA population were re-ranked with a
    stability penalty, and what does the eligibility guardrail
    contribute?

Design
------
The GA is run **once** per ``(planner, scenario, seed)``.  Every
condition then re-selects a route from the *identical* final Pareto
front, so the population, the generations, the candidate window, the
decoder and the seed are provably identical across conditions.  Nothing
about the search is perturbed; only the selection key and the
eligibility guardrail change.  This is a 2 x 2 factorial:

===========================  ==================  ==============
condition                    selection key       guardrail
===========================  ==================  ==============
``frozen``                   frozen (travel/WES)  applied
``frozen_key_unguarded``     frozen (travel/WES)  disabled
``stability_guarded``        stability penalty    applied
``stability_unguarded``      stability penalty    disabled
===========================  ==================  ==============

``frozen`` reproduces the published route exactly, which is asserted by
``tests/test_ablation_studies.py``.

Interpretation rules
--------------------
- ``frozen`` vs ``stability_guarded`` isolates the *selection key*.
- ``stability_guarded`` vs ``stability_unguarded`` isolates the
  *guardrail*.
- A stability-preferring key can only trade utility and exposure for a
  smaller edit distance; the study measures that trade, it does not
  assert that either side is better.

Usage
-----
    python run_stability_ablation.py [--smoke]
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from ablation_common import (
    NOMINAL_INSTANCE_IDS,
    STUDY_STABILITY,
    build_manifest,
    load_study_inputs,
    print_study_footer,
    seeds_for_run,
    study_files,
    tag_records,
    write_study_outputs,
)
from adaptive_ga_core import (
    REPLANNING_GA_CONFIG,
    calculate_locked_travel_minutes,
    prepare_adaptive_problem,
    run_adaptive_without_thermal_ga,
)
from benchmark_common import build_metric_record
from exposure_core import calculate_exposure_metrics
from planning_core import ScheduleResult
from run_rq1_full_benchmark import (
    PLANNER_ADAPTIVE,
    PLANNER_TRIPMATES,
    PLANNERS,
)
from scenario_core import simulate_replanning_state
from stability_metrics import (
    StabilityWeightConfig,
    compute_stability_metrics,
)
from tripmates_lite_ga import run_tripmates_lite_ga


STUDY_PLANNERS = (PLANNER_ADAPTIVE, PLANNER_TRIPMATES)

FACTOR_COLUMN = "selection_policy"

# ---- condition identifiers -------------------------------------

SELECTION_FROZEN = "frozen"
SELECTION_FROZEN_UNGUARDED = "frozen_key_unguarded"
SELECTION_STABILITY_GUARDED = "stability_guarded"
SELECTION_STABILITY_UNGUARDED = "stability_unguarded"

SELECTION_POLICIES: tuple[str, ...] = (
    SELECTION_FROZEN,
    SELECTION_FROZEN_UNGUARDED,
    SELECTION_STABILITY_GUARDED,
    SELECTION_STABILITY_UNGUARDED,
)

SELECTION_POLICY_DESCRIPTIONS: dict[str, dict] = {
    SELECTION_FROZEN: {
        "selection_key": "frozen_objective",
        "guardrail_applied": True,
        "description": (
            "Published rule: eligible first-front routes only "
            "(utility >= 0.90 U*, count >= N* - 1 for the non-thermal "
            "planner; the frozen alpha / visit guardrails for the "
            "thermal planner), then minimum travel time (non-thermal) "
            "or minimum WES (thermal)."
        ),
    },
    SELECTION_FROZEN_UNGUARDED: {
        "selection_key": "frozen_objective",
        "guardrail_applied": False,
        "description": (
            "Same key as the published rule but the eligibility "
            "guardrail is disabled, so a low-utility, low-travel route "
            "on the first front can win."
        ),
    },
    SELECTION_STABILITY_GUARDED: {
        "selection_key": "stability_penalty",
        "guardrail_applied": True,
        "description": (
            "Published eligibility guardrail retained, but the route "
            "with the smallest weighted itinerary edit distance wins; "
            "ties broken by travel time, then utility, then route."
        ),
    },
    SELECTION_STABILITY_UNGUARDED: {
        "selection_key": "stability_penalty",
        "guardrail_applied": False,
        "description": (
            "Smallest weighted itinerary edit distance over the whole "
            "first front, with no utility or visit guardrail.  This is "
            "the pure stability-first extreme."
        ),
    },
}


# ================================================================
# Front access and selection
# ================================================================

def front_solutions(result) -> tuple:
    """Return the final first Pareto front of a GA run."""

    front = tuple(getattr(result, "first_front", ()))

    if not front:
        raise RuntimeError(
            "The GA result exposes no first Pareto front."
        )

    return front


def guardrail_mask(
    *,
    planner: str,
    result,
) -> tuple[bool, ...]:
    """Eligibility mask aligned with ``front_solutions(result)``.

    The thermal planner reports its own mask (it is measured against
    the seed-matched non-thermal reference and the frozen ``alpha``),
    so this function never re-derives the thermal guardrail.  The
    non-thermal mask mirrors ``ga_core.select_final_solution`` exactly.
    """

    front = front_solutions(result)

    if planner == PLANNER_TRIPMATES:
        reported = tuple(
            getattr(result, "front_guardrail_eligible", ())
        )

        if len(reported) == len(front):
            return tuple(bool(value) for value in reported)

    maximum_utility = max(
        solution.achieved_utility for solution in front
    )
    maximum_count = max(
        solution.completed_poi_count for solution in front
    )

    return tuple(
        solution.achieved_utility >= 0.90 * maximum_utility
        and solution.completed_poi_count >= maximum_count - 1
        for solution in front
    )


def frozen_key(solution, *, planner: str) -> tuple:
    """The published selection key, per planner family."""

    if planner == PLANNER_TRIPMATES:
        return (
            solution.wes,
            -solution.achieved_utility,
            solution.exposure.total_outdoor_exposure_min,
            solution.total_travel_min,
            solution.route,
        )

    return (
        solution.total_travel_min,
        -solution.achieved_utility,
        solution.route,
    )


def stability_key(
    solution,
    *,
    penalties: dict[int, float],
) -> tuple:
    return (
        penalties[id(solution)],
        solution.total_travel_min,
        -solution.achieved_utility,
        solution.route,
    )


def front_stability_penalties(
    *,
    front,
    data,
    state,
    initial_schedule: ScheduleResult,
    weights: StabilityWeightConfig,
) -> dict[int, float]:
    """Weighted edit distance of every first-front route."""

    locked_prefix = list(state.completed_poi_ids)
    original_suffix = list(state.original_remaining_poi_ids)

    penalties: dict[int, float] = {}

    for solution in front:
        metrics = compute_stability_metrics(
            original_suffix=original_suffix,
            replanned_suffix=tuple(solution.route),
            locked_prefix=locked_prefix,
            original_schedule=initial_schedule,
            replanned_schedule=solution.schedule,
            weights=weights,
            final_route=locked_prefix + list(solution.route),
            replanning_feasible=True,
        )

        penalties[id(solution)] = metrics.edit_distance_score

    return penalties


def select_for_policy(
    *,
    planner: str,
    policy: str,
    result,
    penalties: dict[int, float],
) -> tuple[object, bool]:
    """Return ``(solution, guardrail_fallback_used)`` for one policy."""

    if policy == SELECTION_FROZEN:
        # The published route, produced by the frozen rule itself.
        return result.selected_solution, bool(
            getattr(result, "guardrail_fallback_used", False)
        )

    front = front_solutions(result)

    if policy in (
        SELECTION_FROZEN_UNGUARDED,
        SELECTION_STABILITY_UNGUARDED,
    ):
        candidates = list(front)
        fallback_used = False
    else:
        mask = guardrail_mask(planner=planner, result=result)
        candidates = [
            solution
            for solution, eligible in zip(front, mask)
            if eligible
        ]
        fallback_used = False

        if not candidates:
            # No front route passes the guardrail.  The frozen rule
            # falls back to the seed-matched non-thermal reference;
            # the stability conditions keep the frozen route rather
            # than inventing a third behaviour.
            return result.selected_solution, True

    if policy in (
        SELECTION_FROZEN_UNGUARDED,
        SELECTION_FROZEN,
    ):
        key = lambda solution: frozen_key(  # noqa: E731
            solution, planner=planner
        )
    else:
        key = lambda solution: stability_key(  # noqa: E731
            solution, penalties=penalties
        )

    return min(candidates, key=key), fallback_used


# ================================================================
# Record building
# ================================================================

def solution_exposure(
    *,
    solution,
    data,
    state,
    initial_schedule: ScheduleResult,
    thermal_multiplier: float,
):
    """Return the solution's exposure block, computing it if absent."""

    exposure = getattr(solution, "exposure", None)

    if exposure is not None:
        return exposure

    return calculate_exposure_metrics(
        data=data,
        initial_schedule=initial_schedule,
        state=state,
        repaired_schedule=solution.schedule,
        thermal_multiplier=thermal_multiplier,
    )


def selection_record(
    *,
    planner: str,
    policy: str,
    result,
    solution,
    front_size: int,
    front_mean_edit_distance: float,
    selected_edit_distance: float,
    guardrail_fallback_used: bool,
    data,
    state,
    scenario_instance,
    seed: int,
    initial_route: list[int],
    initial_schedule: ScheduleResult,
    initial_utility: float,
    runtime_seconds: float,
) -> dict:
    """Build the metric record for a re-selected solution.

    The record uses the same ``benchmark_common.build_metric_record``
    schema as every other runner, so outcome and stability columns mean
    exactly the same thing here as in the RQ1 benchmark.  The only
    difference from ``ga_planner_record`` is the source of the route:
    here it is chosen by the study, not by the frozen selection rule.
    """

    exposure = solution_exposure(
        solution=solution,
        data=data,
        state=state,
        initial_schedule=initial_schedule,
        thermal_multiplier=(
            scenario_instance.scenario.thermal_multiplier
        ),
    )

    locked_travel = calculate_locked_travel_minutes(
        initial_schedule=initial_schedule,
        state=state,
    )

    description = SELECTION_POLICY_DESCRIPTIONS[policy]

    return build_metric_record(
        planner=planner,
        scenario_instance=scenario_instance,
        seed=seed,
        initial_route=initial_route,
        initial_schedule=initial_schedule,
        initial_route_utility=initial_utility,
        state=state,
        repaired_route=list(solution.route),
        achieved_utility=solution.achieved_utility,
        completed_poi_count=solution.completed_poi_count,
        total_travel_min=solution.total_travel_min,
        locked_travel_min=locked_travel,
        repaired_schedule=solution.schedule,
        exposure=exposure,
        feasible=True,
        failure_reason=None,
        mandatory_complete=(
            set(state.remaining_mandatory_ids)
            <= set(solution.route)
        ),
        runtime_seconds=runtime_seconds,
        deterministic=False,
        generations_executed=getattr(
            result, "generations_executed", None
        ),
        convergence_generation=getattr(
            result, "convergence_generation", None
        ),
        evaluated_chromosomes=getattr(
            result, "evaluated_chromosome_count", None
        ),
        unique_decoded_routes=getattr(
            result, "unique_decoded_route_count", None
        ),
        pareto_front_size=front_size,
        repair_record=getattr(result, "repair_record", None),
        extra={
            FACTOR_COLUMN: policy,
            "selection_key": description["selection_key"],
            "guardrail_applied": description["guardrail_applied"],
            "guardrail_fallback_used": guardrail_fallback_used,
            "front_size": front_size,
            "front_mean_edit_distance": front_mean_edit_distance,
            "selected_edit_distance": selected_edit_distance,
        },
    )


# ================================================================
# Runner
# ================================================================

def run_study(*, smoke: bool) -> pd.DataFrame:
    inputs = load_study_inputs()

    seeds = seeds_for_run(smoke)
    instance_ids = (
        ("S1_nominal",) if smoke else NOMINAL_INSTANCE_IDS
    )

    weights = StabilityWeightConfig()

    records: list[dict] = []

    for instance_id in instance_ids:
        scenario_instance = inputs.instance(instance_id)

        for seed in seeds:
            initial_route = inputs.route(seed)
            initial_schedule = inputs.schedule(seed)
            initial_utility = inputs.utility(seed)

            state = simulate_replanning_state(
                data=inputs.data,
                initial_schedule=initial_schedule,
                scenario_instance=scenario_instance,
            )

            try:
                problem = prepare_adaptive_problem(
                    data=inputs.data,
                    initial_schedule=initial_schedule,
                    state=state,
                    scenario_instance=scenario_instance,
                    constraints=inputs.constraints,
                )
            except (RuntimeError, ValueError, KeyError) as error:
                records.extend(
                    tag_records(
                        [
                            _problem_failure_record(
                                planner=planner,
                                data=inputs.data,
                                state=state,
                                scenario_instance=scenario_instance,
                                seed=seed,
                                initial_route=initial_route,
                                initial_schedule=initial_schedule,
                                initial_utility=initial_utility,
                                reason=(
                                    "problem preparation failed: "
                                    f"{error}"
                                ),
                                policy=policy,
                            )
                            for planner in STUDY_PLANNERS
                            for policy in SELECTION_POLICIES
                        ]
                    )
                )
                continue

            for planner in STUDY_PLANNERS:
                started_at = time.perf_counter()

                if planner == PLANNER_ADAPTIVE:
                    result = run_adaptive_without_thermal_ga(
                        data=inputs.data,
                        problem=problem,
                        seed=seed,
                        config=REPLANNING_GA_CONFIG,
                    )
                else:
                    result = run_tripmates_lite_ga(
                        data=inputs.data,
                        initial_schedule=initial_schedule,
                        problem=problem,
                        seed=seed,
                        config=REPLANNING_GA_CONFIG,
                    )

                ga_runtime = time.perf_counter() - started_at

                front = front_solutions(result)
                penalties = front_stability_penalties(
                    front=front,
                    data=inputs.data,
                    state=state,
                    initial_schedule=initial_schedule,
                    weights=weights,
                )
                mean_edit_distance = (
                    sum(penalties.values()) / len(penalties)
                )

                for policy in SELECTION_POLICIES:
                    selection_started = time.perf_counter()

                    solution, fallback_used = select_for_policy(
                        planner=planner,
                        policy=policy,
                        result=result,
                        penalties=penalties,
                    )

                    selection_runtime = (
                        time.perf_counter() - selection_started
                    )

                    records.append(
                        selection_record(
                            planner=planner,
                            policy=policy,
                            result=result,
                            solution=solution,
                            front_size=len(front),
                            front_mean_edit_distance=(
                                mean_edit_distance
                            ),
                            selected_edit_distance=penalties.get(
                                id(solution),
                                0.0,
                            ),
                            guardrail_fallback_used=fallback_used,
                            data=inputs.data,
                            state=state,
                            scenario_instance=scenario_instance,
                            seed=seed,
                            initial_route=initial_route,
                            initial_schedule=initial_schedule,
                            initial_utility=initial_utility,
                            runtime_seconds=(
                                ga_runtime + selection_runtime
                            ),
                        )
                    )

                print(
                    f"  {instance_id} seed={seed} {planner} "
                    f"front={len(front)} "
                    f"policies={len(SELECTION_POLICIES)}"
                )

    return pd.DataFrame(records)


def _problem_failure_record(
    *,
    planner: str,
    data,
    state,
    scenario_instance,
    seed: int,
    initial_route: list[int],
    initial_schedule: ScheduleResult,
    initial_utility: float,
    reason: str,
    policy: str,
) -> dict:
    from ablation_common import failure_record

    return failure_record(
        planner=planner,
        scenario_instance=scenario_instance,
        seed=seed,
        initial_route=initial_route,
        initial_schedule=initial_schedule,
        initial_route_utility=initial_utility,
        state=state,
        failure_reason=reason,
        runtime_seconds=0.0,
        deterministic=False,
    ) | {
        FACTOR_COLUMN: policy,
        "selection_key": (
            SELECTION_POLICY_DESCRIPTIONS[policy]["selection_key"]
        ),
        "guardrail_applied": (
            SELECTION_POLICY_DESCRIPTIONS[policy][
                "guardrail_applied"
            ]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Selection-policy ablation over an identical GA population."
        )
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    started_at = time.perf_counter()

    frame = run_study(smoke=args.smoke)

    total_runtime = time.perf_counter() - started_at

    failure_count = int((~frame["feasible"].astype(bool)).sum())

    study_seeds = seeds_for_run(args.smoke)
    study_instances = (
        ("S1_nominal",) if args.smoke else NOMINAL_INSTANCE_IDS
    )

    manifest = build_manifest(
        study=STUDY_STABILITY,
        smoke=args.smoke,
        factors={
            "factor": FACTOR_COLUMN,
            "levels": list(SELECTION_POLICIES),
            "level_definitions": SELECTION_POLICY_DESCRIPTIONS,
            "design": (
                "2 x 2 factorial (selection key x eligibility "
                "guardrail) evaluated on one identical Pareto front "
                "per planner, scenario and seed."
            ),
            "scenario_instances": list(study_instances),
            "planners": list(STUDY_PLANNERS),
            "ga_run_count": (
                len(study_instances)
                * len(study_seeds)
                * len(STUDY_PLANNERS)
            ),
            "records_per_ga_run": len(SELECTION_POLICIES),
            "note": (
                "The GA is executed once per (planner, scenario, seed); "
                "the four conditions differ only in how a route is "
                "chosen from the resulting front, so runtime is shared "
                "and reported identically across conditions."
            ),
        },
        seeds=study_seeds,
        planners=list(STUDY_PLANNERS),
        run_count=len(frame),
        failure_count=failure_count,
        total_runtime_seconds=total_runtime,
        extra={
            "primary_contrast": [
                SELECTION_FROZEN,
                SELECTION_STABILITY_GUARDED,
            ],
            "direction_note": (
                "The published pipeline applies no stability penalty. "
                "The stability conditions are counterfactual additions; "
                "this study cannot show that the published method is "
                "better or worse, only what a stability term would "
                "trade."
            ),
            "all_planners": list(PLANNERS),
        },
    )

    files = write_study_outputs(
        study=STUDY_STABILITY,
        smoke=args.smoke,
        frame=frame,
        factor_columns=[FACTOR_COLUMN],
        manifest=manifest,
    )

    print_study_footer(files=files, manifest=manifest)


if __name__ == "__main__":
    main()
