"""
Regression tests for the C2 alpha-independence fix.

The C2 sensitivity run must satisfy one design rule:

    The ``alpha`` parameter affects ONLY the final route-selection
    policy.  It must NOT affect the GA search evolution, the
    stagnation detection, the convergence generation or the evaluated
    Pareto-front population.

The fix separates one explicit, alpha-independent frozen search policy
from the alpha-sensitive final-selection policy.

Tests
-----
A. Search alpha-independence
   For a fixed (instance, seed), running with alpha in
   {1.00, 0.95, 0.90, 0.85} must produce an identical search: the same
   generations executed, stagnation stop, convergence generation,
   evaluated-chromosome count, unique-route count and Pareto front,
   and the same U* / guardrail baselines.

B. Final alpha sensitivity
   The final selection must still respond to alpha: a synthetic front
   where alpha changes eligibility/fallback, plus an integration check
   that the requested alpha is recorded and that per-front utility
   eligibility is monotone non-increasing in alpha.

C. Frozen default compatibility
   Running with no policy and with an explicit frozen default must be
   byte-identical, and the frozen default must leave the search policy
   equal to the final policy.

D. Smoke-output isolation (V2)
   The C2 runner targets the versioned V2 directory and redirects a
   ``--smoke`` run into ``<V2>/smoke/``.

Usage:
    python test_c2_alpha_independence.py
"""

from __future__ import annotations

import json
from pathlib import Path

from adaptive_ga_core import (
    REPLANNING_GA_CONFIG,
    prepare_adaptive_problem,
)
from exposure_core import ExposureMetrics
from ga_core import Individual
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
from tripmates_lite_ga import (
    DEFAULT_THERMAL_SELECTION_POLICY,
    ThermalEvaluatedSolution,
    ThermalSelectionPolicy,
    evaluate_thermal_selection,
    run_tripmates_lite_ga,
)


POI_FILE = Path("poi_data_calculated_final.csv")
MATRIX_FILE = Path("walking_time_matrix_final.csv")
INITIAL_ROUTES_FILE = Path(
    "results/static_utility/initial_routes_by_seed.json"
)

ALPHAS = (1.00, 0.95, 0.90, 0.85)

SAMPLE_INSTANCE_IDS = ("S1_nominal", "S5_nominal")
SAMPLE_SEEDS = (1001, 1005)

FROZEN_ALPHA = 0.90
FROZEN_PRIORITY = "frozen"
FROZEN_BASELINE = "non_thermal_reference"


# ================================================================
# Shared helpers
# ================================================================

def c2_policy(alpha: float) -> ThermalSelectionPolicy:
    """The C2 policy used by the sensitivity runner."""

    return ThermalSelectionPolicy(
        alpha=float(alpha),
        visit_guardrail_margin=1,
        utility_guardrail_baseline=(
            "front_maximum_plus_reference"
        ),
        priority="c2",
    )


def build_context():
    data = load_planning_data(
        poi_file=POI_FILE,
        matrix_file=MATRIX_FILE,
    )
    routes_by_seed = json.loads(
        INITIAL_ROUTES_FILE.read_text(encoding="utf-8")
    )
    scenario_lookup = {
        instance.instance_id: instance
        for instance in build_scenario_instances()
    }

    return data, routes_by_seed, scenario_lookup


def build_problem(
    data,
    routes_by_seed,
    scenario_lookup,
    *,
    seed: int,
    instance_id: str,
):
    initial_route = [
        int(poi_id)
        for poi_id in routes_by_seed[str(int(seed))]
    ]

    initial_schedule = decode_route(
        data=data,
        visit_sequence=initial_route,
        start_poi_id=1,
        start_time_min=TRIP_START_MIN,
        trip_end_min=TRIP_END_MIN,
        day_code=EXPERIMENT_DAY,
        cost_budget_rm=COST_BUDGET_RM,
        start_poi_is_visit=True,
        required_mandatory_ids=data.mandatory_ids,
    )

    if not initial_schedule.feasible:
        raise RuntimeError(
            f"Seed {seed}: initial route is infeasible."
        )

    scenario_instance = scenario_lookup[instance_id]

    state = simulate_replanning_state(
        data=data,
        initial_schedule=initial_schedule,
        scenario_instance=scenario_instance,
    )

    problem = prepare_adaptive_problem(
        data=data,
        initial_schedule=initial_schedule,
        state=state,
        scenario_instance=scenario_instance,
    )

    return initial_schedule, problem


def front_signature(result) -> tuple:
    """Order-independent identity of the evaluated first front."""

    return tuple(
        sorted(
            (
                tuple(solution.route),
                round(solution.achieved_utility, 9),
                round(solution.wes, 9),
                round(solution.total_travel_min, 9),
                round(
                    solution.exposure
                    .total_outdoor_exposure_min,
                    9,
                ),
                solution.completed_poi_count,
            )
            for solution in result.first_front
        )
    )


def search_signature(result) -> tuple:
    """Everything that must be identical across C2 alphas."""

    return (
        result.generations_executed,
        result.stopped_by_stagnation,
        result.convergence_generation,
        result.evaluated_chromosome_count,
        result.unique_decoded_route_count,
        front_signature(result),
        round(result.u_star, 9),
        round(result.reference_utility, 9),
        round(result.front_max_utility, 9),
        result.front_visit_guardrail,
    )


def make_solution(
    *,
    route: tuple[int, ...],
    utility: float,
    visit_count: int,
    travel_min: float,
    total_outdoor_exposure_min: float,
    wes: float,
) -> ThermalEvaluatedSolution:
    exposure = ExposureMetrics(
        prefix_walking_min=0.0,
        prefix_visit_min=0.0,
        repaired_walking_min=0.0,
        repaired_visit_min=0.0,
        outdoor_walking_min=0.0,
        outdoor_visit_min=0.0,
        total_outdoor_exposure_min=(
            total_outdoor_exposure_min
        ),
        weather_weighted_exposure_score=wes,
        thermal_multiplier=1.8,
    )

    return ThermalEvaluatedSolution(
        chromosome=route,
        route=route,
        schedule=None,
        achieved_utility=utility,
        completed_poi_count=visit_count,
        total_travel_min=travel_min,
        exposure=exposure,
    )


# ================================================================
# A. Search alpha-independence
# ================================================================

def test_a_search_is_alpha_independent() -> None:
    data, routes_by_seed, scenario_lookup = (
        build_context()
    )

    comparisons = 0

    for instance_id in SAMPLE_INSTANCE_IDS:
        for seed in SAMPLE_SEEDS:
            initial_schedule, problem = build_problem(
                data,
                routes_by_seed,
                scenario_lookup,
                seed=seed,
                instance_id=instance_id,
            )

            results = {
                alpha: run_tripmates_lite_ga(
                    data=data,
                    initial_schedule=initial_schedule,
                    problem=problem,
                    seed=seed,
                    config=REPLANNING_GA_CONFIG,
                    selection_policy=c2_policy(alpha),
                )
                for alpha in ALPHAS
            }

            baseline = results[ALPHAS[0]]
            baseline_search = search_signature(baseline)

            for alpha in ALPHAS[1:]:
                result = results[alpha]

                if search_signature(result) != baseline_search:
                    raise AssertionError(
                        "C2 search must not depend on alpha, but the "
                        f"search changed for {instance_id} seed "
                        f"{seed} at alpha={alpha}."
                    )

                # The recorded search policy must always be the
                # explicit frozen default, never the requested alpha.
                if (
                    result.search_selection_policy
                    != DEFAULT_THERMAL_SELECTION_POLICY
                ):
                    raise AssertionError(
                        "The internal search policy must be the "
                        "frozen default for every alpha."
                    )

                # The final policy must record the requested alpha.
                if (
                    result.final_selection_policy.alpha
                    != alpha
                ):
                    raise AssertionError(
                        "The final selection policy must record the "
                        f"requested alpha ({alpha})."
                    )

                comparisons += 1

            if (
                baseline.search_selection_policy.alpha
                != FROZEN_ALPHA
            ):
                raise AssertionError(
                    "The frozen search policy alpha changed."
                )

            if (
                baseline.search_selection_policy.priority
                != FROZEN_PRIORITY
            ):
                raise AssertionError(
                    "The frozen search policy priority changed."
                )

            if (
                baseline
                .search_selection_policy
                .utility_guardrail_baseline
                != FROZEN_BASELINE
            ):
                raise AssertionError(
                    "The frozen search policy guardrail baseline "
                    "changed."
                )

    print(
        "PASS (A): C2 search evolution, stagnation detection, "
        "convergence generation and Pareto front are "
        f"alpha-independent ({comparisons} alpha comparisons)."
    )


# ================================================================
# B. Final alpha sensitivity
# ================================================================

def test_b_final_selection_is_alpha_sensitive() -> None:
    # ------------------------------------------------------------
    # Synthetic front: alpha changes eligibility and fallback.
    # ------------------------------------------------------------

    reference = make_solution(
        route=(1, 2, 3, 4, 5),
        utility=100.0,
        visit_count=5,
        travel_min=30.0,
        total_outdoor_exposure_min=80.0,
        wes=120.0,
    )
    candidate = make_solution(
        route=(1, 6, 7, 8, 9),
        utility=88.0,
        visit_count=5,
        travel_min=18.0,
        total_outdoor_exposure_min=35.0,
        wes=95.0,
    )

    population = [Individual(solution=candidate)]

    strict = evaluate_thermal_selection(
        population,
        reference,
        c2_policy(1.00),
    )
    relaxed = evaluate_thermal_selection(
        population,
        reference,
        c2_policy(0.85),
    )

    if not strict.guardrail_fallback_used:
        raise AssertionError(
            "Utility 88 < 1.00 * 100 must fall back at alpha=1.00."
        )

    if relaxed.guardrail_fallback_used:
        raise AssertionError(
            "Utility 88 >= 0.85 * 100 must be eligible at "
            "alpha=0.85."
        )

    if relaxed.selected_solution != candidate:
        raise AssertionError(
            "The relaxed alpha must select the low-WES candidate."
        )

    if strict.selected_solution != reference:
        raise AssertionError(
            "The strict alpha must return the non-thermal "
            "reference."
        )

    # ------------------------------------------------------------
    # Integration: alpha is recorded, eligibility is monotone and
    # the final route may differ across alphas.
    # ------------------------------------------------------------

    data, routes_by_seed, scenario_lookup = (
        build_context()
    )

    saw_alpha_dependent_final = False

    for instance_id in SAMPLE_INSTANCE_IDS:
        for seed in SAMPLE_SEEDS:
            initial_schedule, problem = build_problem(
                data,
                routes_by_seed,
                scenario_lookup,
                seed=seed,
                instance_id=instance_id,
            )

            previous_eligible = None
            selected_routes = set()

            for alpha in ALPHAS:
                result = run_tripmates_lite_ga(
                    data=data,
                    initial_schedule=initial_schedule,
                    problem=problem,
                    seed=seed,
                    config=REPLANNING_GA_CONFIG,
                    selection_policy=c2_policy(alpha),
                )

                if (
                    result.final_selection_policy.alpha
                    != alpha
                ):
                    raise AssertionError(
                        "The requested alpha must be recorded on "
                        "the final selection policy."
                    )

                # Eligibility can only shrink as alpha grows, so a
                # route eligible at a stricter alpha must stay
                # eligible at a looser one.
                if previous_eligible is not None:
                    for stricter, looser in zip(
                        previous_eligible,
                        result.front_guardrail_eligible,
                    ):
                        if stricter and not looser:
                            raise AssertionError(
                                "Utility eligibility must be "
                                "monotone non-increasing in "
                                "alpha."
                            )

                previous_eligible = (
                    result.front_guardrail_eligible
                )

                selected_routes.add(
                    tuple(result.selected_solution.route)
                )

            if len(selected_routes) > 1:
                saw_alpha_dependent_final = True

    if not saw_alpha_dependent_final:
        raise AssertionError(
            "The final selection must respond to alpha for at "
            "least one sampled instance/seed."
        )

    print(
        "PASS (B): the final route selection remains "
        "alpha-sensitive."
    )


# ================================================================
# C. Frozen default compatibility
# ================================================================

def test_c_frozen_default_compatibility() -> None:
    data, routes_by_seed, scenario_lookup = (
        build_context()
    )

    instance_id = SAMPLE_INSTANCE_IDS[0]
    seed = SAMPLE_SEEDS[0]

    initial_schedule, problem = build_problem(
        data,
        routes_by_seed,
        scenario_lookup,
        seed=seed,
        instance_id=instance_id,
    )

    implicit = run_tripmates_lite_ga(
        data=data,
        initial_schedule=initial_schedule,
        problem=problem,
        seed=seed,
        config=REPLANNING_GA_CONFIG,
    )

    explicit = run_tripmates_lite_ga(
        data=data,
        initial_schedule=initial_schedule,
        problem=problem,
        seed=seed,
        config=REPLANNING_GA_CONFIG,
        selection_policy=ThermalSelectionPolicy(),
    )

    if search_signature(implicit) != search_signature(
        explicit
    ):
        raise AssertionError(
            "Omitting the policy must equal the explicit frozen "
            "default."
        )

    if (
        implicit.selected_solution.route
        != explicit.selected_solution.route
    ):
        raise AssertionError(
            "The frozen default must select the same route with "
            "and without an explicit policy."
        )

    if (
        implicit.guardrail_fallback_used
        != explicit.guardrail_fallback_used
    ):
        raise AssertionError(
            "The frozen default fallback status changed."
        )

    if (
        implicit.selection_status
        != explicit.selection_status
    ):
        raise AssertionError(
            "The frozen default selection status changed."
        )

    # With the frozen default the search and final policies coincide,
    # and the backward-compatible alias still resolves.
    if (
        implicit.search_selection_policy
        != implicit.final_selection_policy
    ):
        raise AssertionError(
            "The frozen default must use one policy for search and "
            "final selection."
        )

    if (
        implicit.selection_policy
        != implicit.final_selection_policy
    ):
        raise AssertionError(
            "The selection_policy alias must resolve to the final "
            "selection policy."
        )

    print(
        "PASS (C): the frozen default policy is preserved exactly "
        "and search/final policies coincide for it."
    )


# ================================================================
# D. Smoke-output isolation (V2)
# ================================================================

def test_d_smoke_output_isolation_v2() -> None:
    import run_thermal_tradeoff_sensitivity as runner

    full_directory = Path(runner.OUTPUT_DIRECTORY)

    if (
        full_directory.name
        != "thermal_tradeoff_sensitivity_alpha_independent_v2"
    ):
        raise AssertionError(
            "The C2 runner must target the versioned V2 directory."
        )

    if (
        runner.SUPERSEDES_DIRECTORY
        != "results/thermal_tradeoff_sensitivity/"
    ):
        raise AssertionError(
            "The V2 runner must declare the superseded directory."
        )

    if runner.IS_CONFIRMATORY is not False:
        raise AssertionError(
            "The V2 run must be exploratory, not confirmatory."
        )

    if (
        runner.SEARCH_POLICY_MODE
        != "alpha_independent_frozen_policy"
    ):
        raise AssertionError(
            "Unexpected search-policy mode label."
        )

    if (
        runner.FINAL_SELECTION_POLICY_MODE
        != "alpha_sensitive_exploratory"
    ):
        raise AssertionError(
            "Unexpected final-selection-policy mode label."
        )

    file_attributes = (
        "RAW_RUNS_FILE",
        "PARETO_FRONTS_FILE",
        "INSTANCE_SUMMARY_FILE",
        "RUN_CONFIG_FILE",
        "CHART_WES_FILE",
        "CHART_TRADEOFF_FILE",
    )

    for name in file_attributes:
        if Path(getattr(runner, name)).parent != (
            full_directory
        ):
            raise AssertionError(
                f"{name} is not inside the V2 directory."
            )

    runner.isolate_smoke_outputs()

    smoke_directory = full_directory / "smoke"

    if Path(runner.OUTPUT_DIRECTORY) != smoke_directory:
        raise AssertionError(
            "A smoke run must redirect the output directory into "
            "the V2 smoke folder."
        )

    for name in file_attributes:
        redirected = Path(getattr(runner, name))

        if redirected.parent != smoke_directory:
            raise AssertionError(
                f"{name} must be redirected into the V2 smoke "
                "folder."
            )

    print(
        "PASS (D): the C2 runner targets the V2 directory and "
        "isolates smoke output under V2/smoke."
    )


def main() -> None:
    test_a_search_is_alpha_independent()
    test_b_final_selection_is_alpha_sensitive()
    test_c_frozen_default_compatibility()
    test_d_smoke_output_isolation_v2()

    print(
        "PASS: C2 alpha affects only the final selection policy."
    )


if __name__ == "__main__":
    main()
