"""
Contract tests for the configurable thermal trade-off selection policy.

Verifies the C2 utility-retention guardrail, the separately reported
utility and visit guardrails, the U* baseline definitions, the C2
selection priority and the structured fallback.
"""

from __future__ import annotations

from exposure_core import ExposureMetrics
from ga_core import Individual
from tripmates_lite_ga import (
    ThermalEvaluatedSolution,
    ThermalSelectionPolicy,
    evaluate_thermal_selection,
    select_thermal_solution,
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
        total_outdoor_exposure_min=total_outdoor_exposure_min,
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


def test_default_policy_reproduces_frozen_contract() -> None:
    reference = make_solution(
        route=(1, 2, 3, 4, 5),
        utility=100.0,
        visit_count=5,
        travel_min=30.0,
        total_outdoor_exposure_min=80.0,
        wes=120.0,
    )

    lower_utility = make_solution(
        route=(1, 6, 7, 8, 9),
        utility=90.0,
        visit_count=4,
        travel_min=20.0,
        total_outdoor_exposure_min=40.0,
        wes=100.0,
    )
    higher_utility = make_solution(
        route=(1, 10, 11, 12, 13),
        utility=95.0,
        visit_count=4,
        travel_min=25.0,
        total_outdoor_exposure_min=70.0,
        wes=100.0,
    )

    selected, used_fallback = select_thermal_solution(
        [
            Individual(solution=lower_utility),
            Individual(solution=higher_utility),
        ],
        reference,
    )

    if used_fallback or selected != higher_utility:
        raise RuntimeError(
            "Default policy must reproduce the frozen guardrail "
            "and break a WES tie by higher utility."
        )


def test_u_star_baselines() -> None:
    reference = make_solution(
        route=(1, 2),
        utility=100.0,
        visit_count=2,
        travel_min=10.0,
        total_outdoor_exposure_min=10.0,
        wes=100.0,
    )
    front = [
        Individual(
            solution=make_solution(
                route=(1, 3),
                utility=90.0,
                visit_count=2,
                travel_min=10.0,
                total_outdoor_exposure_min=10.0,
                wes=90.0,
            )
        )
    ]

    reference_based = evaluate_thermal_selection(
        front,
        reference,
        ThermalSelectionPolicy(
            alpha=0.90,
            utility_guardrail_baseline="non_thermal_reference",
        ),
    )
    if reference_based.u_star != 100.0:
        raise RuntimeError("U* must equal the reference utility.")

    front_based = evaluate_thermal_selection(
        front,
        reference,
        ThermalSelectionPolicy(
            alpha=0.90,
            utility_guardrail_baseline="front_maximum",
        ),
    )
    if front_based.u_star != 90.0:
        raise RuntimeError("U* must equal the front maximum.")

    combined = evaluate_thermal_selection(
        front,
        reference,
        ThermalSelectionPolicy(
            alpha=0.90,
            utility_guardrail_baseline="front_maximum_plus_reference",
        ),
    )
    if combined.u_star != 100.0:
        raise RuntimeError(
            "Combined U* must be the larger of front maximum and "
            "reference utility."
        )


def test_alpha_changes_eligibility() -> None:
    reference = make_solution(
        route=(1, 2),
        utility=100.0,
        visit_count=5,
        travel_min=10.0,
        total_outdoor_exposure_min=10.0,
        wes=100.0,
    )
    candidate = make_solution(
        route=(1, 3),
        utility=88.0,
        visit_count=5,
        travel_min=8.0,
        total_outdoor_exposure_min=8.0,
        wes=80.0,
    )

    front = [Individual(solution=candidate)]

    strict = evaluate_thermal_selection(
        front,
        reference,
        ThermalSelectionPolicy(
            alpha=0.90,
            utility_guardrail_baseline="non_thermal_reference",
        ),
    )
    if not strict.guardrail_fallback_used:
        raise RuntimeError(
            "Utility 88 < 0.90 * 100 must fall back at alpha=0.90."
        )

    relaxed = evaluate_thermal_selection(
        front,
        reference,
        ThermalSelectionPolicy(
            alpha=0.85,
            utility_guardrail_baseline="non_thermal_reference",
        ),
    )
    if relaxed.guardrail_fallback_used:
        raise RuntimeError(
            "Utility 88 >= 0.85 * 100 must be eligible at alpha=0.85."
        )

    if relaxed.utility_guardrail_passed != (True,):
        raise RuntimeError(
            "Utility-guardrail status must be reported per route."
        )


def test_separate_utility_and_visit_guardrails() -> None:
    reference = make_solution(
        route=(1, 2),
        utility=100.0,
        visit_count=5,
        travel_min=10.0,
        total_outdoor_exposure_min=10.0,
        wes=100.0,
    )
    # Utility passes but the visit guardrail fails.
    candidate = make_solution(
        route=(1, 3),
        utility=95.0,
        visit_count=2,
        travel_min=8.0,
        total_outdoor_exposure_min=8.0,
        wes=80.0,
    )

    outcome = evaluate_thermal_selection(
        [Individual(solution=candidate)],
        reference,
        ThermalSelectionPolicy(
            alpha=0.90,
            visit_guardrail_margin=1,
            utility_guardrail_baseline="non_thermal_reference",
        ),
    )

    if outcome.utility_guardrail_passed != (True,):
        raise RuntimeError("Utility guardrail should pass.")

    if outcome.visit_guardrail_passed != (False,):
        raise RuntimeError("Visit guardrail should fail.")

    if not outcome.guardrail_fallback_used:
        raise RuntimeError(
            "A failed visit guardrail must trigger the fallback."
        )


def test_c2_priority_differs_from_frozen() -> None:
    reference = make_solution(
        route=(1, 2),
        utility=100.0,
        visit_count=5,
        travel_min=30.0,
        total_outdoor_exposure_min=80.0,
        wes=120.0,
    )
    higher_utility = make_solution(
        route=(1, 10, 11, 12, 13),
        utility=95.0,
        visit_count=4,
        travel_min=25.0,
        total_outdoor_exposure_min=70.0,
        wes=100.0,
    )
    lower_travel = make_solution(
        route=(1, 6, 7, 8, 9),
        utility=90.0,
        visit_count=4,
        travel_min=20.0,
        total_outdoor_exposure_min=40.0,
        wes=100.0,
    )

    population = [
        Individual(solution=higher_utility),
        Individual(solution=lower_travel),
    ]

    frozen = evaluate_thermal_selection(
        population,
        reference,
        ThermalSelectionPolicy(
            alpha=0.90,
            utility_guardrail_baseline="non_thermal_reference",
            priority="frozen",
        ),
    )
    if frozen.selected_solution != higher_utility:
        raise RuntimeError(
            "Frozen priority must prefer higher utility on a WES tie."
        )

    c2 = evaluate_thermal_selection(
        population,
        reference,
        ThermalSelectionPolicy(
            alpha=0.90,
            utility_guardrail_baseline="non_thermal_reference",
            priority="c2",
        ),
    )
    if c2.selected_solution != lower_travel:
        raise RuntimeError(
            "C2 priority must prefer lower travel time on a WES tie."
        )


def main() -> None:
    test_default_policy_reproduces_frozen_contract()
    test_u_star_baselines()
    test_alpha_changes_eligibility()
    test_separate_utility_and_visit_guardrails()
    test_c2_priority_differs_from_frozen()

    print("PASS: thermal trade-off selection policy behaves as specified.")


if __name__ == "__main__":
    main()
