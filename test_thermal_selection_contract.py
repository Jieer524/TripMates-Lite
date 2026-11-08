"""Regression checks for the Chapter 4 thermal-selection contract."""

from __future__ import annotations

from ga_core import Individual
from exposure_core import ExposureMetrics
from tripmates_lite_ga import (
    ThermalEvaluatedSolution,
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
        schedule=None,  # The selector does not inspect schedule data.
        achieved_utility=utility,
        completed_poi_count=visit_count,
        total_travel_min=travel_min,
        exposure=exposure,
    )


def main() -> None:
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
            "Thermal selection must apply the guardrails against "
            "the non-thermal reference and break a WES tie by utility."
        )

    ineligible = make_solution(
        route=(1, 14, 15),
        utility=89.0,
        visit_count=3,
        travel_min=5.0,
        total_outdoor_exposure_min=10.0,
        wes=1.0,
    )

    selected, used_fallback = select_thermal_solution(
        [Individual(solution=ineligible)],
        reference,
    )

    if not used_fallback or selected != reference:
        raise RuntimeError(
            "When no thermal route passes both guardrails, the "
            "selector must return the non-thermal reference."
        )

    print("PASS: thermal selection follows the reference guardrails.")


if __name__ == "__main__":
    main()
