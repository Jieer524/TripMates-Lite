"""Regression tests for request validation and dynamic GA constraints."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from ga_core import GAConfig
from itinerary_service import (
    ItineraryService,
    PREFERRED_UTILITY_MULTIPLIER,
    RequestValidationError,
    TripRequest,
)


ROOT = Path(__file__).resolve().parent


def main() -> None:
    service = ItineraryService.from_files(
        poi_file=ROOT / "poi_data_calculated_final.csv",
        matrix_file=ROOT / "walking_time_matrix_final.csv",
    )

    request = TripRequest(
        trip_date=date(2026, 9, 9),  # Wednesday
        start_poi_id=1,
        start_time_min=9 * 60 + 30,
        end_time_min=17 * 60 + 30,
        cost_budget_rm=160.0,
        mandatory_poi_ids=(4, 13),
        preferred_poi_ids=(12, 33),
        seed=1001,
    )

    instance = service.build_instance(request)

    if instance.effective_mandatory_poi_ids != (1, 4, 13):
        raise RuntimeError("The starting POI must join the mandatory set.")

    for preferred_poi_id in request.preferred_poi_ids:
        expected_utility = (
            service.data.pois[preferred_poi_id].utility_score
            * PREFERRED_UTILITY_MULTIPLIER
        )
        if instance.data.pois[preferred_poi_id].utility_score != expected_utility:
            raise RuntimeError("Preferred POIs did not receive the soft utility boost.")

    if instance.data.pois[2].utility_score != service.data.pois[2].utility_score:
        raise RuntimeError("A non-preferred POI was modified unexpectedly.")

    outcome = service.plan_initial_itinerary(
        instance,
        config=GAConfig(
            population_size=8,
            maximum_generations=2,
            tournament_size=3,
            crossover_probability=0.90,
            mutation_probability=0.20,
            elitism_count=2,
            stagnation_limit=2,
        ),
    )

    if (
        outcome.status != "success"
        or outcome.schedule is None
        or not outcome.schedule.feasible
        or set(instance.effective_mandatory_poi_ids)
        - set(outcome.schedule.completed_sequence)
    ):
        raise RuntimeError("The service did not produce a valid itinerary.")

    scenario = next(
        item
        for item in service.scenario_instances_for(instance)
        if item.instance_id == "S1_nominal"
    )

    if scenario.trigger_absolute_min != request.start_time_min + 120:
        raise RuntimeError("Scenario trigger was not aligned to trip start.")

    replanning_outcome = service.replan(
        instance=instance,
        initial_outcome=outcome,
        scenario_instance=scenario,
        planner_id="tripmates_lite",
        config=GAConfig(
            population_size=8,
            maximum_generations=2,
            tournament_size=3,
            crossover_probability=0.90,
            mutation_probability=0.20,
            elitism_count=2,
            stagnation_limit=2,
        ),
    )

    if (
        replanning_outcome.status != "success"
        or replanning_outcome.repaired_schedule is None
        or not replanning_outcome.repaired_schedule.feasible
        or replanning_outcome.exposure is None
        or "Heavy Rain" not in replanning_outcome.explanation
    ):
        raise RuntimeError(
            "The service did not produce a traceable repaired itinerary."
        )

    greedy_outcome = service.replan(
        instance=instance,
        initial_outcome=outcome,
        scenario_instance=scenario,
        planner_id="greedy_repair",
    )

    if (
        greedy_outcome.status != "success"
        or greedy_outcome.repaired_schedule is None
        or not greedy_outcome.repaired_schedule.feasible
        or greedy_outcome.exposure is None
    ):
        raise RuntimeError(
            "The service did not expose a valid Greedy Repair itinerary."
        )

    try:
        service.build_instance(
            TripRequest(
                trip_date=date(2026, 9, 9),
                start_poi_id=999,
                start_time_min=17 * 60,
                end_time_min=9 * 60,
                cost_budget_rm=-1.0,
                mandatory_poi_ids=(4, 4),
            )
        )
    except RequestValidationError as error:
        fields = {issue.field for issue in error.issues}
        expected = {
            "start_poi_id",
            "end_time_min",
            "cost_budget_rm",
            "mandatory_poi_ids",
        }

        if not expected.issubset(fields):
            raise RuntimeError("Validation did not report all input errors.")
    else:
        raise RuntimeError("Invalid request unexpectedly passed validation.")

    print("PASS: itinerary service validates requests and applies constraints.")


if __name__ == "__main__":
    main()
