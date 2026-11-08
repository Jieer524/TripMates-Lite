"""Category C - repair behaviour.

Covers the deterministic Greedy Repair baseline and the shared stability
metrics.  The repair tests use the six-POI synthetic instance; the
stability tests use pure in-memory itineraries so they are exact and
instantaneous.
"""

from __future__ import annotations

import pytest

from adaptive_ga_core import (
    BoundedRepairConfig,
    MAXIMUM_CANDIDATE_COUNT,
    prepare_adaptive_problem,
)
from conftest import (
    TRIGGER_DURING_FIRST_VISIT,
    TRIGGER_DURING_VISIT,
    TRIGGER_DURING_WAITING,
    TRIGGER_DURING_WALKING,
    replanning_state_at,
)
from greedy_repair import run_greedy_repair
from planning_core import PlanningConstraints, decode_route
from stability_metrics import compute_stability_metrics


SEED = 1001


def _repair(
    planning_data,
    initial_schedule,
    trigger_offset_min,
    *,
    constraints=None,
    bounded_repair_config=None,
    **scenario_kwargs,
):
    """Simulate a disruption and run Greedy Repair on it."""

    state, instance = replanning_state_at(
        planning_data,
        initial_schedule,
        trigger_offset_min,
        **scenario_kwargs,
    )

    result = run_greedy_repair(
        data=planning_data,
        initial_schedule=initial_schedule,
        state=state,
        scenario_instance=instance,
        constraints=constraints,
        seed=SEED,
        bounded_repair_config=bounded_repair_config,
    )

    return state, instance, result


def _final_route(state, result):
    return tuple(state.completed_poi_ids) + tuple(result.selected_solution.route)


# ================================================================
# C1. Greedy Repair finds a feasible replacement
# ================================================================


def test_greedy_repair_finds_a_feasible_replacement(
    planning_data,
    initial_schedule,
):
    """POI 3 is closed, so the repair must substitute around it."""

    state, instance, result = _repair(
        planning_data,
        initial_schedule,
        TRIGGER_DURING_VISIT,
        closed_poi_ids=frozenset({3}),
    )

    assert result.status == "success"
    assert result.failure_reason is None
    assert result.selected_solution is not None

    solution = result.selected_solution

    assert solution.schedule.feasible is True
    assert 3 not in solution.route
    assert 4 in solution.route

    assert set(state.remaining_mandatory_ids) <= set(solution.route)
    assert set(solution.route) & set(state.unavailable_poi_ids) == set()

    # The repair is auditable.
    assert result.repair_record is not None
    assert result.repair_record.failure_status == "success"
    assert result.repair_record.locked_prefix == (1, 2)
    assert (
        result.repair_record.actual_modification_count
        == len(solution.route)
    )


# ================================================================
# C2. Greedy Repair reports an explicit failure
# ================================================================


def test_greedy_repair_reports_explicit_failure_when_no_repair_exists(
    planning_data,
    initial_schedule,
):
    """A 50x network delay makes the remaining mandatory POI unreachable."""

    state, _, result = _repair(
        planning_data,
        initial_schedule,
        TRIGGER_DURING_VISIT,
        global_travel_multiplier=50.0,
    )

    assert result.status == "infeasible"
    assert result.selected_solution is None
    assert result.failure_reason
    assert "no feasible post-trigger ordering" in result.failure_reason

    assert result.repair_record is not None
    assert result.repair_record.failure_status == "infeasible"
    assert result.repair_record.modified_suffix == ()
    assert result.repair_record.failure_reason


def test_greedy_repair_reports_explicit_failure_when_horizon_is_too_tight(
    planning_data,
    initial_schedule,
):
    """Tightening the horizon to 11:00 makes POI 4 unreachable."""

    state, _, result = _repair(
        planning_data,
        initial_schedule,
        TRIGGER_DURING_VISIT,
        constraints=PlanningConstraints(trip_end_min=11 * 60),
    )

    assert result.status == "infeasible"
    assert result.selected_solution is None
    assert result.repair_record is not None
    assert result.repair_record.failure_status == "infeasible"


# ================================================================
# C3. Optional POIs can all be removed
# ================================================================


def test_all_optional_pois_can_be_removed(planning_data, initial_schedule):
    """With every optional POI closed only the mandatory stop remains."""

    state, _, result = _repair(
        planning_data,
        initial_schedule,
        TRIGGER_DURING_VISIT,
        closed_poi_ids=frozenset({3, 5, 6}),
    )

    assert state.remaining_mandatory_ids == (4,)

    assert result.status == "success"
    assert result.selected_solution is not None
    assert result.selected_solution.route == (4,)
    assert result.selected_solution.schedule.feasible is True

    record = result.repair_record
    assert record is not None
    assert record.actual_modification_count == 1
    assert {
        decision.poi_id for decision in record.removed_activities
    } == {3, 5}


# ================================================================
# C4. Mandatory POIs are never silently dropped
# ================================================================


def test_mandatory_pois_are_never_silently_dropped(
    planning_data,
    initial_schedule,
):
    # (a) Every outstanding mandatory POI appears in a successful repair.
    state, _, result = _repair(
        planning_data,
        initial_schedule,
        TRIGGER_DURING_WAITING,
    )

    assert result.status == "success"
    assert state.remaining_mandatory_ids == (4,)
    assert 4 in result.selected_solution.route

    # (b) A closed mandatory POI is reported, not silently ignored.
    closed_state, _, closed_result = _repair(
        planning_data,
        initial_schedule,
        TRIGGER_DURING_VISIT,
        closed_poi_ids=frozenset({4}),
    )

    assert closed_state.unavailable_mandatory_ids == (4,)
    assert closed_state.remaining_mandatory_ids == ()
    assert closed_result.status == "success"

    # (c) If the mandatory POI is reachable in principle but not in time,
    # the outcome is an explicit failure rather than a shorter route.
    _, _, failed = _repair(
        planning_data,
        initial_schedule,
        TRIGGER_DURING_VISIT,
        global_travel_multiplier=50.0,
    )

    assert failed.status == "infeasible"
    assert failed.selected_solution is None


# ================================================================
# C5. Bounded repair never touches the locked prefix
# ================================================================


@pytest.mark.parametrize(
    "trigger_offset",
    [
        TRIGGER_DURING_FIRST_VISIT,
        TRIGGER_DURING_WALKING,
        TRIGGER_DURING_VISIT,
        TRIGGER_DURING_WAITING,
    ],
    ids=["at-start", "walking", "visiting", "waiting"],
)
def test_bounded_repair_never_modifies_the_locked_prefix(
    planning_data,
    initial_schedule,
    trigger_offset,
):
    state, _, result = _repair(
        planning_data,
        initial_schedule,
        trigger_offset,
    )

    assert result.status == "success"

    locked = tuple(state.completed_poi_ids)
    route = tuple(result.selected_solution.route)

    assert set(route).isdisjoint(set(locked))
    assert _final_route(state, result)[: len(locked)] == locked

    assert result.repair_record is not None
    assert result.repair_record.locked_prefix == locked


# ================================================================
# C6. The modification limit is enforced
# ================================================================


def test_modification_limit_is_enforced(planning_data, initial_schedule):
    state, instance = replanning_state_at(
        planning_data,
        initial_schedule,
        TRIGGER_DURING_VISIT,
    )

    for limit in (1, 2):
        result = run_greedy_repair(
            data=planning_data,
            initial_schedule=initial_schedule,
            state=state,
            scenario_instance=instance,
            seed=SEED,
            bounded_repair_config=BoundedRepairConfig(
                max_modifiable_suffix_activities=limit
            ),
        )

        assert result.status == "success"
        assert len(result.candidate_window) <= limit
        assert len(result.selected_solution.route) <= limit
        assert result.repair_record.bounded_repair_limit == limit

    # The shared candidate window respects the same cap.
    problem = prepare_adaptive_problem(
        data=planning_data,
        initial_schedule=initial_schedule,
        state=state,
        scenario_instance=instance,
        bounded_repair_config=BoundedRepairConfig(
            max_modifiable_suffix_activities=2
        ),
    )

    assert len(problem.candidate_window.all_candidate_ids) <= 2


@pytest.mark.parametrize("invalid_limit", [0, -1, MAXIMUM_CANDIDATE_COUNT + 1])
def test_modification_limit_rejects_invalid_values(invalid_limit):
    with pytest.raises(ValueError):
        BoundedRepairConfig(
            max_modifiable_suffix_activities=invalid_limit
        )


# ================================================================
# C7. Replacement POIs are not duplicated
# ================================================================


def test_replacement_pois_are_not_duplicated(planning_data, initial_schedule):
    state, _, result = _repair(
        planning_data,
        initial_schedule,
        TRIGGER_DURING_VISIT,
        closed_poi_ids=frozenset({3}),
    )

    route = tuple(result.selected_solution.route)

    assert len(route) == len(set(route)), "Repaired suffix repeats a POI."

    final = _final_route(state, result)
    assert len(final) == len(set(final)), "Final itinerary repeats a POI."

    assert result.repair_record.modified_suffix == route


# ================================================================
# C8. Stability metrics
# ================================================================


def _replanned_schedule(planning_data, route, start_time_min, start_poi_id=2):
    return decode_route(
        data=planning_data,
        visit_sequence=route,
        start_poi_id=start_poi_id,
        start_time_min=start_time_min,
        trip_end_min=17 * 60,
        day_code="WED",
        cost_budget_rm=160.0,
        start_poi_is_visit=False,
        required_mandatory_ids=(4,),
    )


def test_stability_metrics_for_unchanged_itinerary(
    planning_data,
    initial_schedule,
):
    metrics = compute_stability_metrics(
        original_suffix=(3, 5, 4),
        replanned_suffix=(3, 5, 4),
        locked_prefix=(1, 2),
        original_schedule=initial_schedule,
        replanned_schedule=initial_schedule,
    )

    assert metrics.preserved_poi_count == 3
    assert metrics.preserved_poi_percentage == 1.0
    assert metrics.removed_pois == ()
    assert metrics.added_pois == ()
    assert metrics.changed_activity_count == 0
    assert metrics.time_shift_count == 0
    assert metrics.order_deviation == 0
    assert metrics.edit_distance_score == 0.0
    assert metrics.locked_prefix_changed is False
    assert metrics.hard_stability_violation is False


def test_stability_metrics_for_removed_poi(planning_data, initial_schedule):
    metrics = compute_stability_metrics(
        original_suffix=(3, 5, 4),
        replanned_suffix=(5, 4),
        locked_prefix=(1, 2),
    )

    assert metrics.preserved_poi_count == 2
    assert metrics.preserved_poi_percentage == pytest.approx(2 / 3, abs=1e-6)
    assert metrics.removed_pois == (3,)
    assert metrics.added_pois == ()
    assert metrics.changed_activity_count == 1
    assert metrics.order_deviation == 0
    assert metrics.edit_distance_score == pytest.approx(1.0)


def test_stability_metrics_for_inserted_poi(planning_data, initial_schedule):
    metrics = compute_stability_metrics(
        original_suffix=(3, 5),
        replanned_suffix=(3, 5, 6),
        locked_prefix=(1, 2),
    )

    assert metrics.preserved_poi_count == 2
    assert metrics.preserved_poi_percentage == 1.0
    assert metrics.removed_pois == ()
    assert metrics.added_pois == (6,)
    assert metrics.changed_activity_count == 1
    assert metrics.edit_distance_score == pytest.approx(1.0)


def test_stability_metrics_for_shifted_activity(
    planning_data,
    initial_schedule,
):
    """POI 3 starts 30 minutes later; POIs 5 and 4 keep their times."""

    shifted_schedule = _replanned_schedule(
        planning_data,
        (3, 5, 4),
        start_time_min=11 * 60 + 10,
    )

    assert shifted_schedule.feasible, shifted_schedule.failure_reason

    metrics = compute_stability_metrics(
        original_suffix=(3, 5, 4),
        replanned_suffix=(3, 5, 4),
        locked_prefix=(1, 2),
        original_schedule=initial_schedule,
        replanned_schedule=shifted_schedule,
    )

    assert metrics.preserved_poi_count == 3
    assert metrics.removed_pois == ()
    assert metrics.added_pois == ()
    assert metrics.time_shift_count == 1
    assert metrics.shifted_pois == (3,)
    assert metrics.total_time_shift_min == pytest.approx(30.0)
    assert metrics.changed_activity_count == 1
    assert metrics.order_deviation == 0

    # The time-shift weight is applied per shifted activity, not per
    # shifted minute (see ``stability_metrics``).
    assert metrics.edit_distance_score == pytest.approx(0.25)


def test_stability_metrics_for_reordered_suffix(
    planning_data,
    initial_schedule,
):
    reordered_schedule = _replanned_schedule(
        planning_data,
        (5, 3, 4),
        start_time_min=10 * 60 + 40,
    )

    assert reordered_schedule.feasible, reordered_schedule.failure_reason

    metrics = compute_stability_metrics(
        original_suffix=(3, 5, 4),
        replanned_suffix=(5, 3, 4),
        locked_prefix=(1, 2),
        original_schedule=initial_schedule,
        replanned_schedule=reordered_schedule,
    )

    assert metrics.preserved_poi_count == 3
    assert metrics.removed_pois == ()
    assert metrics.added_pois == ()

    # One discordant pair: (3, 5) became (5, 3).
    assert metrics.order_deviation == 1

    # Both POI 3 and POI 4 also moved in time.
    assert metrics.time_shift_count == 2

    expected = (
        0.5 * metrics.order_deviation
        + 0.25 * metrics.time_shift_count
    )

    assert metrics.edit_distance_score == pytest.approx(expected)


def test_stability_metrics_report_a_locked_prefix_violation(
    planning_data,
    initial_schedule,
):
    """A repair that re-visits POI 2 is a hard stability violation."""

    metrics = compute_stability_metrics(
        original_suffix=(3, 5, 4),
        replanned_suffix=(2, 3, 5, 4),
        locked_prefix=(1, 2),
        final_route=(1, 2, 2, 3, 5, 4),
    )

    assert metrics.locked_prefix_changed is True
    assert metrics.hard_stability_violation is True
    assert metrics.prefix_reappearing_pois == (2,)
    assert metrics.violation_reasons

    # The offending POI is reported, never silently removed.
    assert 2 in metrics.added_pois

    # A violation dominates any amount of soft change.
    assert metrics.edit_distance_score >= 1000.0
