"""Category A - basic feasibility of the shared route decoder.

Every planner ultimately calls ``planning_core.decode_route``.  If that
decoder ever accepts an infeasible itinerary, every downstream metric
becomes meaningless.  These tests pin each hard constraint individually
on the six-POI synthetic instance so a failure names the exact rule that
broke.
"""

from __future__ import annotations

import pytest

from planning_core import decode_route


START = 9 * 60
END = 17 * 60


def _decode(planning_data, sequence, **overrides):
    """Decode a route on the synthetic instance with explicit defaults."""

    kwargs = {
        "start_poi_id": 1,
        "start_time_min": START,
        "trip_end_min": END,
        "day_code": "WED",
        "cost_budget_rm": 160.0,
        "start_poi_is_visit": True,
        "required_mandatory_ids": (1, 2, 4),
    }
    kwargs.update(overrides)

    return decode_route(
        data=planning_data,
        visit_sequence=sequence,
        **kwargs,
    )


# ================================================================
# A1. A valid itinerary
# ================================================================


def test_valid_itinerary_is_feasible(initial_schedule):
    assert initial_schedule.feasible is True
    assert initial_schedule.failure_reason is None

    assert initial_schedule.completed_sequence == (1, 2, 3, 5, 4)
    assert initial_schedule.missing_mandatory_ids == ()
    assert initial_schedule.unavailable_mandatory_ids == ()

    assert initial_schedule.finish_min == 15 * 60
    assert initial_schedule.total_cost_rm == pytest.approx(35.0)


# ================================================================
# A2. Late arrival - arrival after opening means no waiting
# ================================================================


def test_late_arrival_starts_the_visit_immediately(planning_data):
    """Arriving after the opening time must not invent waiting time."""

    schedule = _decode(
        planning_data,
        (1, 2),
        start_time_min=9 * 60 + 45,
        required_mandatory_ids=(1, 2),
    )

    assert schedule.feasible, schedule.failure_reason

    first, second = schedule.entries

    assert first.visit_end_min == 10 * 60 + 15
    assert second.arrival_min == 10 * 60 + 25
    assert second.waiting_min == 0.0
    assert second.visit_start_min == second.arrival_min
    assert second.visit_end_min == 11 * 60 + 25


# ================================================================
# A3. POI closed before arrival
# ================================================================


def test_poi_closed_before_arrival_is_infeasible(planning_data):
    """POI 2 closes at 12:00; arriving at 13:40 must be rejected."""

    schedule = _decode(
        planning_data,
        (1, 2),
        start_time_min=13 * 60,
        required_mandatory_ids=(1, 2),
    )

    assert schedule.feasible is False
    assert schedule.failure_reason is not None
    assert "cannot be started before closing" in schedule.failure_reason
    assert schedule.completed_sequence == (1,)


# ================================================================
# A4. Visit ending after closing time
# ================================================================


def test_visit_ending_after_closing_time_is_infeasible(planning_data):
    """Starting at 11:40 is legal but the 60-minute visit overruns 12:00."""

    schedule = _decode(
        planning_data,
        (1, 2),
        start_time_min=11 * 60,
        required_mandatory_ids=(1, 2),
    )

    assert schedule.feasible is False
    assert schedule.failure_reason is not None
    assert "would finish after closing" in schedule.failure_reason


# ================================================================
# A5. Visit ending after the daily horizon
# ================================================================


def test_visit_ending_after_daily_horizon_is_infeasible(planning_data):
    """POI 4 is open until 18:00, but the trip ends at 17:00."""

    schedule = _decode(
        planning_data,
        (1, 4),
        start_time_min=15 * 60 + 30,
        required_mandatory_ids=(1, 4),
    )

    assert schedule.feasible is False
    assert schedule.failure_reason is not None
    assert "17:00 trip limit" in schedule.failure_reason


# ================================================================
# A6. Activities never overlap
# ================================================================


def test_schedule_activities_never_overlap(initial_schedule):
    """Each leg starts when the previous visit ends, so no two
    activities occupy the same minute."""

    entries = initial_schedule.entries
    assert len(entries) == 5

    for previous, current in zip(entries, entries[1:]):
        assert current.travel_min > 0
        assert current.arrival_min == pytest.approx(
            previous.visit_end_min + current.travel_min
        )
        assert current.visit_start_min >= current.arrival_min
        assert previous.visit_end_min <= current.visit_start_min

    for entry in entries:
        assert entry.visit_end_min > entry.visit_start_min
        assert entry.travel_min >= 0.0

    # The timeline is strictly monotone across the whole itinerary.
    visit_ends = [entry.visit_end_min for entry in entries]
    assert visit_ends == sorted(visit_ends)


# ================================================================
# A7. Duplicate POI
# ================================================================


def test_duplicate_poi_is_rejected(planning_data):
    schedule = _decode(planning_data, (1, 2, 3, 2, 4))

    assert schedule.feasible is False
    assert schedule.failure_reason is not None
    assert "duplicate" in schedule.failure_reason.lower()


# ================================================================
# A8. Missing route
# ================================================================


def test_missing_route_is_rejected(planning_data):
    """An initial itinerary must contain at least the starting POI."""

    schedule = _decode(planning_data, ())

    assert schedule.feasible is False
    assert schedule.failure_reason is not None
    assert "must contain the starting POI" in schedule.failure_reason
    assert schedule.entries == ()


# ================================================================
# A9. Unavailable POI
# ================================================================


def test_unavailable_poi_is_rejected(planning_data):
    schedule = _decode(
        planning_data,
        (1, 2, 3, 5, 4),
        unavailable_poi_ids=(3,),
    )

    assert schedule.feasible is False
    assert schedule.failure_reason is not None
    assert "is unavailable" in schedule.failure_reason
    assert schedule.completed_sequence == (1, 2)


# ================================================================
# A10. Budget exceeded
# ================================================================


def test_budget_exceeded_is_rejected(planning_data):
    """The itinerary costs RM35; a RM15 budget must fail."""

    schedule = _decode(
        planning_data,
        (1, 2, 3, 5, 4),
        cost_budget_rm=15.0,
    )

    assert schedule.feasible is False
    assert schedule.failure_reason is not None
    assert "exceed the cost budget" in schedule.failure_reason


# ================================================================
# A11. Missing mandatory POI
# ================================================================


def test_missing_mandatory_poi_is_rejected(planning_data):
    schedule = _decode(
        planning_data,
        (1, 3),
        required_mandatory_ids=(1, 2, 4),
    )

    assert schedule.feasible is False
    assert schedule.failure_reason is not None
    assert "does not contain all available mandatory POIs" in (
        schedule.failure_reason
    )
    assert schedule.missing_mandatory_ids == (2, 4)


# ================================================================
# A12. Unknown POI ID
# ================================================================


def test_unknown_poi_id_is_rejected(planning_data):
    schedule = _decode(planning_data, (1, 99))

    assert schedule.feasible is False
    assert schedule.failure_reason is not None
    assert "Unknown POI IDs" in schedule.failure_reason


# ================================================================
# Additional structural guards
# ================================================================


def test_poi_closed_on_the_selected_day_is_rejected(planning_data):
    """POI 6 is closed on Wednesday."""

    schedule = _decode(planning_data, (1, 6))

    assert schedule.feasible is False
    assert schedule.failure_reason is not None
    assert "is closed on WED" in schedule.failure_reason


def test_initial_route_must_start_with_the_start_poi(planning_data):
    schedule = _decode(planning_data, (2, 1, 4))

    assert schedule.feasible is False
    assert schedule.failure_reason is not None
    assert "first route ID must equal start_poi_id" in (
        schedule.failure_reason
    )


def test_unavailable_mandatory_poi_can_be_reported_explicitly(planning_data):
    """A closed mandatory POI is reported, not silently dropped."""

    schedule = _decode(
        planning_data,
        (1, 3),
        required_mandatory_ids=(1, 2, 4),
        unavailable_poi_ids=(2, 4),
    )

    assert schedule.unavailable_mandatory_ids == (2, 4)
    assert schedule.missing_mandatory_ids == ()
    assert schedule.feasible is True
