"""Category B - disruption simulation and replanning state.

The trigger-state simulator is the only component that decides what the
traveller has already committed to.  These tests pin each commit point
(before the start, walking, waiting, visiting) and each disruption
channel (closure, network delay, zone congestion, thermal multiplier) on
the synthetic timeline documented in ``conftest``.
"""

from __future__ import annotations

import pytest

from adaptive_ga_core import (
    calculate_locked_travel_minutes,
    decode_replanned_route,
    prepare_adaptive_problem,
)
from conftest import (
    MANDATORY_POI_IDS,
    TRIGGER_BEFORE_START,
    TRIGGER_DURING_FIRST_VISIT,
    TRIGGER_DURING_VISIT,
    TRIGGER_DURING_WAITING,
    TRIGGER_DURING_WALKING,
    replanning_state_at,
)
from planning_core import TravelAdjustments
from scenario_core import get_travel_adjustments


# ================================================================
# B1. Disruption before the first activity
# ================================================================


def test_disruption_before_first_activity_commits_nothing(
    planning_data,
    initial_schedule,
):
    """A trigger before 09:00 leaves the whole itinerary modifiable."""

    state, _ = replanning_state_at(
        planning_data,
        initial_schedule,
        TRIGGER_BEFORE_START,
    )

    assert state.trigger_absolute_min == 8 * 60 + 30
    assert state.committed_component == "none"
    assert state.completed_poi_ids == ()
    assert state.route_finished_before_trigger is False

    # The traveller is still standing at the origin and has not visited
    # it, so the origin remains eligible for the repaired route.
    assert state.current_poi_id == 1
    assert state.current_poi_requires_visit is True

    assert state.replan_start_min == state.trigger_absolute_min
    assert state.original_remaining_poi_ids == (1, 2, 3, 5, 4)
    assert state.remaining_mandatory_ids == MANDATORY_POI_IDS

    assert state.cost_spent_rm == 0.0
    assert state.utility_achieved == 0.0
    assert calculate_locked_travel_minutes(initial_schedule, state) == 0.0


def test_disruption_at_trip_start_locks_the_first_visit(
    planning_data,
    initial_schedule,
):
    """A trigger at exactly 09:00 catches POI 1 mid-visit."""

    state, _ = replanning_state_at(
        planning_data,
        initial_schedule,
        TRIGGER_DURING_FIRST_VISIT,
    )

    assert state.committed_component == "visit"
    assert state.completed_poi_ids == (1,)
    assert state.current_poi_id == 1
    assert state.current_poi_requires_visit is False

    # The in-progress visit is completed, so replanning starts at its
    # scheduled end rather than at the trigger.
    assert state.replan_start_min == 9 * 60 + 30
    assert state.original_remaining_poi_ids == (2, 3, 5, 4)


# ================================================================
# B2. Disruption during walking
# ================================================================


def test_disruption_during_walking_completes_the_leg(
    planning_data,
    initial_schedule,
):
    """At 09:35 the traveller is between POI 1 and POI 2."""

    state, _ = replanning_state_at(
        planning_data,
        initial_schedule,
        TRIGGER_DURING_WALKING,
    )

    assert state.trigger_absolute_min == 9 * 60 + 35
    assert state.committed_component == "walking_leg"

    # The leg is committed, so the traveller arrives at POI 2 at the
    # originally scheduled time and replanning starts there.
    assert state.current_poi_id == 2
    assert state.current_poi_requires_visit is True
    assert state.replan_start_min == 9 * 60 + 40

    # POI 1 is finished; POI 2 has not been visited yet.
    assert state.completed_poi_ids == (1,)
    assert state.original_remaining_poi_ids == (2, 3, 5, 4)
    assert state.remaining_mandatory_ids == (2, 4)


# ================================================================
# B3. Disruption during waiting
# ================================================================


def test_disruption_during_waiting_starts_at_the_trigger(
    planning_data,
    initial_schedule,
):
    """At 12:30 the traveller has arrived at POI 5 but it opens at 13:00."""

    state, _ = replanning_state_at(
        planning_data,
        initial_schedule,
        TRIGGER_DURING_WAITING,
    )

    assert state.committed_component == "waiting"
    assert state.current_poi_id == 5
    assert state.current_poi_requires_visit is True

    # Waiting is not committed time: replanning starts at the trigger.
    assert state.replan_start_min == state.trigger_absolute_min

    assert state.completed_poi_ids == (1, 2, 3)
    assert state.original_remaining_poi_ids == (5, 4)
    assert state.remaining_mandatory_ids == (4,)


# ================================================================
# B4. Disruption during a visit
# ================================================================


def test_disruption_during_visit_locks_the_visit(
    planning_data,
    initial_schedule,
):
    """At 10:00 the traveller is inside POI 2 (09:40-10:40)."""

    state, _ = replanning_state_at(
        planning_data,
        initial_schedule,
        TRIGGER_DURING_VISIT,
    )

    assert state.committed_component == "visit"
    assert state.completed_poi_ids == (1, 2)
    assert state.current_poi_id == 2
    assert state.current_poi_requires_visit is False

    assert state.replan_start_min == 10 * 60 + 40
    assert state.original_remaining_poi_ids == (3, 5, 4)
    assert state.remaining_mandatory_ids == (4,)

    # Costs and utility of the locked visits are carried forward.
    assert state.cost_spent_rm == pytest.approx(10.0)
    assert state.utility_achieved == pytest.approx(15.0)


# ================================================================
# B5. The completed prefix stays locked
# ================================================================


def test_completed_prefix_remains_locked(
    planning_data,
    initial_schedule,
):
    """No planner input may re-open a completed visit."""

    state, instance = replanning_state_at(
        planning_data,
        initial_schedule,
        TRIGGER_DURING_VISIT,
    )

    problem = prepare_adaptive_problem(
        data=planning_data,
        initial_schedule=initial_schedule,
        state=state,
        scenario_instance=instance,
    )

    assert problem.locked_completed_count == 2
    assert problem.locked_utility == pytest.approx(15.0)

    window = set(problem.candidate_window.all_candidate_ids)

    assert window.isdisjoint({1, 2}), (
        "A completed POI must never re-enter the candidate window."
    )
    assert set(problem.mandatory_backbone).isdisjoint({1, 2})


# ================================================================
# B6. In-transit leg semantics
# ================================================================


def test_in_transit_leg_follows_documented_semantics(
    planning_data,
    initial_schedule,
):
    """A leg already under way is completed at its pre-trigger duration
    and counted as locked travel, but its destination is not visited."""

    state, _ = replanning_state_at(
        planning_data,
        initial_schedule,
        TRIGGER_DURING_WALKING,
    )

    # Replanning begins at the destination arrival time, not at the
    # trigger, because the traveller cannot teleport back.
    assert state.replan_start_min == 9 * 60 + 40
    assert state.replan_start_min > state.trigger_absolute_min

    # The 10-minute leg is locked travel.
    assert calculate_locked_travel_minutes(
        initial_schedule, state
    ) == pytest.approx(10.0)

    # The destination is unvisited, so it stays in the mandatory set.
    assert 2 in state.remaining_mandatory_ids
    assert 2 not in state.completed_poi_ids


# ================================================================
# B7. Closed POI excluded from the replanned suffix
# ================================================================


def test_closed_poi_is_excluded_from_the_replanned_suffix(
    planning_data,
    initial_schedule,
):
    state, instance = replanning_state_at(
        planning_data,
        initial_schedule,
        TRIGGER_DURING_VISIT,
        scenario_id="TX_closure",
        closed_poi_ids=frozenset({3}),
    )

    assert state.unavailable_poi_ids == (3,)
    assert 3 not in state.remaining_mandatory_ids

    problem = prepare_adaptive_problem(
        data=planning_data,
        initial_schedule=initial_schedule,
        state=state,
        scenario_instance=instance,
    )

    assert 3 not in problem.candidate_window.all_candidate_ids

    # A route that still contains the closed POI must be rejected.
    rejected = decode_replanned_route(
        data=planning_data,
        state=state,
        scenario_instance=instance,
        route=(3, 4),
    )

    assert rejected.feasible is False
    assert "is unavailable" in (rejected.failure_reason or "")


# ================================================================
# B8. Delayed travel time is applied to the repaired suffix
# ================================================================


def test_delayed_travel_time_is_applied_to_the_repaired_suffix(
    planning_data,
    initial_schedule,
):
    state, instance = replanning_state_at(
        planning_data,
        initial_schedule,
        TRIGGER_DURING_VISIT,
        scenario_id="TX_delay",
        global_travel_multiplier=1.5,
    )

    assert state.global_travel_multiplier == pytest.approx(1.5)

    schedule = decode_replanned_route(
        data=planning_data,
        state=state,
        scenario_instance=instance,
        route=(3, 4),
    )

    assert schedule.feasible, schedule.failure_reason

    by_poi = {entry.poi_id: entry for entry in schedule.entries}

    # 2 -> 3 base 20 min, 3 -> 4 base 25 min.
    assert by_poi[3].travel_min == pytest.approx(30.0)
    assert by_poi[4].travel_min == pytest.approx(37.5)

    # The locked prefix keeps its original, undelayed travel.
    assert calculate_locked_travel_minutes(
        initial_schedule, state
    ) == pytest.approx(10.0)


# ================================================================
# B9. Zone congestion applies only to the correct links
# ================================================================


def test_zone_congestion_applies_only_to_zone_links(
    planning_data,
    initial_schedule,
):
    """Zone A is congested; only A -> A links may change."""

    _, instance = replanning_state_at(
        planning_data,
        initial_schedule,
        TRIGGER_DURING_VISIT,
        scenario_id="TX_congestion",
        congested_zone="A",
        congestion_multiplier=1.3,
    )

    adjustments = get_travel_adjustments(
        planning_data, instance.scenario
    )

    assert isinstance(adjustments, TravelAdjustments)
    assert adjustments.global_multiplier == pytest.approx(1.0)

    # A -> A links are congested.
    assert adjustments.apply(1, 2, 10.0) == pytest.approx(13.0)
    assert adjustments.apply(2, 4, 15.0) == pytest.approx(19.5)

    # B -> B links are untouched.
    assert adjustments.apply(3, 5, 10.0) == pytest.approx(10.0)
    assert adjustments.apply(5, 6, 10.0) == pytest.approx(10.0)

    # Cross-zone links are untouched in either direction.
    assert adjustments.apply(2, 3, 20.0) == pytest.approx(20.0)
    assert adjustments.apply(4, 5, 30.0) == pytest.approx(30.0)

    # Exactly the nine ordered A -> A pairs are congested.
    assert len(adjustments.link_multipliers) == 3 * 2


# ================================================================
# B10. Combined disruption applies all intended changes
# ================================================================


def test_combined_disruption_applies_all_intended_changes(
    planning_data,
    initial_schedule,
):
    """Thermal load, network delay and a closure at the same time."""

    state, instance = replanning_state_at(
        planning_data,
        initial_schedule,
        TRIGGER_DURING_VISIT,
        scenario_id="TX_combined",
        thermal_multiplier=1.8,
        global_travel_multiplier=1.3,
        closed_poi_ids=frozenset({3}),
    )

    assert state.thermal_multiplier == pytest.approx(1.8)
    assert state.global_travel_multiplier == pytest.approx(1.3)
    assert state.unavailable_poi_ids == (3,)
    assert state.completed_poi_ids == (1, 2)
    assert state.remaining_mandatory_ids == (4,)

    schedule = decode_replanned_route(
        data=planning_data,
        state=state,
        scenario_instance=instance,
        route=(4,),
    )

    assert schedule.feasible, schedule.failure_reason
    assert schedule.entries[0].travel_min == pytest.approx(19.5)
