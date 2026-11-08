"""
Tests for the deterministic itinerary-stability metrics.

Covered scenarios (the eleven required cases plus contract checks):

1.  no changes after replanning
2.  one removed POI
3.  one replacement POI
4.  a shifted activity time
5.  a reordered suffix
6.  an unchanged locked prefix
7.  a locked-prefix violation
8.  an empty suffix
9.  an infeasible replanning result
10. duplicate POIs
11. identical original and replanned itineraries

Plus:
- weight-configuration validation;
- weight sensitivity (a different configuration changes the score);
- dominance of the locked-prefix penalty over every soft change;
- determinism of repeated calls;
- integration with ``benchmark_common.build_metric_record``;
- a real-pipeline check over all 18 instances that a conforming repair
  never reports a locked-prefix violation.
"""

from __future__ import annotations

from dataclasses import dataclass

from adaptive_ga_core import MAXIMUM_CANDIDATE_COUNT
from benchmark_common import (
    build_initial_schedules,
    build_metric_record,
    build_scenario_lookup,
    load_initial_routes,
)
from greedy_repair import run_greedy_repair
from planning_core import PlanningConstraints, load_planning_data
from scenario_core import simulate_replanning_state
from stability_metrics import (
    DEFAULT_STABILITY_WEIGHTS,
    StabilityWeightConfig,
    compute_stability_metrics,
    maximum_soft_edit_distance,
)


POI_FILE = "poi_data_calculated_final.csv"
MATRIX_FILE = "walking_time_matrix_final.csv"

TEST_SEED = 1001


# ================================================================
# Minimal schedule test doubles
#
# The metrics module only reads ``.entries``, ``.poi_id`` and
# ``.visit_start_min``, so a small stub keeps these tests independent of
# the full decoder.  One integration test below uses the real
# ``ScheduleResult`` produced by ``decode_route``.
# ================================================================

@dataclass(frozen=True)
class StubEntry:
    poi_id: int
    visit_start_min: float


@dataclass(frozen=True)
class StubSchedule:
    entries: tuple[StubEntry, ...]


def build_stub_schedule(start_by_poi: dict[int, float]) -> StubSchedule:
    """Build a schedule stub from ``{poi_id: visit_start_min}``."""

    return StubSchedule(
        entries=tuple(
            StubEntry(poi_id=poi_id, visit_start_min=start)
            for poi_id, start in start_by_poi.items()
        )
    )


# ================================================================
# 1. No changes after replanning
# ================================================================

def test_no_changes_after_replanning() -> None:
    metrics = compute_stability_metrics(
        original_suffix=[4, 13, 33, 36],
        replanned_suffix=[4, 13, 33, 36],
        locked_prefix=[1, 2],
    )

    assert metrics.preserved_poi_count == 4
    assert metrics.preserved_poi_percentage == 1.0
    assert metrics.removed_pois == ()
    assert metrics.added_pois == ()
    assert metrics.changed_activity_count == 0
    assert metrics.order_deviation == 0
    assert metrics.edit_distance_score == 0.0
    assert metrics.locked_prefix_changed is False
    assert metrics.hard_stability_violation is False

    print("PASS: no_changes_after_replanning")


# ================================================================
# 2. One removed POI
# ================================================================

def test_one_removed_poi() -> None:
    metrics = compute_stability_metrics(
        original_suffix=[4, 13, 33, 36],
        replanned_suffix=[4, 13, 36],
        locked_prefix=[1, 2],
    )

    assert metrics.preserved_poi_count == 3
    assert metrics.preserved_poi_percentage == 0.75
    assert metrics.removed_pois == (33,)
    assert metrics.added_pois == ()
    assert metrics.changed_activity_count == 1
    assert metrics.order_deviation == 0
    # One removal at weight 1.0 and nothing else.
    assert metrics.edit_distance_score == 1.0
    assert (
        metrics.edit_distance_components["removed"] == 1.0
    )

    print("PASS: one_removed_poi")


# ================================================================
# 3. One replacement POI
# ================================================================

def test_one_replacement_poi() -> None:
    metrics = compute_stability_metrics(
        original_suffix=[4, 13, 33, 36],
        replanned_suffix=[4, 13, 34, 36],
        locked_prefix=[1, 2],
    )

    assert metrics.preserved_poi_count == 3
    assert metrics.removed_pois == (33,)
    assert metrics.added_pois == (34,)
    assert metrics.changed_activity_count == 2
    # One removal plus one insertion, both at weight 1.0.
    assert metrics.edit_distance_score == 2.0

    print("PASS: one_replacement_poi")


# ================================================================
# 4. A shifted activity time
# ================================================================

def test_shifted_activity_time() -> None:
    original_schedule = build_stub_schedule(
        {4: 600.0, 13: 650.0, 33: 700.0}
    )
    replanned_schedule = build_stub_schedule(
        {4: 600.0, 13: 670.0, 33: 700.0}
    )

    metrics = compute_stability_metrics(
        original_suffix=[4, 13, 33],
        replanned_suffix=[4, 13, 33],
        locked_prefix=[1, 2],
        original_schedule=original_schedule,
        replanned_schedule=replanned_schedule,
    )

    assert metrics.timing_measurement_available is True
    assert metrics.time_shift_count == 1
    assert metrics.shifted_pois == (13,)
    assert metrics.total_time_shift_min == 20.0
    assert metrics.changed_activity_count == 1
    # No membership change; one time shift at weight 0.25.
    assert metrics.edit_distance_score == 0.25

    # A shift below the tolerance must not be counted.
    tiny_schedule = build_stub_schedule(
        {4: 600.0, 13: 650.0 + 1e-9, 33: 700.0}
    )

    tiny = compute_stability_metrics(
        original_suffix=[4, 13, 33],
        replanned_suffix=[4, 13, 33],
        locked_prefix=[1, 2],
        original_schedule=original_schedule,
        replanned_schedule=tiny_schedule,
    )

    assert tiny.time_shift_count == 0
    assert tiny.edit_distance_score == 0.0

    print("PASS: shifted_activity_time")


# ================================================================
# 5. A reordered suffix
# ================================================================

def test_reordered_suffix() -> None:
    metrics = compute_stability_metrics(
        original_suffix=[4, 13, 33],
        replanned_suffix=[33, 4, 13],
        locked_prefix=[1, 2],
    )

    assert metrics.preserved_poi_count == 3
    assert metrics.removed_pois == ()
    assert metrics.added_pois == ()
    # Discordant pairs: (4, 33) and (13, 33).
    assert metrics.order_deviation == 2
    # Two discordant pairs at weight 0.5.
    assert metrics.edit_distance_score == 1.0
    assert (
        metrics.edit_distance_components["order"] == 1.0
    )

    # A complete reversal is the maximum possible deviation.
    reversal = compute_stability_metrics(
        original_suffix=[4, 13, 33],
        replanned_suffix=[33, 13, 4],
        locked_prefix=[1, 2],
    )

    assert reversal.order_deviation == 3

    # Inserting a new POI without disturbing relative order is not a
    # reordering of the retained POIs.
    inserted = compute_stability_metrics(
        original_suffix=[4, 13, 33],
        replanned_suffix=[4, 34, 13, 33],
        locked_prefix=[1, 2],
    )

    assert inserted.order_deviation == 0
    assert inserted.added_pois == (34,)

    print("PASS: reordered_suffix")


# ================================================================
# 6. An unchanged locked prefix
# ================================================================

def test_unchanged_locked_prefix() -> None:
    metrics = compute_stability_metrics(
        original_suffix=[4, 13, 33],
        replanned_suffix=[4, 33],
        locked_prefix=[1, 2, 12],
        final_route=[1, 2, 12, 4, 33],
    )

    assert metrics.locked_prefix_changed is False
    assert metrics.hard_stability_violation is False
    assert metrics.violation_reasons == ()
    assert metrics.prefix_reappearing_pois == ()
    # Only the removal is charged; no prefix penalty.
    assert metrics.edit_distance_score == 1.0
    assert (
        metrics.edit_distance_components[
            "locked_prefix_violation"
        ]
        == 0.0
    )

    print("PASS: unchanged_locked_prefix")


# ================================================================
# 7. A locked-prefix violation
# ================================================================

def test_locked_prefix_violation() -> None:
    # (a) A completed prefix POI reappears in the repaired suffix.
    reappearing = compute_stability_metrics(
        original_suffix=[4, 13],
        replanned_suffix=[2, 4, 13],
        locked_prefix=[1, 2],
    )

    assert reappearing.locked_prefix_changed is True
    assert reappearing.hard_stability_violation is True
    assert reappearing.prefix_reappearing_pois == (2,)
    assert reappearing.violation_reasons
    assert (
        reappearing.edit_distance_components[
            "locked_prefix_violation"
        ]
        == DEFAULT_STABILITY_WEIGHTS.locked_prefix_violation_weight
    )
    assert reappearing.edit_distance_score >= 1000.0

    # The violation is reported, never silently corrected: the offending
    # POI still appears in ``added_pois``.
    assert 2 in reappearing.added_pois

    # (b) A supplied final route whose prefix differs is also a
    # violation.
    mismatched = compute_stability_metrics(
        original_suffix=[4, 13],
        replanned_suffix=[4, 13],
        locked_prefix=[1, 2],
        final_route=[1, 99, 4, 13],
    )

    assert mismatched.locked_prefix_changed is True
    assert any(
        "does not begin with the locked prefix" in reason
        for reason in mismatched.violation_reasons
    )

    print("PASS: locked_prefix_violation")


# ================================================================
# 8. An empty suffix
# ================================================================

def test_empty_suffix() -> None:
    # Nothing was planned and nothing was repaired.
    empty = compute_stability_metrics(
        original_suffix=[],
        replanned_suffix=[],
        locked_prefix=[1, 2, 3],
    )

    assert empty.preserved_poi_count == 0
    assert empty.preserved_poi_percentage is None
    assert empty.removed_pois == ()
    assert empty.added_pois == ()
    assert empty.modified_suffix_activity_count == 0
    assert empty.order_deviation == 0
    assert empty.edit_distance_score == 0.0
    assert empty.locked_prefix_changed is False

    # Everything was planned and nothing was repaired: all removed.
    dropped = compute_stability_metrics(
        original_suffix=[4, 13, 33],
        replanned_suffix=[],
        locked_prefix=[1, 2],
    )

    assert dropped.preserved_poi_count == 0
    assert dropped.removed_pois == (4, 13, 33)
    assert dropped.modified_suffix_activity_count == 0
    assert dropped.edit_distance_score == 3.0

    print("PASS: empty_suffix")


# ================================================================
# 9. An infeasible replanning result
# ================================================================

def test_infeasible_replanning_result() -> None:
    original_schedule = build_stub_schedule(
        {4: 600.0, 13: 650.0, 33: 700.0}
    )

    metrics = compute_stability_metrics(
        original_suffix=[4, 13, 33],
        replanned_suffix=[],
        locked_prefix=[1, 2],
        original_schedule=original_schedule,
        replanned_schedule=None,
        replanning_feasible=False,
    )

    assert metrics.replanning_feasible is False
    assert metrics.timing_measurement_available is False
    # Timing is not guessed when there is no schedule.
    assert metrics.time_shift_count is None
    assert metrics.total_time_shift_min is None
    assert metrics.shifted_pois == ()
    # Membership changes are still counted honestly.
    assert metrics.removed_pois == (4, 13, 33)
    assert metrics.changed_activity_count == 3
    assert metrics.edit_distance_score == 3.0
    # An infeasible repair is not itself a prefix violation.
    assert metrics.hard_stability_violation is False

    print("PASS: infeasible_replanning_result")


# ================================================================
# 10. Duplicate POIs
# ================================================================

def test_duplicate_pois() -> None:
    metrics = compute_stability_metrics(
        original_suffix=[4, 4, 13, 33],
        replanned_suffix=[4, 13, 33, 33],
        locked_prefix=[1, 1, 2],
    )

    # Duplicates are reported, never counted twice.
    assert metrics.duplicate_original_pois == (4,)
    assert metrics.duplicate_replanned_pois == (33,)
    assert metrics.duplicate_locked_prefix_pois == (1,)

    assert metrics.preserved_poi_count == 3
    assert metrics.removed_pois == ()
    assert metrics.added_pois == ()
    assert metrics.order_deviation == 0
    assert metrics.edit_distance_score == 0.0

    # Duplicates cannot inflate the suffix size either.
    assert metrics.modified_suffix_activity_count == 3

    print("PASS: duplicate_pois")


# ================================================================
# 11. Identical original and replanned itineraries
# ================================================================

def test_identical_itineraries() -> None:
    schedule = build_stub_schedule(
        {4: 600.0, 13: 650.0, 33: 700.0, 36: 750.0}
    )

    metrics = compute_stability_metrics(
        original_suffix=[4, 13, 33, 36],
        replanned_suffix=[4, 13, 33, 36],
        locked_prefix=[1, 2],
        original_schedule=schedule,
        replanned_schedule=schedule,
        final_route=[1, 2, 4, 13, 33, 36],
    )

    assert metrics.preserved_poi_count == 4
    assert metrics.preserved_poi_percentage == 1.0
    assert metrics.removed_pois == ()
    assert metrics.added_pois == ()
    assert metrics.changed_activity_count == 0
    assert metrics.time_shift_count == 0
    assert metrics.total_time_shift_min == 0.0
    assert metrics.order_deviation == 0
    assert metrics.edit_distance_score == 0.0
    assert metrics.hard_stability_violation is False

    print("PASS: identical_itineraries")


# ================================================================
# Weight configuration
# ================================================================

def test_weight_config_validation() -> None:
    try:
        StabilityWeightConfig(removed_poi_weight=-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "A negative weight must be rejected."
        )

    try:
        StabilityWeightConfig(
            locked_prefix_violation_weight=0.0
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "A zero violation weight must be rejected."
        )

    try:
        StabilityWeightConfig(removed_poi_weight="heavy")
    except TypeError:
        pass
    else:
        raise AssertionError(
            "A non-numeric weight must be rejected."
        )

    default_payload = DEFAULT_STABILITY_WEIGHTS.to_dict()

    assert default_payload["removed_poi_weight"] == 1.0
    assert default_payload["inserted_poi_weight"] == 1.0
    assert default_payload["order_change_weight"] == 0.5
    assert default_payload["time_shift_weight"] == 0.25
    assert (
        default_payload["locked_prefix_violation_weight"]
        == 1000.0
    )

    print("PASS: weight_config_validation")


def test_weight_sensitivity() -> None:
    original = [4, 13, 33, 36]
    replanned = [4, 13, 34]

    default = compute_stability_metrics(
        original_suffix=original,
        replanned_suffix=replanned,
        locked_prefix=[1, 2],
    )

    doubled = compute_stability_metrics(
        original_suffix=original,
        replanned_suffix=replanned,
        locked_prefix=[1, 2],
        weights=StabilityWeightConfig(
            removed_poi_weight=2.0,
            inserted_poi_weight=1.0,
            order_change_weight=0.5,
            time_shift_weight=0.25,
        ),
    )

    # Default: 2 removals + 1 insertion = 3.0.
    assert default.edit_distance_score == 3.0
    # Doubled removal weight: 2 * 2.0 + 1.0 = 5.0.
    assert doubled.edit_distance_score == 5.0

    # The chosen configuration is recorded with the metrics.
    assert (
        doubled.weights.removed_poi_weight == 2.0
    )
    assert '"removed_poi_weight": 2.0' in (
        doubled.to_dict()["stability_weights"]
    )

    print("PASS: weight_sensitivity")


def test_locked_prefix_penalty_dominates_soft_cost() -> None:
    maximum_soft = maximum_soft_edit_distance(
        MAXIMUM_CANDIDATE_COUNT
    )

    # Documented bound: 2.25n + 0.25n(n-1) = 45.0 at n = 10.
    assert maximum_soft == 45.0
    assert (
        DEFAULT_STABILITY_WEIGHTS.locked_prefix_violation_weight
        > maximum_soft
    )

    # Worst permitted soft repair without a violation.
    soft_worst_case = compute_stability_metrics(
        original_suffix=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        replanned_suffix=[19, 18, 17, 16, 15, 14, 13, 12, 11, 10],
        locked_prefix=[1, 2],
    )

    assert soft_worst_case.order_deviation == 45
    assert soft_worst_case.locked_prefix_changed is False
    assert soft_worst_case.edit_distance_score == 22.5

    # A repair that only re-executes one locked POI scores far worse.
    violating = compute_stability_metrics(
        original_suffix=[10, 11],
        replanned_suffix=[2, 10, 11],
        locked_prefix=[1, 2],
    )

    assert violating.locked_prefix_changed is True
    assert (
        violating.edit_distance_score
        > soft_worst_case.edit_distance_score
    )

    print("PASS: locked_prefix_penalty_dominates_soft_cost")


# ================================================================
# Determinism
# ================================================================

def test_determinism() -> None:
    original_schedule = build_stub_schedule(
        {4: 600.0, 13: 650.0, 33: 700.0, 36: 750.0}
    )
    replanned_schedule = build_stub_schedule(
        {4: 610.0, 36: 640.0, 33: 720.0}
    )

    first = compute_stability_metrics(
        original_suffix=[4, 13, 33, 36],
        replanned_suffix=[36, 4, 33],
        locked_prefix=[1, 2],
        original_schedule=original_schedule,
        replanned_schedule=replanned_schedule,
    )
    second = compute_stability_metrics(
        original_suffix=[4, 13, 33, 36],
        replanned_suffix=[36, 4, 33],
        locked_prefix=[1, 2],
        original_schedule=original_schedule,
        replanned_schedule=replanned_schedule,
    )

    assert first == second
    assert first.to_dict() == second.to_dict()

    # A different input order for the same sets must not change the
    # identity-based counts (only the reported order may differ).
    reordered = compute_stability_metrics(
        original_suffix=[13, 4, 33, 36],
        replanned_suffix=[36, 4, 33],
        locked_prefix=[1, 2],
        original_schedule=original_schedule,
        replanned_schedule=replanned_schedule,
    )

    assert reordered.preserved_poi_count == first.preserved_poi_count
    assert reordered.removed_pois == first.removed_pois

    print("PASS: determinism")


# ================================================================
# Integration with benchmark_common.build_metric_record
# ================================================================

def test_metric_record_integration() -> None:
    data = load_planning_data(
        poi_file=POI_FILE,
        matrix_file=MATRIX_FILE,
    )

    routes_by_seed = load_initial_routes()
    initial_schedules = build_initial_schedules(
        data, routes_by_seed
    )
    scenario_lookup = build_scenario_lookup()

    initial_route = routes_by_seed[TEST_SEED]
    initial_schedule = initial_schedules[TEST_SEED]
    scenario_instance = scenario_lookup["S1_nominal"]
    constraints = PlanningConstraints()

    state = simulate_replanning_state(
        data=data,
        initial_schedule=initial_schedule,
        scenario_instance=scenario_instance,
    )

    result = run_greedy_repair(
        data=data,
        initial_schedule=initial_schedule,
        state=state,
        scenario_instance=scenario_instance,
        constraints=constraints,
        seed=TEST_SEED,
    )

    assert result.status == "success"
    assert result.selected_solution is not None

    solution = result.selected_solution

    record = build_metric_record(
        planner="Stability Test",
        scenario_instance=scenario_instance,
        seed=TEST_SEED,
        initial_route=initial_route,
        initial_schedule=initial_schedule,
        initial_route_utility=1.0,
        state=state,
        repaired_route=list(solution.route),
        achieved_utility=solution.achieved_utility,
        completed_poi_count=solution.completed_poi_count,
        total_travel_min=solution.total_travel_min,
        locked_travel_min=result.locked_travel_min,
        repaired_schedule=solution.schedule,
        exposure=None,
        feasible=True,
        failure_reason=None,
        mandatory_complete=True,
        runtime_seconds=0.01,
        deterministic=True,
        repair_record=result.repair_record,
    )

    required_stability_fields = {
        "preserved_poi_count",
        "preserved_poi_percentage",
        "removed_pois",
        "added_pois",
        "changed_activity_count",
        "modified_suffix_activity_count",
        "time_shift_count",
        "total_time_shift_min",
        "shifted_pois",
        "timing_measurement_available",
        "order_deviation",
        "edit_distance_score",
        "edit_distance_removed_component",
        "edit_distance_inserted_component",
        "edit_distance_order_component",
        "edit_distance_time_shift_component",
        "edit_distance_locked_prefix_component",
        "locked_prefix_changed",
        "hard_stability_violation",
        "stability_violation_reasons",
        "prefix_reappearing_pois",
        "removed_poi_reasons",
        "replacement_poi_reasons",
        "duplicate_original_pois",
        "duplicate_replanned_pois",
        "duplicate_locked_prefix_pois",
        "stability_weights",
        "stability_replanning_feasible",
    }

    missing = required_stability_fields - set(record)

    assert not missing, f"Missing stability fields: {sorted(missing)}"

    assert record["hard_stability_violation"] is False
    assert record["timing_measurement_available"] is True
    assert record["stability_replanning_feasible"] is True
    assert record["preserved_poi_count"] >= 0

    # The reasons recorded by the planner reach the metric record.
    if result.repair_record is not None and (
        result.repair_record.removed_activities
    ):
        assert record["removed_poi_reasons"] != "[]"

    print("PASS: metric_record_integration")


# ================================================================
# Real pipeline: a conforming repair never violates the prefix
# ================================================================

def test_real_pipeline_never_violates_locked_prefix() -> None:
    data = load_planning_data(
        poi_file=POI_FILE,
        matrix_file=MATRIX_FILE,
    )

    routes_by_seed = load_initial_routes()
    initial_schedules = build_initial_schedules(
        data, routes_by_seed
    )
    scenario_lookup = build_scenario_lookup()

    initial_route = routes_by_seed[TEST_SEED]
    initial_schedule = initial_schedules[TEST_SEED]
    constraints = PlanningConstraints()

    checked = 0

    for instance_id in sorted(scenario_lookup):
        scenario_instance = scenario_lookup[instance_id]

        state = simulate_replanning_state(
            data=data,
            initial_schedule=initial_schedule,
            scenario_instance=scenario_instance,
        )

        result = run_greedy_repair(
            data=data,
            initial_schedule=initial_schedule,
            state=state,
            scenario_instance=scenario_instance,
            constraints=constraints,
            seed=TEST_SEED,
        )

        metrics = compute_stability_metrics(
            original_suffix=state.original_remaining_poi_ids,
            replanned_suffix=(
                ()
                if result.selected_solution is None
                else result.selected_solution.route
            ),
            locked_prefix=state.completed_poi_ids,
            original_schedule=initial_schedule,
            replanned_schedule=(
                None
                if result.selected_solution is None
                else result.selected_solution.schedule
            ),
            repair_record=result.repair_record,
            replanning_feasible=(
                result.status == "success"
            ),
        )

        # The bounded-repair contract must hold on every instance.
        assert metrics.hard_stability_violation is False, (
            f"{instance_id}: locked prefix was reported as changed."
        )
        assert metrics.prefix_reappearing_pois == (), (
            f"{instance_id}: prefix POI reappeared in the suffix."
        )

        limit = (
            result.repair_record.bounded_repair_limit
            if result.repair_record is not None
            else MAXIMUM_CANDIDATE_COUNT
        )

        assert (
            metrics.modified_suffix_activity_count <= limit
        ), (
            f"{instance_id}: suffix size exceeds the bounded limit."
        )

        checked += 1

    assert checked == 18

    print("PASS: real_pipeline_never_violates_locked_prefix")


# ================================================================
# Runner
# ================================================================

TESTS = (
    test_no_changes_after_replanning,
    test_one_removed_poi,
    test_one_replacement_poi,
    test_shifted_activity_time,
    test_reordered_suffix,
    test_unchanged_locked_prefix,
    test_locked_prefix_violation,
    test_empty_suffix,
    test_infeasible_replanning_result,
    test_duplicate_pois,
    test_identical_itineraries,
    test_weight_config_validation,
    test_weight_sensitivity,
    test_locked_prefix_penalty_dominates_soft_cost,
    test_determinism,
    test_metric_record_integration,
    test_real_pipeline_never_violates_locked_prefix,
)


def main() -> None:
    for test in TESTS:
        test()

    print()
    print(
        f"All {len(TESTS)} stability-metric tests passed."
    )


if __name__ == "__main__":
    main()
