"""
Deterministic itinerary-stability metrics for TripMates Lite.

Purpose
-------
The repository measures *what a repaired itinerary achieves* (utility,
completed visits, travel time, WES).  It does not measure *how far the
repaired itinerary moved away from the original plan*.  This module adds
that missing measurement as a single reusable component, so no planner
duplicates the logic.

The comparison is always between two itineraries observed from the same
post-trigger state:

- the **original itinerary**: ``locked_prefix + original_suffix``, where
  ``original_suffix`` is ``ReplanningState.original_remaining_poi_ids``;
- the **replanned itinerary**: ``locked_prefix + replanned_suffix``,
  where ``replanned_suffix`` is the route returned by the planner.

Locked-prefix contract
----------------------
The locked prefix is **not** a modifiable part of the itinerary.  It is
therefore never counted as a permitted modification, and it is never
silently repaired or reordered here.  If the replanned suffix contains a
POI that belongs to the locked prefix, or if a supplied complete final
route does not start with the locked prefix, the result carries
``locked_prefix_changed=True`` and ``hard_stability_violation=True``
together with human-readable ``violation_reasons``.  The violation is
reported, never corrected.

Metric definitions (exact semantics)
------------------------------------
All POI-level counts are computed on the **unlocked suffix** only.  Both
sequences are first de-duplicated by first occurrence
(``_dedupe_preserving_order``) so a malformed route with repeated POI
IDs cannot inflate any count; repeated IDs are reported separately in
``duplicate_*`` fields.

1. ``preserved_poi_count``
   ``|original_suffix ∩ replanned_suffix|`` by POI identity,
   order-independent.  Answers: how many planned visits survived.

2. ``preserved_poi_percentage``
   ``preserved_poi_count / |original_suffix|``.  ``None`` when the
   original suffix is empty, because the ratio is then undefined.

3. ``removed_pois``
   Original-suffix POIs absent from the replanned suffix, listed in
   original execution order.

4. ``added_pois``
   Replanned-suffix POIs absent from the original suffix, listed in
   repaired execution order.  These are the inserted / replacement POIs.

5. ``changed_activity_count``
   ``len(removed_pois) + len(added_pois) + len(shifted_pois)``.
   A POI counts once even if it is both reordered and shifted.  When no
   replanned schedule is available, timing cannot be measured, so only
   membership changes are counted; ``timing_measurement_available`` is
   then ``False`` so the difference is never hidden.

6. ``time_shift_count`` / ``total_time_shift_min`` / ``shifted_pois``
   Among preserved POIs, those whose ``visit_start_min`` differs between
   the original and replanned schedules by more than
   ``TIME_SHIFT_TOLERANCE_MIN``.  ``None`` when either schedule is
   unavailable (for example an infeasible repair).

7. ``order_deviation``
   Kendall tau distance (number of discordant pairs) between the two
   *retained subsequences* — the original suffix and the replanned
   suffix each restricted to the preserved POIs.  ``0`` means the
   retained POIs keep their relative order even when new POIs are
   inserted between them.  Range: ``0 .. n(n-1)/2`` for ``n`` preserved
   POIs.

8. ``itinerary_edit_distance``
   Weighted sum of the five components below.  Deterministic, because
   every component is an integer count and every weight is a fixed
   constant from ``StabilityWeightConfig``.

9. ``modified_suffix_activity_count``
   ``len(replanned_suffix)`` — the number of activities the replanner
   placed in the unlocked (modifiable) suffix.  This is exactly the
   quantity bounded by ``BoundedRepairConfig
   .max_modifiable_suffix_activities`` and equal to
   ``RepairRecord.actual_modification_count``.

10. ``locked_prefix_changed``
    ``True`` when the locked prefix was altered.  See the contract
    section above.

11. ``removed_poi_reasons`` / ``replacement_poi_reasons``
    ``RepairDecision`` dictionaries copied from the ``RepairRecord``
    when one is supplied, otherwise empty tuples.

Edit-distance weights
---------------------
``StabilityWeightConfig`` holds every weight as a named constant so a
sensitivity analysis can substitute a different configuration without
editing this module:

======================  ======  =============================================
Component               Weight  Rationale
======================  ======  =============================================
removed POI             1.00    Losing one planned visit is the base unit of
                                change.
inserted POI            1.00    Inserting a visit is the symmetric unit of
                                change: one slot of the itinerary differs.
order change            0.50    Reordering keeps every POI; it costs half of
                                an identity change.
time shift              0.25    A retained POI that only moves in time is the
                                mildest change; a quarter unit.
locked-prefix violation 1000.0  A contract breach must dominate any
                                combination of permitted soft changes.
======================  ======  =============================================

Dominance proof for the violation weight: inside the bounded window of
at most ``MAXIMUM_CANDIDATE_COUNT`` = 10 suffix activities the maximum
soft cost is ``2.25n + 0.25n(n-1)`` = 45.0 at ``n = 10`` (see
``maximum_soft_edit_distance``).  A weight of 1000 therefore guarantees
that a violating repair always scores strictly worse than any conforming
repair, at every window size the contract permits.

Determinism guarantees
----------------------
- No set iteration affects any output order; every list follows the
  original or the repaired execution order.
- Kendall tau is computed by direct pairwise comparison, not by a
  library routine with unspecified tie handling.
- All floats are rounded to six decimals.
- The same inputs always produce the same ``StabilityMetrics``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    from planning_core import ScheduleResult


# ================================================================
# Tolerances
# ================================================================

TIME_SHIFT_TOLERANCE_MIN = 1e-6
"""Visit start times closer than this (in minutes) are treated as equal.

The decoder works in floating-point minutes; a one-microsecond tolerance
absorbs representation noise without ever hiding a real schedule shift
(the smallest meaningful shift in this dataset is one whole minute).
"""


# ================================================================
# Edit-distance weights (named constants)
# ================================================================

DEFAULT_REMOVED_POI_WEIGHT = 1.0
"""Cost of one original-suffix POI that disappears."""

DEFAULT_INSERTED_POI_WEIGHT = 1.0
"""Cost of one new POI inserted into the repaired suffix."""

DEFAULT_ORDER_CHANGE_WEIGHT = 0.5
"""Cost per discordant pair among the retained POIs."""

DEFAULT_TIME_SHIFT_WEIGHT = 0.25
"""Cost per retained POI whose visit start time moved."""

DEFAULT_LOCKED_PREFIX_VIOLATION_WEIGHT = 1000.0
"""Cost of a single locked-prefix violation.

Must dominate every permitted soft change inside the bounded window; see
``maximum_soft_edit_distance`` for the bound this value is derived from.
"""


@dataclass(frozen=True)
class StabilityWeightConfig:
    """Weights of the deterministic itinerary edit distance.

    Every field is a named constant with a documented rationale in the
    module docstring.  Substituting a different instance is the supported
    way to run a weight-sensitivity analysis; the chosen configuration is
    recorded in every metric record via ``to_dict``.
    """

    removed_poi_weight: float = DEFAULT_REMOVED_POI_WEIGHT
    inserted_poi_weight: float = DEFAULT_INSERTED_POI_WEIGHT
    order_change_weight: float = DEFAULT_ORDER_CHANGE_WEIGHT
    time_shift_weight: float = DEFAULT_TIME_SHIFT_WEIGHT
    locked_prefix_violation_weight: float = (
        DEFAULT_LOCKED_PREFIX_VIOLATION_WEIGHT
    )

    def __post_init__(self) -> None:
        soft_weights = {
            "removed_poi_weight": self.removed_poi_weight,
            "inserted_poi_weight": self.inserted_poi_weight,
            "order_change_weight": self.order_change_weight,
            "time_shift_weight": self.time_shift_weight,
        }

        for name, value in soft_weights.items():
            if not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric.")
            if value < 0:
                raise ValueError(f"{name} cannot be negative.")

        violation_weight = self.locked_prefix_violation_weight

        if not isinstance(violation_weight, (int, float)):
            raise TypeError(
                "locked_prefix_violation_weight must be numeric."
            )

        if violation_weight <= 0:
            raise ValueError(
                "locked_prefix_violation_weight must be positive so "
                "that a hard violation always registers."
            )

    def to_dict(self) -> dict:
        return {
            "removed_poi_weight": float(self.removed_poi_weight),
            "inserted_poi_weight": float(self.inserted_poi_weight),
            "order_change_weight": float(self.order_change_weight),
            "time_shift_weight": float(self.time_shift_weight),
            "locked_prefix_violation_weight": float(
                self.locked_prefix_violation_weight
            ),
        }


DEFAULT_STABILITY_WEIGHTS = StabilityWeightConfig()


# ================================================================
# Small deterministic helpers
# ================================================================

def _encode_json(values: Iterable) -> str:
    """Encode a sequence or mapping as a compact JSON string.

    CSV cells cannot hold native Python containers, so every list-valued
    field is stored as JSON text and remains machine-parseable.
    """

    return json.dumps(values, ensure_ascii=False)


def _dedupe_preserving_order(
    values: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return ``(unique_ids, duplicate_ids)`` in first-occurrence order."""

    seen: set[int] = set()
    unique: list[int] = []
    duplicates: list[int] = []

    for raw_value in values:
        poi_id = int(raw_value)

        if poi_id in seen:
            if poi_id not in duplicates:
                duplicates.append(poi_id)
            continue

        seen.add(poi_id)
        unique.append(poi_id)

    return tuple(unique), tuple(duplicates)


def _kendall_tau_distance(
    left: Sequence[int],
    right: Sequence[int],
) -> int:
    """Number of discordant pairs between two orderings of the same set.

    Both sequences must be permutations of the same POI IDs.  The result
    is the Kendall tau distance: ``0`` for identical order, and
    ``n(n-1)/2`` for a complete reversal.
    """

    if len(left) != len(right):
        raise ValueError(
            "Kendall tau requires two orderings of the same length."
        )

    if set(left) != set(right):
        raise ValueError(
            "Kendall tau requires two orderings of the same POI set."
        )

    right_position = {
        poi_id: index for index, poi_id in enumerate(right)
    }

    discordant_pairs = 0

    for first_index in range(len(left)):
        for second_index in range(first_index + 1, len(left)):
            if (
                right_position[left[first_index]]
                > right_position[left[second_index]]
            ):
                discordant_pairs += 1

    return discordant_pairs


def maximum_soft_edit_distance(
    max_suffix_activities: int,
    weights: StabilityWeightConfig | None = None,
) -> float:
    """Upper bound on the soft part of the edit distance.

    At most ``n`` POIs can be removed, ``n`` inserted, ``n`` shifted, and
    the order term is at most ``n(n-1)/2`` discordant pairs.  Used to
    justify ``DEFAULT_LOCKED_PREFIX_VIOLATION_WEIGHT`` and available to
    tests and documentation.
    """

    if max_suffix_activities < 0:
        raise ValueError(
            "max_suffix_activities cannot be negative."
        )

    active_weights = weights or DEFAULT_STABILITY_WEIGHTS
    n = int(max_suffix_activities)

    return round(
        n * active_weights.removed_poi_weight
        + n * active_weights.inserted_poi_weight
        + (n * (n - 1) / 2.0) * active_weights.order_change_weight
        + n * active_weights.time_shift_weight,
        6,
    )


# ================================================================
# Result structure
# ================================================================

@dataclass(frozen=True)
class StabilityMetrics:
    """Deterministic comparison of an original and a replanned itinerary.

    See the module docstring for the exact definition of every field.
    """

    replanning_feasible: bool

    preserved_poi_count: int
    preserved_poi_percentage: float | None
    removed_pois: tuple[int, ...]
    added_pois: tuple[int, ...]

    changed_activity_count: int
    modified_suffix_activity_count: int

    time_shift_count: int | None
    total_time_shift_min: float | None
    shifted_pois: tuple[int, ...]
    timing_measurement_available: bool

    order_deviation: int

    edit_distance_score: float
    edit_distance_components: dict[str, float]

    locked_prefix_changed: bool
    hard_stability_violation: bool
    violation_reasons: tuple[str, ...]
    prefix_reappearing_pois: tuple[int, ...]

    removed_poi_reasons: tuple[dict, ...]
    replacement_poi_reasons: tuple[dict, ...]

    duplicate_original_pois: tuple[int, ...]
    duplicate_replanned_pois: tuple[int, ...]
    duplicate_locked_prefix_pois: tuple[int, ...]

    weights: StabilityWeightConfig

    # ------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------

    def to_dict(self) -> dict:
        """Flat, CSV-safe representation for benchmark records."""

        return {
            "stability_replanning_feasible": self.replanning_feasible,
            "bounded_repair_feasible": self.replanning_feasible,

            "preserved_poi_count": self.preserved_poi_count,
            "preserved_poi_percentage": self.preserved_poi_percentage,
            "removed_pois": _encode_json(self.removed_pois),
            "added_pois": _encode_json(self.added_pois),
            "replacement_pois": _encode_json(self.added_pois),

            "changed_activity_count": self.changed_activity_count,
            "modification_count": self.changed_activity_count,
            "modified_suffix_activity_count": (
                self.modified_suffix_activity_count
            ),

            "time_shift_count": self.time_shift_count,
            "total_time_shift_min": self.total_time_shift_min,
            "shifted_pois": _encode_json(self.shifted_pois),
            "shifted_activity_count": len(self.shifted_pois),
            "timing_measurement_available": (
                self.timing_measurement_available
            ),

            "order_deviation": self.order_deviation,
            "order_change_count": self.order_deviation,

            "edit_distance_score": self.edit_distance_score,
            "itinerary_edit_distance": self.edit_distance_score,
            "edit_distance_removed_component": (
                self.edit_distance_components["removed"]
            ),
            "edit_distance_inserted_component": (
                self.edit_distance_components["inserted"]
            ),
            "edit_distance_order_component": (
                self.edit_distance_components["order"]
            ),
            "edit_distance_time_shift_component": (
                self.edit_distance_components["time_shift"]
            ),
            "edit_distance_locked_prefix_component": (
                self.edit_distance_components[
                    "locked_prefix_violation"
                ]
            ),

            "locked_prefix_changed": self.locked_prefix_changed,
            "locked_prefix_violation": self.locked_prefix_changed,
            "hard_stability_violation": (
                self.hard_stability_violation
            ),
            "stability_violation_reasons": _encode_json(
                self.violation_reasons
            ),
            "prefix_reappearing_pois": _encode_json(
                self.prefix_reappearing_pois
            ),

            "removed_poi_reasons": _encode_json(
                self.removed_poi_reasons
            ),
            "replacement_poi_reasons": _encode_json(
                self.replacement_poi_reasons
            ),

            "duplicate_original_pois": _encode_json(
                self.duplicate_original_pois
            ),
            "duplicate_replanned_pois": _encode_json(
                self.duplicate_replanned_pois
            ),
            "duplicate_locked_prefix_pois": _encode_json(
                self.duplicate_locked_prefix_pois
            ),

            "stability_weights": _encode_json(self.weights.to_dict()),
        }

    def to_summary_dict(self) -> dict:
        """Compact numeric subset for paired comparison tables."""

        return {
            "preserved_poi_count": self.preserved_poi_count,
            "preserved_poi_percentage": self.preserved_poi_percentage,
            "removed_poi_count": len(self.removed_pois),
            "added_poi_count": len(self.added_pois),
            "changed_activity_count": self.changed_activity_count,
            "modified_suffix_activity_count": (
                self.modified_suffix_activity_count
            ),
            "time_shift_count": self.time_shift_count,
            "order_deviation": self.order_deviation,
            "edit_distance_score": self.edit_distance_score,
            "locked_prefix_changed": self.locked_prefix_changed,
            "hard_stability_violation": (
                self.hard_stability_violation
            ),
        }


# ================================================================
# Core computation
# ================================================================

def compute_stability_metrics(
    *,
    original_suffix: Sequence[int],
    replanned_suffix: Sequence[int],
    locked_prefix: Sequence[int] = (),
    original_schedule: "ScheduleResult | None" = None,
    replanned_schedule: "ScheduleResult | None" = None,
    repair_record=None,
    weights: StabilityWeightConfig | None = None,
    final_route: Sequence[int] | None = None,
    replanning_feasible: bool = True,
) -> StabilityMetrics:
    """Compare an original suffix with a replanned suffix.

    Parameters
    ----------
    original_suffix:
        ``ReplanningState.original_remaining_poi_ids`` — the unfinished
        part of the original itinerary, in original execution order.
    replanned_suffix:
        The route returned by the planner, in repaired execution order.
        Pass ``()`` when the planner produced no route.
    locked_prefix:
        ``ReplanningState.completed_poi_ids`` — immutable, never counted
        as a modification, but verified.
    original_schedule:
        The full pre-disruption schedule.  Used only to read original
        visit start times for the time-shift metric.
    replanned_schedule:
        The post-disruption schedule.  ``None`` disables the time-shift
        metric rather than guessing.
    repair_record:
        Optional ``RepairRecord``; supplies per-POI reasons for removed
        and replacement POIs when the planner recorded them.
    weights:
        Edit-distance weights.  Defaults to
        ``DEFAULT_STABILITY_WEIGHTS``.
    final_route:
        Optional complete itinerary (``locked_prefix + suffix``).  When
        supplied, its prefix is verified against ``locked_prefix``.
    replanning_feasible:
        Recorded verbatim so a reader can tell an infeasible repair with
        an empty suffix apart from a deliberately empty repair.
    """

    active_weights = weights or DEFAULT_STABILITY_WEIGHTS

    original_ids, duplicate_original = _dedupe_preserving_order(
        original_suffix
    )
    replanned_ids, duplicate_replanned = _dedupe_preserving_order(
        replanned_suffix
    )
    locked_ids, duplicate_locked = _dedupe_preserving_order(
        locked_prefix
    )

    original_set = set(original_ids)
    replanned_set = set(replanned_ids)
    locked_set = set(locked_ids)

    # ------------------------------------------------------------
    # 1-4. Preservation, removal and insertion
    # ------------------------------------------------------------

    preserved_ids = tuple(
        poi_id
        for poi_id in original_ids
        if poi_id in replanned_set
    )

    removed_ids = tuple(
        poi_id
        for poi_id in original_ids
        if poi_id not in replanned_set
    )

    added_ids = tuple(
        poi_id
        for poi_id in replanned_ids
        if poi_id not in original_set
    )

    preserved_percentage = (
        None
        if not original_ids
        else round(len(preserved_ids) / len(original_ids), 6)
    )

    # ------------------------------------------------------------
    # 6. Time shifts (only when both schedules are available)
    # ------------------------------------------------------------

    timing_available = (
        original_schedule is not None
        and replanned_schedule is not None
    )

    shifted_ids: tuple[int, ...] = ()
    total_time_shift_min: float | None = None
    time_shift_count: int | None = None

    if timing_available:
        original_start_by_poi = {
            entry.poi_id: float(entry.visit_start_min)
            for entry in original_schedule.entries
        }

        replanned_start_by_poi = {
            entry.poi_id: float(entry.visit_start_min)
            for entry in replanned_schedule.entries
        }

        shift_records: list[tuple[int, float]] = []

        for poi_id in preserved_ids:
            original_start = original_start_by_poi.get(poi_id)
            replanned_start = replanned_start_by_poi.get(poi_id)

            if original_start is None or replanned_start is None:
                continue

            difference = replanned_start - original_start

            if abs(difference) > TIME_SHIFT_TOLERANCE_MIN:
                shift_records.append((poi_id, difference))

        shifted_ids = tuple(
            poi_id for poi_id, _ in shift_records
        )

        time_shift_count = len(shifted_ids)

        total_time_shift_min = round(
            sum(abs(delta) for _, delta in shift_records),
            6,
        )

    # ------------------------------------------------------------
    # 7. Order deviation among retained POIs
    # ------------------------------------------------------------

    preserved_set = set(preserved_ids)

    original_retained_order = tuple(
        poi_id
        for poi_id in original_ids
        if poi_id in preserved_set
    )

    replanned_retained_order = tuple(
        poi_id
        for poi_id in replanned_ids
        if poi_id in preserved_set
    )

    order_deviation = _kendall_tau_distance(
        original_retained_order,
        replanned_retained_order,
    )

    # ------------------------------------------------------------
    # 5. Changed activities
    # ------------------------------------------------------------

    changed_activity_count = (
        len(removed_ids)
        + len(added_ids)
        + (time_shift_count or 0)
    )

    # ------------------------------------------------------------
    # 10. Locked-prefix verification (report, never correct)
    # ------------------------------------------------------------

    prefix_reappearing_ids = tuple(
        poi_id
        for poi_id in replanned_ids
        if poi_id in locked_set
    )

    violation_reasons: list[str] = []

    if prefix_reappearing_ids:
        violation_reasons.append(
            "locked-prefix POIs reappear in the repaired suffix: "
            f"{list(prefix_reappearing_ids)}"
        )

    if final_route is not None:
        final_ids, _ = _dedupe_preserving_order(final_route)
        expected_prefix = tuple(locked_ids)
        observed_prefix = tuple(final_ids[: len(expected_prefix)])

        if observed_prefix != expected_prefix:
            violation_reasons.append(
                "final route does not begin with the locked prefix: "
                f"expected {list(expected_prefix)}, "
                f"observed {list(observed_prefix)}"
            )

    locked_prefix_changed = bool(violation_reasons)

    # ------------------------------------------------------------
    # 8. Deterministic weighted edit distance
    # ------------------------------------------------------------

    components = {
        "removed": round(
            len(removed_ids) * active_weights.removed_poi_weight, 6
        ),
        "inserted": round(
            len(added_ids) * active_weights.inserted_poi_weight, 6
        ),
        "order": round(
            order_deviation * active_weights.order_change_weight, 6
        ),
        "time_shift": round(
            (time_shift_count or 0) * active_weights.time_shift_weight, 6
        ),
        "locked_prefix_violation": round(
            (1.0 if locked_prefix_changed else 0.0)
            * active_weights.locked_prefix_violation_weight,
            6,
        ),
    }

    edit_distance_score = round(sum(components.values()), 6)

    # ------------------------------------------------------------
    # 11. Reasons from the repair record, when available
    # ------------------------------------------------------------

    removed_reasons: tuple[dict, ...] = ()
    replacement_reasons: tuple[dict, ...] = ()

    if repair_record is not None:
        removed_reasons = tuple(
            decision.to_dict()
            for decision in repair_record.removed_activities
        )
        replacement_reasons = tuple(
            decision.to_dict()
            for decision in repair_record.replacements
        )

    return StabilityMetrics(
        replanning_feasible=bool(replanning_feasible),
        preserved_poi_count=len(preserved_ids),
        preserved_poi_percentage=preserved_percentage,
        removed_pois=removed_ids,
        added_pois=added_ids,
        changed_activity_count=changed_activity_count,
        modified_suffix_activity_count=len(replanned_ids),
        time_shift_count=time_shift_count,
        total_time_shift_min=total_time_shift_min,
        shifted_pois=shifted_ids,
        timing_measurement_available=timing_available,
        order_deviation=order_deviation,
        edit_distance_score=edit_distance_score,
        edit_distance_components=components,
        locked_prefix_changed=locked_prefix_changed,
        hard_stability_violation=locked_prefix_changed,
        violation_reasons=tuple(violation_reasons),
        prefix_reappearing_pois=prefix_reappearing_ids,
        removed_poi_reasons=removed_reasons,
        replacement_poi_reasons=replacement_reasons,
        duplicate_original_pois=duplicate_original,
        duplicate_replanned_pois=duplicate_replanned,
        duplicate_locked_prefix_pois=duplicate_locked,
        weights=active_weights,
    )


# ================================================================
# Convenience wrapper for a ReplanningState
# ================================================================

def compute_stability_from_state(
    *,
    state,
    replanned_suffix: Sequence[int],
    original_schedule: "ScheduleResult | None" = None,
    replanned_schedule: "ScheduleResult | None" = None,
    repair_record=None,
    weights: StabilityWeightConfig | None = None,
    replanning_feasible: bool = True,
) -> StabilityMetrics:
    """Compute stability directly from a ``ReplanningState``.

    Reads ``completed_poi_ids`` as the locked prefix and
    ``original_remaining_poi_ids`` as the original suffix, so callers do
    not repeat that mapping.

    The locked-prefix violation signal here is
    ``prefix_reappearing_pois``: a repaired suffix that re-executes a
    completed visit.  Callers that hold an independently produced
    complete itinerary should call ``compute_stability_metrics`` directly
    and pass it as ``final_route`` so the prefix can be audited too.
    """

    locked_prefix = tuple(
        getattr(state, "completed_poi_ids", ())
    )
    original_suffix = tuple(
        getattr(state, "original_remaining_poi_ids", ())
    )

    return compute_stability_metrics(
        original_suffix=original_suffix,
        replanned_suffix=replanned_suffix,
        locked_prefix=locked_prefix,
        original_schedule=original_schedule,
        replanned_schedule=replanned_schedule,
        repair_record=repair_record,
        weights=weights,
        replanning_feasible=replanning_feasible,
    )
