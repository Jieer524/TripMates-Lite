"""
Bounded-replanning GA core for TripMates Lite.

This module defines the enforceable bounded-repair contract:

1. **Locked prefix.**  Visits completed before the disruption trigger
   are immutable.  The replanner may not reorder, remove or substitute
   them.

2. **Current activity.**  If the traveller is mid-walking, waiting, or
   mid-visit at the trigger, the existing scenario semantics apply
   consistently (see ``simulate_replanning_state``).  A committed
   walking leg is completed; a committed visit is completed and locked;
   a waiting interval starts replanning from the trigger time.

3. **Modifiable suffix only.**  Only the unfinished suffix may be
   modified by the repair algorithm.

4. **Bounded window.**  An explicit ``max_modifiable_suffix_activities``
   parameter caps the number of activities the replanner may place in
   the repaired suffix.  The default is ``MAXIMUM_CANDIDATE_COUNT`` (10).

5. **Configurable and recorded.**  The bound is configurable via
   ``BoundedRepairConfig`` and recorded in every experiment
   configuration and every repair record.

6. **Repair-decision audit.**  Removed, skipped, shifted and replacement
   POIs are recorded with a reason in ``RepairDecision`` entries.

7. **Explicit failure.**  If no feasible bounded repair exists, an
   explicit failure result is returned — never a silently infeasible
   itinerary.

8. **Hard-constraint enforcement.**  The single decoder
   (``planning_core.decode_route``) is the only feasibility gate.
   Opening hours, budget, mandatory POIs, route, duplication and
   end-time constraints are never bypassed.

9. **Shared decoder.**  All adaptive planners use the same decoder and
   hard constraints.

10. **Planner parity.**  Adaptive Without Thermal and TripMates Lite
    use identical candidate pools, search budgets and bounded-repair
    constraints.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import permutations
import math
import random

from ga_core import (
    GAConfig,
    EvaluatedSolution,
    Individual,
    order_crossover,
    rank_population,
    select_final_solution,
    swap_mutation,
    tournament_select,
)

from planning_core import (
    PlanningConstraints,
    PlanningData,
    ScheduleResult,
    decode_route,
)

from scenario_core import (
    ReplanningState,
    ScenarioInstance,
    get_travel_adjustments,
)


# ================================================================
# Replanning configuration
# ================================================================

REPLANNING_GA_CONFIG = GAConfig(
    population_size=40,
    maximum_generations=60,
    tournament_size=3,
    crossover_probability=0.90,
    mutation_probability=0.20,
    elitism_count=2,
    stagnation_limit=15,
)

MAXIMUM_CANDIDATE_COUNT = 10


# ================================================================
# Bounded-repair configuration
# ================================================================

@dataclass(frozen=True)
class BoundedRepairConfig:
    """Enforceable bounded-repair policy.

    ``candidate_window_size`` controls how many feasible candidates are
    offered to the optimizer. ``max_modifiable_suffix_activities`` is a
    separate hard cap on the repaired suffix length.

    This object is recorded in every experiment configuration so that
    the bounded-repair behaviour can be audited after an experiment.
    """

    max_modifiable_suffix_activities: int = (
        MAXIMUM_CANDIDATE_COUNT
    )
    candidate_window_size: int | str | None = None

    def __post_init__(self) -> None:
        if (
            self.max_modifiable_suffix_activities
            < 1
        ):
            raise ValueError(
                "max_modifiable_suffix_activities "
                "must be at least 1."
            )
        if self.max_modifiable_suffix_activities > MAXIMUM_CANDIDATE_COUNT:
            raise ValueError(
                "max_modifiable_suffix_activities must not exceed "
                f"{MAXIMUM_CANDIDATE_COUNT}."
            )

        effective_window = (
            self.max_modifiable_suffix_activities
            if self.candidate_window_size is None
            else self.candidate_window_size
        )
        object.__setattr__(self, "candidate_window_size", effective_window)

        if effective_window != "all" and effective_window > 0 and effective_window < 1:
            raise ValueError("candidate_window_size must be at least 1.")

        if effective_window != "all" and effective_window < 1:
            raise ValueError("candidate_window_size must be at least 1.")

        # The candidate window and the repaired-suffix limit are deliberately
        # independent controls. A small candidate window simply offers fewer
        # choices; it must not rewrite the separately configured repair cap.

    def to_dict(self) -> dict:
        return asdict(self)


# ================================================================
# Repair-decision audit structures
# ================================================================

@dataclass(frozen=True)
class RepairDecision:
    """A single POI-level repair decision with a reason.

    ``decision_type`` is one of:
      - ``"retained"``     – original suffix POI kept in the repair.
      - ``"removed"``      – original suffix POI removed (skipped).
      - ``"shifted"``      – original suffix POI retained but moved
                              to a different position.
      - ``"replacement"``  – a new POI not in the original suffix,
                              substituted for a removed/skipped POI.
    """

    poi_id: int
    decision_type: str
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RepairRecord:
    """Full, auditable record of a bounded-repair outcome.

    Fields:
      locked_prefix: completed POI IDs before the trigger (immutable).
      current_activity: what the traveller was doing at the trigger.
      modified_suffix: POI IDs in the repaired suffix (in order).
      retained_activities: original suffix POIs kept in the repair.
      removed_activities: original suffix POIs removed, with reasons.
      replacements: new POIs introduced, with reasons.
      failure_status: "success", "no_action", or "infeasible".
      failure_reason: human-readable reason on failure, else None.
      bounded_repair_limit: the configured max suffix activities.
      actual_modification_count: number of suffix activities in the
                                  repair (len of modified_suffix).
    """

    locked_prefix: tuple[int, ...]
    current_activity: str
    modified_suffix: tuple[int, ...]
    retained_activities: tuple[int, ...]
    removed_activities: tuple[RepairDecision, ...]
    replacements: tuple[RepairDecision, ...]
    failure_status: str
    failure_reason: str | None
    bounded_repair_limit: int
    actual_modification_count: int

    @property
    def bounded_repair_feasible(self) -> bool:
        return self.failure_status != "infeasible"

    def to_dict(self) -> dict:
        return {
            "locked_prefix": list(self.locked_prefix),
            "current_activity": self.current_activity,
            "modified_suffix": list(self.modified_suffix),
            "retained_activities": list(
                self.retained_activities
            ),
            "removed_activities": [
                d.to_dict()
                for d in self.removed_activities
            ],
            "replacements": [
                d.to_dict()
                for d in self.replacements
            ],
            "failure_status": self.failure_status,
            "failure_reason": self.failure_reason,
            "bounded_repair_limit": (
                self.bounded_repair_limit
            ),
            "actual_modification_count": (
                self.actual_modification_count
            ),
            "bounded_repair_feasible": self.bounded_repair_feasible,
            "failure_reason": self.failure_reason,
        }


# ================================================================
# Replanning data structures
# ================================================================

@dataclass(frozen=True)
class CandidateWindow:
    all_candidate_ids: tuple[int, ...]
    mandatory_ids: tuple[int, ...]
    optional_ids: tuple[int, ...]

    # Stored for reproducibility and later reporting.
    prescreen_scores: tuple[
        tuple[int, float],
        ...
    ]

@dataclass(frozen=True)
class AdaptiveProblem:
    state: ReplanningState
    scenario_instance: ScenarioInstance

    candidate_window: CandidateWindow
    mandatory_backbone: tuple[int, ...]

    locked_travel_min: float
    locked_completed_count: int
    locked_utility: float
    constraints: PlanningConstraints = PlanningConstraints()
    bounded_repair_config: BoundedRepairConfig = field(
        default_factory=BoundedRepairConfig,
    )


@dataclass(frozen=True)
class AdaptiveGARunResult:
    seed: int
    problem: AdaptiveProblem

    selected_solution: EvaluatedSolution

    generations_executed: int
    convergence_generation: int
    stopped_by_stagnation: bool

    evaluated_chromosome_count: int
    unique_decoded_route_count: int

    # Final first Pareto front, in population order.  Exposed so that
    # selection-policy ablations can re-select a route from the identical
    # population without re-running (and therefore perturbing) the GA.
    # The field is additive: the frozen selection rule still decides
    # ``selected_solution``.
    first_front: tuple[EvaluatedSolution, ...] = ()

    repair_record: RepairRecord | None = None


# ================================================================
# Shared route decoding
# ================================================================

def decode_replanned_route(
    data: PlanningData,
    state: ReplanningState,
    scenario_instance: ScenarioInstance,
    route: list[int] | tuple[int, ...],
    constraints: PlanningConstraints | None = None,
) -> ScheduleResult:
    """
    Decode a route beginning from the post-trigger state.

    Completed costs are supplied as initial_cost_rm.

    Post-trigger travel changes are applied only to the repaired
    route. A completed or committed pre-trigger component is not
    recalculated using disruption multipliers.
    """

    active_constraints = constraints or PlanningConstraints()

    travel_adjustments = (
        get_travel_adjustments(
            data,
            scenario_instance.scenario,
        )
    )

    return decode_route(
        data=data,
        visit_sequence=route,

        start_poi_id=state.current_poi_id,
        start_time_min=state.replan_start_min,
        trip_end_min=active_constraints.trip_end_min,

        day_code=active_constraints.day_code,
        cost_budget_rm=active_constraints.cost_budget_rm,
        initial_cost_rm=state.cost_spent_rm,

        start_poi_is_visit=False,

        required_mandatory_ids=(
            state.remaining_mandatory_ids
        ),

        unavailable_poi_ids=(
            state.unavailable_poi_ids
        ),

        travel_adjustments=(
            travel_adjustments
        ),
    )


# ================================================================
# Locked prefix calculation
# ================================================================

def calculate_locked_travel_minutes(
    initial_schedule: ScheduleResult,
    state: ReplanningState,
) -> float:
    """
    Calculate executed or committed travel before replanning.

    Travel to completed visits is always locked.

    When the trigger occurs during a walking leg or while waiting
    at its destination, that complete walking leg is also locked.
    """

    completed_ids = set(
        state.completed_poi_ids
    )

    total = sum(
        entry.travel_min
        for entry in initial_schedule.entries
        if entry.poi_id in completed_ids
    )

    if (
        state.current_poi_requires_visit
        and state.committed_component
        in {"walking_leg", "waiting"}
    ):
        matching_entries = [
            entry
            for entry in initial_schedule.entries
            if entry.poi_id == state.current_poi_id
        ]

        if len(matching_entries) != 1:
            raise RuntimeError(
                "Could not identify the committed "
                "walking-leg destination."
            )

        total += matching_entries[0].travel_min

    return total


# ================================================================
# Remaining mandatory backbone
# ================================================================

def build_remaining_mandatory_backbone(
    data: PlanningData,
    state: ReplanningState,
    scenario_instance: ScenarioInstance,
    constraints: PlanningConstraints | None = None,
) -> tuple[int, ...]:
    """
    Enumerate all remaining mandatory-POI orders exactly.

    Selection priority:

    1. lowest post-trigger travel time;
    2. earliest finish time;
    3. lower POI-ID sequence.
    """

    mandatory_ids = tuple(
        sorted(
            state.remaining_mandatory_ids
        )
    )

    if not mandatory_ids:
        empty_result = (
            decode_replanned_route(
                data=data,
                state=state,
                scenario_instance=(
                    scenario_instance
                ),
                route=(),
                constraints=constraints,
            )
        )

        if not empty_result.feasible:
            raise RuntimeError(
                "The empty remaining route "
                "is unexpectedly infeasible."
            )

        return ()

    feasible_backbones: list[
        tuple[
            float,
            float,
            tuple[int, ...],
        ]
    ] = []

    for ordering in permutations(
        mandatory_ids
    ):
        result = decode_replanned_route(
            data=data,
            state=state,
            scenario_instance=(
                scenario_instance
            ),
            route=ordering,
            constraints=constraints,
        )

        if not result.feasible:
            continue

        feasible_backbones.append(
            (
                result.total_travel_min,
                result.finish_min,
                tuple(ordering),
            )
        )

    if not feasible_backbones:
        raise RuntimeError(
            f"{state.instance_id}: no feasible "
            "post-trigger ordering exists for "
            "the remaining mandatory POIs "
            f"{list(mandatory_ids)}."
        )

    feasible_backbones.sort()

    return feasible_backbones[0][2]


# ================================================================
# Candidate feasibility
# ================================================================

def candidate_has_feasible_insertion(
    data: PlanningData,
    state: ReplanningState,
    scenario_instance: ScenarioInstance,
    mandatory_backbone: tuple[int, ...],
    candidate_poi_id: int,
    constraints: PlanningConstraints | None = None,
) -> bool:
    """
    Check whether a candidate can be inserted while preserving all
    still-feasible mandatory visits.
    """

    if candidate_poi_id in mandatory_backbone:
        result = decode_replanned_route(
            data=data,
            state=state,
            scenario_instance=(
                scenario_instance
            ),
            route=mandatory_backbone,
            constraints=constraints,
        )

        return result.feasible

    for insertion_position in range(
        0,
        len(mandatory_backbone) + 1,
    ):
        test_route = (
            list(
                mandatory_backbone[
                    :insertion_position
                ]
            )
            + [candidate_poi_id]
            + list(
                mandatory_backbone[
                    insertion_position:
                ]
            )
        )

        result = decode_replanned_route(
            data=data,
            state=state,
            scenario_instance=(
                scenario_instance
            ),
            route=test_route,
            constraints=constraints,
        )

        if result.feasible:
            return True

    return False


# ================================================================
# Shared candidate window
# ================================================================

def calculate_prescreen_score(
    data: PlanningData,
    state: ReplanningState,
    scenario_instance: ScenarioInstance,
    candidate_poi_id: int,
) -> float:
    """
    Rank an eligible optional POI using the shared screening rule:

        U_j / (T_ij + V_j)

    The candidate window is deliberately identical for Adaptive
    Without Thermal and TripMates Lite. Thermal information affects
    only Pareto evaluation and final route selection, not which
    candidates enter the bounded search space.
    """

    poi = data.pois[
        candidate_poi_id
    ]

    if candidate_poi_id == state.current_poi_id:
        travel_min = 0.0

    else:
        base_travel_min = (
            data.get_travel_time(
                state.current_poi_id,
                candidate_poi_id,
            )
        )

        adjustments = get_travel_adjustments(
            data,
            scenario_instance.scenario,
        )

        travel_min = adjustments.apply(
            state.current_poi_id,
            candidate_poi_id,
            base_travel_min,
        )

    visit_min = (
        poi.visit_duration_min
    )

    denominator = (
        travel_min
        + visit_min
    )

    if denominator <= 0:
        raise ValueError(
            "Candidate pre-screen denominator "
            "must be positive."
        )

    return (
        poi.utility_score
        / denominator
    )

def build_shared_candidate_window(
    data: PlanningData,
    state: ReplanningState,
    scenario_instance: ScenarioInstance,
    mandatory_backbone: tuple[int, ...],
    constraints: PlanningConstraints | None = None,
    bounded_repair_config: BoundedRepairConfig | None = None,
) -> CandidateWindow:
    """
    Build a bounded candidate window.

    The maximum number of candidates is
    ``bounded_repair_config.max_modifiable_suffix_activities``
    (default ``MAXIMUM_CANDIDATE_COUNT`` = 10).

    Rules:

    1. Exclude completed and unavailable POIs.
    2. Include all remaining mandatory POIs first.
    3. Exclude candidates that cannot coexist with all remaining
       mandatory POIs.
    4. Fill remaining positions using U / (travel + visit).
    5. Use lower POI ID as the final deterministic tie-breaker.
    """

    active_bounded_repair = (
        bounded_repair_config or BoundedRepairConfig()
    )

    window_limit = active_bounded_repair.candidate_window_size

    completed_ids = set(
        state.completed_poi_ids
    )

    unavailable_ids = set(
        state.unavailable_poi_ids
    )

    mandatory_ids = tuple(
        sorted(
            state.remaining_mandatory_ids
        )
    )

    if window_limit == "all":
        window_limit = len(data.pois)
    if len(mandatory_ids) > window_limit:
        raise RuntimeError(
            "The number of remaining mandatory POIs "
            f"({len(mandatory_ids)}) exceeds the "
            "bounded-repair window limit "
            f"({window_limit})."
        )

    possible_ids = sorted(
        set(data.pois)
        - completed_ids
        - unavailable_ids
    )

    # A previously completed current POI must not be revisited.
    # A current POI reached through walking or waiting remains
    # unvisited and is therefore still eligible.
    if not state.current_poi_requires_visit:
        possible_ids = [
            poi_id
            for poi_id in possible_ids
            if poi_id != state.current_poi_id
        ]

    optional_records: list[
        tuple[
            float,
            int,
        ]
    ] = []

    score_lookup: dict[
        int,
        float,
    ] = {}

    for candidate_poi_id in possible_ids:
        if candidate_poi_id in mandatory_ids:
            continue

        feasible = (
            candidate_has_feasible_insertion(
                data=data,
                state=state,
                scenario_instance=(
                    scenario_instance
                ),
                mandatory_backbone=(
                    mandatory_backbone
                ),
                candidate_poi_id=(
                    candidate_poi_id
                ),
                constraints=constraints,
            )
        )

        if not feasible:
            continue

        score = calculate_prescreen_score(
            data=data,
            state=state,
            scenario_instance=(
                scenario_instance
            ),
            candidate_poi_id=(
                candidate_poi_id
            ),
        )

        score_lookup[
            candidate_poi_id
        ] = score

        optional_records.append(
            (
                score,
                candidate_poi_id,
            )
        )

    # Highest score first, followed by lower POI ID.
    optional_records.sort(
        key=lambda item: (
            -item[0],
            item[1],
        )
    )

    remaining_slots = window_limit - len(mandatory_ids)

    selected_optional_ids = tuple(
        poi_id
        for _, poi_id in (
            optional_records[
                :remaining_slots
            ]
        )
    )

    all_candidate_ids = (
        mandatory_ids
        + selected_optional_ids
    )

    prescreen_scores = tuple(
        (
            poi_id,
            round(
                score_lookup[poi_id],
                12,
            ),
        )
        for poi_id in selected_optional_ids
    )

    if len(all_candidate_ids) > window_limit:
        raise RuntimeError(
            "Candidate window exceeded "
            "the bounded-repair limit."
        )

    return CandidateWindow(
        all_candidate_ids=(
            all_candidate_ids
        ),

        mandatory_ids=(
            mandatory_ids
        ),

        optional_ids=(
            selected_optional_ids
        ),

        prescreen_scores=(
            prescreen_scores
        ),
    )


# ================================================================
# Adaptive problem preparation
# ================================================================

def prepare_adaptive_problem(
    data: PlanningData,
    initial_schedule: ScheduleResult,
    state: ReplanningState,
    scenario_instance: ScenarioInstance,
    constraints: PlanningConstraints | None = None,
    bounded_repair_config: BoundedRepairConfig | None = None,
) -> AdaptiveProblem:
    active_constraints = constraints or PlanningConstraints()
    active_bounded_repair = (
        bounded_repair_config or BoundedRepairConfig()
    )

    mandatory_ids = tuple(
        sorted(state.remaining_mandatory_ids)
    )

    if (
        len(mandatory_ids)
        > active_bounded_repair
        .max_modifiable_suffix_activities
    ):
        raise RuntimeError(
            f"{state.instance_id}: the number of remaining "
            f"mandatory POIs ({len(mandatory_ids)}) exceeds "
            "the bounded-repair limit "
            f"({active_bounded_repair}"
            f".max_modifiable_suffix_activities). "
            "No feasible bounded repair can exist."
        )

    mandatory_backbone = (
        build_remaining_mandatory_backbone(
            data=data,
            state=state,
            scenario_instance=(
                scenario_instance
            ),
            constraints=active_constraints,
        )
    )

    candidate_window = (
        build_shared_candidate_window(
            data=data,
            state=state,
            scenario_instance=(
                scenario_instance
            ),
            mandatory_backbone=(
                mandatory_backbone
            ),
            constraints=active_constraints,
            bounded_repair_config=(
                active_bounded_repair
            ),
        )
    )

    locked_travel_min = (
        calculate_locked_travel_minutes(
            initial_schedule=(
                initial_schedule
            ),
            state=state,
        )
    )

    return AdaptiveProblem(
        state=state,

        scenario_instance=(
            scenario_instance
        ),

        candidate_window=(
            candidate_window
        ),

        mandatory_backbone=(
            mandatory_backbone
        ),

        locked_travel_min=(
            locked_travel_min
        ),

        locked_completed_count=len(
            state.completed_poi_ids
        ),

        locked_utility=(
            state.utility_achieved
        ),
        constraints=active_constraints,
        bounded_repair_config=(
            active_bounded_repair
        ),
    )


# ================================================================
# Repair-record builder
# ================================================================

def build_repair_record(
    state: ReplanningState,
    problem: AdaptiveProblem,
    repaired_route: tuple[int, ...],
    *,
    failure_status: str = "success",
    failure_reason: str | None = None,
) -> RepairRecord:
    """Build a ``RepairRecord`` from the problem state and the
    repaired suffix route.

    The caller is responsible for ensuring the route has been
    validated by ``decode_replanned_route`` when
    ``failure_status`` is ``"success"``.
    """

    original_remaining = set(
        state.original_remaining_poi_ids
    )

    repaired_set = set(repaired_route)

    retained = [
        poi_id
        for poi_id in state.original_remaining_poi_ids
        if poi_id in repaired_set
    ]

    removed = [
        RepairDecision(
            poi_id=poi_id,
            decision_type="removed",
            reason=(
                "unavailable (closed by scenario)"
                if poi_id in set(
                    state.unavailable_poi_ids
                )
                else "skipped (infeasible in bounded repair)"
            ),
        )
        for poi_id in state.original_remaining_poi_ids
        if poi_id not in repaired_set
    ]

    replacements = [
        RepairDecision(
            poi_id=poi_id,
            decision_type="replacement",
            reason=(
                "substituted for a removed or "
                "skipped suffix POI"
            ),
        )
        for poi_id in repaired_route
        if poi_id not in original_remaining
    ]

    return RepairRecord(
        locked_prefix=state.completed_poi_ids,
        current_activity=state.committed_component,
        modified_suffix=tuple(repaired_route),
        retained_activities=tuple(retained),
        removed_activities=tuple(removed),
        replacements=tuple(replacements),
        failure_status=failure_status,
        failure_reason=failure_reason,
        bounded_repair_limit=(
            problem
            .bounded_repair_config
            .max_modifiable_suffix_activities
        ),
        actual_modification_count=len(repaired_route),
    )


def build_failure_repair_record(
    state: ReplanningState,
    bounded_repair_limit: int,
    *,
    failure_reason: str,
) -> RepairRecord:
    """Build a ``RepairRecord`` for an explicit failure outcome."""

    return RepairRecord(
        locked_prefix=state.completed_poi_ids,
        current_activity=state.committed_component,
        modified_suffix=(),
        retained_activities=(),
        removed_activities=tuple(
            RepairDecision(
                poi_id=poi_id,
                decision_type="removed",
                reason=(
                    "unavailable (closed by scenario)"
                    if poi_id in set(
                        state.unavailable_poi_ids
                    )
                    else "skipped (no feasible bounded repair)"
                ),
            )
            for poi_id in state.original_remaining_poi_ids
        ),
        replacements=(),
        failure_status="infeasible",
        failure_reason=failure_reason,
        bounded_repair_limit=bounded_repair_limit,
        actual_modification_count=0,
    )


# ================================================================
# Adaptive chromosome decoder
# ================================================================

def decode_adaptive_chromosome(
    data: PlanningData,
    problem: AdaptiveProblem,
    chromosome: tuple[int, ...],
) -> EvaluatedSolution:
    """
    Decode an optional-POI chromosome after disruption.

    The mandatory backbone is inserted first. Optional genes are
    scanned from left to right. Every possible insertion position
    is checked, including position zero because the fixed current
    location is not itself necessarily a visit in the repaired
    itinerary.
    """

    allowed_optional_ids = set(
        problem
        .candidate_window
        .optional_ids
    )

    if set(chromosome) != (
        allowed_optional_ids
    ):
        raise ValueError(
            "Adaptive chromosome does not contain "
            "exactly the optional candidate genes."
        )

    if len(chromosome) != len(
        set(chromosome)
    ):
        raise ValueError(
            "Adaptive chromosome contains "
            "duplicate genes."
        )

    current_route = list(
        problem.mandatory_backbone
    )

    current_result = (
        decode_replanned_route(
            data=data,
            state=problem.state,
            scenario_instance=(
                problem.scenario_instance
            ),
            route=current_route,
            constraints=problem.constraints,
        )
    )

    if not current_result.feasible:
        raise RuntimeError(
            "The adaptive mandatory backbone "
            "is infeasible."
        )

    modification_limit = (
        problem.bounded_repair_config.max_modifiable_suffix_activities
    )

    for candidate_poi_id in chromosome:
        if len(current_route) >= modification_limit:
            break
        feasible_insertions: list[
            tuple[
                float,
                float,
                tuple[int, ...],
                ScheduleResult,
            ]
        ] = []

        for insertion_position in range(
            0,
            len(current_route) + 1,
        ):
            test_route = (
                current_route[
                    :insertion_position
                ]
                + [candidate_poi_id]
                + current_route[
                    insertion_position:
                ]
            )

            test_result = (
                decode_replanned_route(
                    data=data,
                    state=problem.state,
                    scenario_instance=(
                        problem
                        .scenario_instance
                    ),
                    route=test_route,
                    constraints=problem.constraints,
                )
            )

            if not test_result.feasible:
                continue

            travel_increase = (
                test_result
                .total_travel_min
                - current_result
                .total_travel_min
            )

            feasible_insertions.append(
                (
                    travel_increase,
                    test_result.finish_min,
                    tuple(test_route),
                    test_result,
                )
            )

        if not feasible_insertions:
            continue

        feasible_insertions.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2],
            )
        )

        selected = (
            feasible_insertions[0]
        )

        current_route = list(
            selected[2]
        )

        current_result = selected[3]

    repaired_utility = sum(
        data.pois[
            poi_id
        ].utility_score
        for poi_id in current_route
    )

    total_utility = (
        problem.locked_utility
        + repaired_utility
    )

    total_completed_count = (
        problem.locked_completed_count
        + len(current_route)
    )

    total_executed_travel = (
        problem.locked_travel_min
        + current_result.total_travel_min
    )

    return EvaluatedSolution(
        chromosome=chromosome,
        route=tuple(current_route),

        schedule=current_result,

        achieved_utility=(
            total_utility
        ),

        completed_poi_count=(
            total_completed_count
        ),

        total_travel_min=(
            total_executed_travel
        ),
    )


# ================================================================
# Adaptive population initialisation
# ================================================================

def build_adaptive_utility_chromosome(
    problem: AdaptiveProblem,
) -> tuple[int, ...]:
    score_lookup = dict(
        problem
        .candidate_window
        .prescreen_scores
    )

    return tuple(
        sorted(
            problem
            .candidate_window
            .optional_ids,
            key=lambda poi_id: (
                -score_lookup[poi_id],
                poi_id,
            ),
        )
    )


def build_adaptive_nearest_chromosome(
    data: PlanningData,
    problem: AdaptiveProblem,
) -> tuple[int, ...]:
    remaining_ids = set(
        problem
        .candidate_window
        .optional_ids
    )

    ordered_ids: list[int] = []

    current_poi_id = (
        problem.state.current_poi_id
    )

    adjustments = get_travel_adjustments(
        data,
        problem.scenario_instance.scenario,
    )

    while remaining_ids:
        def travel_to(
            poi_id: int,
        ) -> float:
            if poi_id == current_poi_id:
                return 0.0

            base = data.get_travel_time(
                current_poi_id,
                poi_id,
            )

            return adjustments.apply(
                current_poi_id,
                poi_id,
                base,
            )

        selected_poi_id = min(
            remaining_ids,
            key=lambda poi_id: (
                travel_to(poi_id),
                -data.pois[
                    poi_id
                ].utility_score,
                poi_id,
            ),
        )

        ordered_ids.append(
            selected_poi_id
        )

        remaining_ids.remove(
            selected_poi_id
        )

        current_poi_id = (
            selected_poi_id
        )

    return tuple(ordered_ids)


def initialise_adaptive_population(
    data: PlanningData,
    problem: AdaptiveProblem,
    config: GAConfig,
    random_generator: random.Random,
) -> list[tuple[int, ...]]:
    """
    Create the 40-chromosome bounded-replanning population.

    Duplicate chromosomes are permitted when the candidate window
    contains too few optional POIs to create 40 unique permutations.
    """

    optional_ids = list(
        problem
        .candidate_window
        .optional_ids
    )

    utility_chromosome = (
        build_adaptive_utility_chromosome(
            problem
        )
    )

    nearest_chromosome = (
        build_adaptive_nearest_chromosome(
            data=data,
            problem=problem,
        )
    )

    chromosomes = [
        utility_chromosome,
        nearest_chromosome,
    ]

    while len(chromosomes) < (
        config.population_size
    ):
        chromosomes.append(
            tuple(
                random_generator.sample(
                    optional_ids,
                    len(optional_ids),
                )
            )
        )

    return chromosomes[
        :config.population_size
    ]


# ================================================================
# Adaptive GA execution
# ================================================================

def run_adaptive_without_thermal_ga(
    data: PlanningData,
    problem: AdaptiveProblem,
    *,
    seed: int,
    config: GAConfig | None = None,
) -> AdaptiveGARunResult:
    """
    Run bounded GA repair without WES.

    Objective vector:

    - maximise total achieved utility;
    - maximise total completed visits;
    - minimise total executed travel time.
    """

    if config is None:
        config = REPLANNING_GA_CONFIG

    random_generator = (
        random.Random(seed)
    )

    evaluation_cache: dict[
        tuple[int, ...],
        EvaluatedSolution,
    ] = {}

    observed_routes: set[
        tuple[int, ...]
    ] = set()

    route_first_seen: dict[
        tuple[int, ...],
        int,
    ] = {}

    def evaluate(
        chromosome: tuple[int, ...],
        generation: int,
    ) -> Individual:
        if chromosome not in (
            evaluation_cache
        ):
            evaluation_cache[
                chromosome
            ] = (
                decode_adaptive_chromosome(
                    data=data,
                    problem=problem,
                    chromosome=(
                        chromosome
                    ),
                )
            )

        solution = (
            evaluation_cache[
                chromosome
            ]
        )

        observed_routes.add(
            solution.route
        )

        route_first_seen.setdefault(
            solution.route,
            generation,
        )

        return Individual(
            solution=solution
        )

    initial_chromosomes = (
        initialise_adaptive_population(
            data=data,
            problem=problem,
            config=config,
            random_generator=(
                random_generator
            ),
        )
    )

    population = [
        evaluate(
            chromosome,
            generation=0,
        )
        for chromosome in (
            initial_chromosomes
        )
    ]

    rank_population(
        population
    )

    selected_solution = (
        select_final_solution(
            population
        )
    )

    previous_selected_route = (
        selected_solution.route
    )

    stagnant_generations = 0
    generations_executed = 0
    stopped_by_stagnation = False

    for generation in range(
        1,
        config.maximum_generations + 1,
    ):
        generations_executed = (
            generation
        )

        sorted_population = sorted(
            population,
            key=lambda individual: (
                individual.pareto_rank,
                -individual.crowding_distance,
                -individual
                .solution
                .achieved_utility,
                -individual
                .solution
                .completed_poi_count,
                individual
                .solution
                .total_travel_min,
                individual
                .solution
                .route,
            ),
        )

        next_chromosomes = [
            individual
            .solution
            .chromosome
            for individual in (
                sorted_population[
                    :config.elitism_count
                ]
            )
        ]

        while len(next_chromosomes) < (
            config.population_size
        ):
            first_parent = (
                tournament_select(
                    population=population,
                    random_generator=(
                        random_generator
                    ),
                    tournament_size=(
                        config
                        .tournament_size
                    ),
                )
            )

            second_parent = (
                tournament_select(
                    population=population,
                    random_generator=(
                        random_generator
                    ),
                    tournament_size=(
                        config
                        .tournament_size
                    ),
                )
            )

            first_chromosome = (
                first_parent
                .solution
                .chromosome
            )

            second_chromosome = (
                second_parent
                .solution
                .chromosome
            )

            if (
                random_generator.random()
                < config
                .crossover_probability
            ):
                (
                    first_child,
                    second_child,
                ) = order_crossover(
                    first_chromosome,
                    second_chromosome,
                    random_generator,
                )

            else:
                first_child = (
                    first_chromosome
                )

                second_child = (
                    second_chromosome
                )

            if (
                random_generator.random()
                < config
                .mutation_probability
            ):
                first_child = (
                    swap_mutation(
                        first_child,
                        random_generator,
                    )
                )

            if (
                random_generator.random()
                < config
                .mutation_probability
            ):
                second_child = (
                    swap_mutation(
                        second_child,
                        random_generator,
                    )
                )

            next_chromosomes.append(
                first_child
            )

            if len(next_chromosomes) < (
                config.population_size
            ):
                next_chromosomes.append(
                    second_child
                )

        population = [
            evaluate(
                chromosome,
                generation=generation,
            )
            for chromosome in (
                next_chromosomes
            )
        ]

        rank_population(
            population
        )

        selected_solution = (
            select_final_solution(
                population
            )
        )

        if (
            selected_solution.route
            == previous_selected_route
        ):
            stagnant_generations += 1

        else:
            previous_selected_route = (
                selected_solution.route
            )

            stagnant_generations = 0

        if stagnant_generations >= (
            config.stagnation_limit
        ):
            stopped_by_stagnation = (
                True
            )

            break

    final_solution = (
        select_final_solution(
            population
        )
    )

    final_first_front = tuple(
        individual.solution
        for individual in population
        if individual.pareto_rank == 0
    )

    convergence_generation = (
        route_first_seen[
            final_solution.route
        ]
    )

    repair_record = build_repair_record(
        state=problem.state,
        problem=problem,
        repaired_route=final_solution.route,
    )

    return AdaptiveGARunResult(
        seed=seed,
        problem=problem,

        selected_solution=(
            final_solution
        ),

        generations_executed=(
            generations_executed
        ),

        convergence_generation=(
            convergence_generation
        ),

        stopped_by_stagnation=(
            stopped_by_stagnation
        ),

        evaluated_chromosome_count=len(
            evaluation_cache
        ),

        unique_decoded_route_count=len(
            observed_routes
        ),

        first_front=final_first_front,

        repair_record=repair_record,
    )
