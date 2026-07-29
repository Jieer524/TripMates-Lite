"""
Shared bounded-replanning Genetic Algorithm.

This module is shared by:

1. Adaptive Without Thermal
2. TripMates Lite

Both planners use exactly the same:

- candidate window;
- initial populations;
- chromosome representation;
- decoder;
- GA parameters;
- feasibility rules;
- random seeds.

TripMates Lite will later differ only through WES Pareto ranking
and thermal final-route selection.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    COST_BUDGET_RM,
    EXPERIMENT_DAY,
    TRIP_END_MIN,
    PlanningData,
    ScheduleResult,
    decode_route,
)

from scenario_core import (
    ReplanningState,
    ScenarioInstance,
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


# ================================================================
# Shared route decoding
# ================================================================

def decode_replanned_route(
    data: PlanningData,
    state: ReplanningState,
    scenario_instance: ScenarioInstance,
    route: list[int] | tuple[int, ...],
) -> ScheduleResult:
    """
    Decode a route beginning from the post-trigger state.

    Completed costs are supplied as initial_cost_rm.

    Post-trigger travel changes are applied only to the repaired
    route. A completed or committed pre-trigger component is not
    recalculated using disruption multipliers.
    """

    travel_adjustments = (
        scenario_instance
        .scenario
        .build_travel_adjustments(data)
    )

    return decode_route(
        data=data,
        visit_sequence=route,

        start_poi_id=state.current_poi_id,
        start_time_min=state.replan_start_min,
        trip_end_min=TRIP_END_MIN,

        day_code=EXPERIMENT_DAY,
        cost_budget_rm=COST_BUDGET_RM,
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
    Shared non-thermal pre-screen score:

        score_j = U_j / (T_ij + V_j)

    This same score must be used by both adaptive planners so that
    WES remains the only thermal ablation difference.
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

        adjustments = (
            scenario_instance
            .scenario
            .build_travel_adjustments(data)
        )

        travel_min = adjustments.apply(
            state.current_poi_id,
            candidate_poi_id,
            base_travel_min,
        )

    denominator = (
        travel_min
        + poi.visit_duration_min
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
) -> CandidateWindow:
    """
    Build a maximum-10-POI candidate window.

    Rules:

    1. Exclude completed and unavailable POIs.
    2. Include all remaining mandatory POIs first.
    3. Exclude candidates that cannot coexist with all remaining
       mandatory POIs.
    4. Fill remaining positions using U / (travel + visit).
    5. Use lower POI ID as the final deterministic tie-breaker.
    """

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

    if len(mandatory_ids) > (
        MAXIMUM_CANDIDATE_COUNT
    ):
        raise RuntimeError(
            "The number of remaining mandatory POIs "
            "exceeds the candidate-window limit."
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

    remaining_slots = (
        MAXIMUM_CANDIDATE_COUNT
        - len(mandatory_ids)
    )

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

    if len(all_candidate_ids) > (
        MAXIMUM_CANDIDATE_COUNT
    ):
        raise RuntimeError(
            "Candidate window exceeded "
            "the maximum size."
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
) -> AdaptiveProblem:
    mandatory_backbone = (
        build_remaining_mandatory_backbone(
            data=data,
            state=state,
            scenario_instance=(
                scenario_instance
            ),
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
        )
    )

    if not current_result.feasible:
        raise RuntimeError(
            "The adaptive mandatory backbone "
            "is infeasible."
        )

    for candidate_poi_id in chromosome:
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

    adjustments = (
        problem
        .scenario_instance
        .scenario
        .build_travel_adjustments(data)
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

    convergence_generation = (
        route_first_seen[
            final_solution.route
        ]
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
    )