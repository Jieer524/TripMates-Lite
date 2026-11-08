"""
Shared Genetic Algorithm engine for TripMates Lite.

This module implements:

1. Exact enumeration of the mandatory-POI backbone
2. Optional-POI permutation chromosomes
3. Deterministic feasibility decoding
4. NSGA-style Pareto ranking
5. Crowding-distance calculation
6. Tournament selection
7. Order crossover
8. Swap mutation
9. Elitism
10. Early stopping
11. Final route selection using the frozen guardrail rule

The same implementation will later be reused by:

- Static Utility
- Adaptive Without Thermal
- TripMates Lite
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import math
import random
from typing import Iterable

from planning_core import (
    PlanningConstraints,
    PlanningData,
    ScheduleResult,
    decode_route,
)


# ================================================================
# GA configuration
# ================================================================

@dataclass(frozen=True)
class GAConfig:
    population_size: int = 80
    maximum_generations: int = 120

    tournament_size: int = 3

    crossover_probability: float = 0.90
    mutation_probability: float = 0.20

    elitism_count: int = 2
    stagnation_limit: int = 15


@dataclass(frozen=True)
class EvaluatedSolution:
    chromosome: tuple[int, ...]
    route: tuple[int, ...]

    schedule: ScheduleResult

    achieved_utility: float
    completed_poi_count: int
    total_travel_min: float


@dataclass
class Individual:
    solution: EvaluatedSolution

    pareto_rank: int = 0
    crowding_distance: float = 0.0


@dataclass(frozen=True)
class GARunResult:
    seed: int
    selected_solution: EvaluatedSolution

    generations_executed: int
    convergence_generation: int
    stopped_by_stagnation: bool

    feasible_solution_count: int
    unique_route_count: int


# ================================================================
# Mandatory backbone
# ================================================================

def build_mandatory_backbone(
    data: PlanningData,
    *,
    constraints: PlanningConstraints | None = None,
) -> tuple[int, ...]:
    """
    Enumerate all orderings of mandatory POIs except the fixed origin.

    The selected mandatory backbone has:

    1. lowest travel time;
    2. earliest finish time;
    3. lexicographically smaller route as the final tie-breaker.
    """

    active_constraints = constraints or PlanningConstraints()
    start_poi_id = active_constraints.start_poi_id
    mandatory_ids = set(data.mandatory_ids)

    if start_poi_id not in mandatory_ids:
        raise ValueError(
            f"The starting POI {start_poi_id} must be mandatory."
        )

    remaining_mandatory = sorted(
        mandatory_ids - {start_poi_id}
    )

    feasible_backbones: list[
        tuple[
            float,
            float,
            tuple[int, ...],
        ]
    ] = []

    for ordering in permutations(
        remaining_mandatory
    ):
        route = (
            start_poi_id,
            *ordering,
        )

        result = decode_route(
            data=data,
            visit_sequence=route,

            start_poi_id=start_poi_id,
            start_time_min=active_constraints.start_time_min,
            trip_end_min=active_constraints.trip_end_min,

            day_code=active_constraints.day_code,
            cost_budget_rm=active_constraints.cost_budget_rm,

            start_poi_is_visit=True,

            required_mandatory_ids=(
                mandatory_ids
            ),
        )

        if not result.feasible:
            continue

        feasible_backbones.append(
            (
                result.total_travel_min,
                result.finish_min,
                route,
            )
        )

    if not feasible_backbones:
        raise RuntimeError(
            "No feasible ordering exists for the mandatory POIs."
        )

    feasible_backbones.sort()

    return feasible_backbones[0][2]


# ================================================================
# Utility helpers
# ================================================================

def calculate_route_utility(
    data: PlanningData,
    route: Iterable[int],
) -> float:
    return sum(
        data.pois[poi_id].utility_score
        for poi_id in route
    )


def evaluate_initial_route_summary(
    data: PlanningData,
    route: tuple[int, ...],
    constraints: PlanningConstraints,
) -> tuple[float, float] | None:
    """Return travel and finish time without constructing a schedule."""

    current_poi_id = constraints.start_poi_id
    current_time = float(constraints.start_time_min)
    current_cost = 0.0
    total_travel = 0.0

    for sequence_index, poi_id in enumerate(route):
        poi = data.pois[poi_id]
        opening_window = poi.get_opening_window(constraints.day_code)
        if opening_window is None:
            return None

        if sequence_index == 0:
            if poi_id != constraints.start_poi_id:
                return None
            travel_min = 0.0
        else:
            travel_min = data.get_travel_time(current_poi_id, poi_id)

        arrival_min = current_time + travel_min
        opening_min, closing_min = opening_window
        visit_start_min = max(arrival_min, float(opening_min))
        visit_end_min = visit_start_min + poi.visit_duration_min
        proposed_cost = current_cost + poi.admission_cost_rm

        if visit_start_min >= closing_min:
            return None
        if visit_end_min > closing_min + 1e-9:
            return None
        if visit_end_min > constraints.trip_end_min + 1e-9:
            return None
        if proposed_cost > constraints.cost_budget_rm + 1e-9:
            return None

        total_travel += travel_min
        current_time = visit_end_min
        current_cost = proposed_cost
        current_poi_id = poi_id

    return total_travel, current_time


# ================================================================
# Chromosome decoder
# ================================================================

def decode_chromosome(
    data: PlanningData,
    chromosome: tuple[int, ...],
    mandatory_backbone: tuple[int, ...],
    constraints: PlanningConstraints | None = None,
) -> EvaluatedSolution:
    """
    Decode an optional-POI permutation.

    The decoder starts with the mandatory backbone. It scans genes
    from left to right. For each optional POI, every insertion
    position after the origin is tested.

    A feasible insertion is selected using:

    1. smallest increase in total walking time;
    2. earliest route finish time;
    3. lexicographically smaller route.

    A gene is skipped when no feasible insertion exists.
    """

    active_constraints = constraints or PlanningConstraints()

    current_route = list(
        mandatory_backbone
    )

    current_result = decode_route(
        data=data,
        visit_sequence=current_route,

        start_poi_id=active_constraints.start_poi_id,
        start_time_min=active_constraints.start_time_min,
        trip_end_min=active_constraints.trip_end_min,

        day_code=active_constraints.day_code,
        cost_budget_rm=active_constraints.cost_budget_rm,

        start_poi_is_visit=True,

        required_mandatory_ids=(
            data.mandatory_ids
        ),
    )

    if not current_result.feasible:
        raise RuntimeError(
            "The mandatory backbone is infeasible."
        )

    for candidate_poi_id in chromosome:
        if candidate_poi_id in current_route:
            continue

        feasible_insertions: list[
            tuple[
                float,
                float,
                tuple[int, ...],
            ]
        ] = []

        # Position 0 is reserved for the fixed origin.
        for insertion_position in range(
            1,
            len(current_route) + 1,
        ):
            test_route = tuple(
                current_route[:insertion_position]
                + [candidate_poi_id]
                + current_route[insertion_position:]
            )
            summary = evaluate_initial_route_summary(
                data,
                test_route,
                active_constraints,
            )
            if summary is None:
                continue

            total_travel_min, finish_min = summary

            travel_increase = (
                total_travel_min
                - current_result.total_travel_min
            )

            feasible_insertions.append(
                (
                    travel_increase,
                    finish_min,
                    test_route,
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

        selected_insertion = (
            feasible_insertions[0]
        )

        current_route = list(
            selected_insertion[2]
        )

        current_result = decode_route(
            data=data,
            visit_sequence=current_route,
            start_poi_id=active_constraints.start_poi_id,
            start_time_min=active_constraints.start_time_min,
            trip_end_min=active_constraints.trip_end_min,
            day_code=active_constraints.day_code,
            cost_budget_rm=active_constraints.cost_budget_rm,
            start_poi_is_visit=True,
            required_mandatory_ids=data.mandatory_ids,
        )

    achieved_utility = (
        calculate_route_utility(
            data=data,
            route=current_route,
        )
    )

    return EvaluatedSolution(
        chromosome=chromosome,
        route=tuple(current_route),

        schedule=current_result,

        achieved_utility=(
            achieved_utility
        ),

        completed_poi_count=len(
            current_route
        ),

        total_travel_min=(
            current_result.total_travel_min
        ),
    )


# ================================================================
# Pareto dominance
# ================================================================

def dominates(
    first: EvaluatedSolution,
    second: EvaluatedSolution,
) -> bool:
    """
    Non-thermal objective vector:

    - maximise achieved utility;
    - maximise completed POI count;
    - minimise total travel time.
    """

    no_worse = (
        first.achieved_utility
        >= second.achieved_utility
        and first.completed_poi_count
        >= second.completed_poi_count
        and first.total_travel_min
        <= second.total_travel_min
    )

    strictly_better = (
        first.achieved_utility
        > second.achieved_utility
        or first.completed_poi_count
        > second.completed_poi_count
        or first.total_travel_min
        < second.total_travel_min
    )

    return no_worse and strictly_better


def assign_pareto_ranks(
    population: list[Individual],
) -> list[list[int]]:
    """
    Perform non-dominated sorting.

    Returns fronts containing population indices.
    """

    population_size = len(population)

    dominated_indices: list[list[int]] = [
        []
        for _ in range(population_size)
    ]

    domination_count = [
        0
        for _ in range(population_size)
    ]

    first_front: list[int] = []

    for first_index in range(
        population_size
    ):
        for second_index in range(
            population_size
        ):
            if first_index == second_index:
                continue

            first_solution = (
                population[first_index]
                .solution
            )

            second_solution = (
                population[second_index]
                .solution
            )

            if dominates(
                first_solution,
                second_solution,
            ):
                dominated_indices[
                    first_index
                ].append(second_index)

            elif dominates(
                second_solution,
                first_solution,
            ):
                domination_count[
                    first_index
                ] += 1

        if domination_count[first_index] == 0:
            population[
                first_index
            ].pareto_rank = 0

            first_front.append(
                first_index
            )

    fronts: list[list[int]] = []

    if first_front:
        fronts.append(first_front)

    current_front_index = 0

    while current_front_index < len(fronts):
        next_front: list[int] = []

        for first_index in fronts[
            current_front_index
        ]:
            for dominated_index in (
                dominated_indices[
                    first_index
                ]
            ):
                domination_count[
                    dominated_index
                ] -= 1

                if domination_count[
                    dominated_index
                ] == 0:
                    population[
                        dominated_index
                    ].pareto_rank = (
                        current_front_index
                        + 1
                    )

                    next_front.append(
                        dominated_index
                    )

        if next_front:
            fronts.append(next_front)

        current_front_index += 1

    return fronts


# ================================================================
# Crowding distance
# ================================================================

def assign_crowding_distance(
    population: list[Individual],
    front_indices: list[int],
) -> None:
    for index in front_indices:
        population[
            index
        ].crowding_distance = 0.0

    if len(front_indices) <= 2:
        for index in front_indices:
            population[
                index
            ].crowding_distance = (
                math.inf
            )

        return

    objective_functions = [
        lambda individual: (
            individual
            .solution
            .achieved_utility
        ),

        lambda individual: (
            individual
            .solution
            .completed_poi_count
        ),

        lambda individual: (
            individual
            .solution
            .total_travel_min
        ),
    ]

    for objective_function in (
        objective_functions
    ):
        sorted_indices = sorted(
            front_indices,
            key=lambda index: (
                objective_function(
                    population[index]
                )
            ),
        )

        minimum_value = (
            objective_function(
                population[
                    sorted_indices[0]
                ]
            )
        )

        maximum_value = (
            objective_function(
                population[
                    sorted_indices[-1]
                ]
            )
        )

        population[
            sorted_indices[0]
        ].crowding_distance = math.inf

        population[
            sorted_indices[-1]
        ].crowding_distance = math.inf

        if math.isclose(
            maximum_value,
            minimum_value,
            abs_tol=1e-12,
        ):
            continue

        denominator = (
            maximum_value
            - minimum_value
        )

        for position in range(
            1,
            len(sorted_indices) - 1,
        ):
            current_index = (
                sorted_indices[position]
            )

            if math.isinf(
                population[
                    current_index
                ].crowding_distance
            ):
                continue

            previous_value = (
                objective_function(
                    population[
                        sorted_indices[
                            position - 1
                        ]
                    ]
                )
            )

            next_value = (
                objective_function(
                    population[
                        sorted_indices[
                            position + 1
                        ]
                    ]
                )
            )

            population[
                current_index
            ].crowding_distance += (
                next_value
                - previous_value
            ) / denominator


def rank_population(
    population: list[Individual],
) -> None:
    fronts = assign_pareto_ranks(
        population
    )

    for front_indices in fronts:
        assign_crowding_distance(
            population,
            front_indices,
        )


# ================================================================
# Final route selection
# ================================================================

def select_final_solution(
    population: list[Individual],
) -> EvaluatedSolution:
    """
    Apply the frozen final route-selection rule.

    First Pareto front only.

    Let:
        U* = maximum utility on the front
        N* = maximum completed-POI count on the front

    Eligible routes satisfy:
        utility >= 0.90 U*
        count >= N* - 1

    The selected route has:
        1. lowest travel time;
        2. higher utility;
        3. lexicographically smaller POI sequence.
    """

    first_front = [
        individual.solution
        for individual in population
        if individual.pareto_rank == 0
    ]

    if not first_front:
        raise RuntimeError(
            "The population has no first Pareto front."
        )

    maximum_utility = max(
        solution.achieved_utility
        for solution in first_front
    )

    maximum_count = max(
        solution.completed_poi_count
        for solution in first_front
    )

    eligible_solutions = [
        solution
        for solution in first_front
        if (
            solution.achieved_utility
            >= 0.90 * maximum_utility
            and solution.completed_poi_count
            >= maximum_count - 1
        )
    ]

    if not eligible_solutions:
        raise RuntimeError(
            "No solution passed the final guardrail rule."
        )

    return min(
        eligible_solutions,
        key=lambda solution: (
            solution.total_travel_min,
            -solution.achieved_utility,
            solution.route,
        ),
    )


# ================================================================
# Initial population
# ================================================================

def build_utility_efficiency_chromosome(
    data: PlanningData,
    optional_ids: list[int],
    start_poi_id: int,
) -> tuple[int, ...]:
    """
    Rank optional POIs by:

        utility /
        (visit duration + travel from origin)
    """

    return tuple(
        sorted(
            optional_ids,
            key=lambda poi_id: (
                -(
                    data.pois[
                        poi_id
                    ].utility_score
                    /
                    (
                        data.pois[
                            poi_id
                        ].visit_duration_min
                        + data.get_travel_time(
                            start_poi_id,
                            poi_id,
                        )
                    )
                ),
                poi_id,
            ),
        )
    )


def build_nearest_neighbour_chromosome(
    data: PlanningData,
    optional_ids: list[int],
    start_poi_id: int,
) -> tuple[int, ...]:
    remaining_ids = set(
        optional_ids
    )

    ordered_ids: list[int] = []
    current_poi_id = start_poi_id

    while remaining_ids:
        selected_poi_id = min(
            remaining_ids,
            key=lambda poi_id: (
                data.get_travel_time(
                    current_poi_id,
                    poi_id,
                ),
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


def initialise_population(
    data: PlanningData,
    config: GAConfig,
    random_generator: random.Random,
    start_poi_id: int = 1,
) -> list[tuple[int, ...]]:
    optional_ids = sorted(
        set(data.pois)
        - set(data.mandatory_ids)
    )

    utility_seed = (
        build_utility_efficiency_chromosome(
            data,
            optional_ids,
            start_poi_id,
        )
    )

    nearest_seed = (
        build_nearest_neighbour_chromosome(
            data,
            optional_ids,
            start_poi_id,
        )
    )

    chromosomes: list[
        tuple[int, ...]
    ] = []

    seen: set[
        tuple[int, ...]
    ] = set()

    def add_chromosome(
        chromosome: tuple[int, ...],
    ) -> None:
        if chromosome not in seen:
            chromosomes.append(
                chromosome
            )
            seen.add(chromosome)

    add_chromosome(
        utility_seed
    )

    add_chromosome(
        nearest_seed
    )

    # A population can never hold more distinct chromosomes than the
    # optional POIs admit.  Without this cap a catalogue with few
    # optional POIs (for example the small synthetic fixture used by
    # the test suite) makes the fill loop spin forever, because every
    # permutation is already in ``seen``.  The 50-POI research
    # catalogue has 47 optional POIs, so the cap is never reached there
    # and the frozen benchmark behaviour is unchanged.
    distinct_limit = (
        math.factorial(len(optional_ids))
        if optional_ids
        else 1
    )

    target_size = min(
        config.population_size,
        distinct_limit,
    )

    while len(chromosomes) < target_size:
        random_chromosome = tuple(
            random_generator.sample(
                optional_ids,
                len(optional_ids),
            )
        )

        add_chromosome(
            random_chromosome
        )

    return chromosomes


# ================================================================
# Genetic operators
# ================================================================

def tournament_select(
    population: list[Individual],
    random_generator: random.Random,
    tournament_size: int,
) -> Individual:
    competitors = (
        random_generator.sample(
            population,
            tournament_size,
        )
    )

    return min(
        competitors,
        key=lambda individual: (
            individual.pareto_rank,
            -individual.crowding_distance,
            -individual.solution.achieved_utility,
            -individual.solution.completed_poi_count,
            individual.solution.total_travel_min,
            individual.solution.route,
        ),
    )


def order_crossover(
    first_parent: tuple[int, ...],
    second_parent: tuple[int, ...],
    random_generator: random.Random,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
]:
    chromosome_length = len(
        first_parent
    )

    if chromosome_length < 2:
        return first_parent, second_parent

    start_index, end_index = sorted(
        random_generator.sample(
            range(chromosome_length),
            2,
        )
    )

    # Use an inclusive segment.
    end_index += 1

    def build_child(
        segment_parent: tuple[int, ...],
        fill_parent: tuple[int, ...],
    ) -> tuple[int, ...]:
        child: list[int | None] = [
            None
            for _ in range(
                chromosome_length
            )
        ]

        child[
            start_index:end_index
        ] = segment_parent[
            start_index:end_index
        ]

        used_genes = {
            gene
            for gene in child
            if gene is not None
        }

        fill_genes = [
            gene
            for gene in fill_parent
            if gene not in used_genes
        ]

        fill_position = 0

        for index in range(
            chromosome_length
        ):
            if child[index] is None:
                child[index] = (
                    fill_genes[
                        fill_position
                    ]
                )

                fill_position += 1

        return tuple(
            int(gene)
            for gene in child
        )

    first_child = build_child(
        first_parent,
        second_parent,
    )

    second_child = build_child(
        second_parent,
        first_parent,
    )

    return first_child, second_child


def swap_mutation(
    chromosome: tuple[int, ...],
    random_generator: random.Random,
) -> tuple[int, ...]:
    if len(chromosome) < 2:
        return chromosome

    first_index, second_index = (
        random_generator.sample(
            range(len(chromosome)),
            2,
        )
    )

    mutated = list(chromosome)

    mutated[first_index], mutated[second_index] = (
        mutated[second_index],
        mutated[first_index],
    )

    return tuple(mutated)


# ================================================================
# GA execution
# ================================================================

def run_static_utility_ga(
    data: PlanningData,
    *,
    seed: int,
    config: GAConfig | None = None,
    constraints: PlanningConstraints | None = None,
) -> GARunResult:
    if config is None:
        config = GAConfig()

    active_constraints = constraints or PlanningConstraints()

    random_generator = (
        random.Random(seed)
    )

    mandatory_backbone = (
        build_mandatory_backbone(
            data=data,
            constraints=active_constraints,
        )
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
            ] = decode_chromosome(
                data=data,
                chromosome=chromosome,
                mandatory_backbone=(
                    mandatory_backbone
                ),
                constraints=active_constraints,
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
        initialise_population(
            data=data,
            config=config,
            random_generator=(
                random_generator
            ),
            start_poi_id=active_constraints.start_poi_id,
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

    rank_population(population)

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
                -individual.solution.achieved_utility,
                -individual.solution.completed_poi_count,
                individual.solution.total_travel_min,
                individual.solution.route,
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

        rank_population(population)

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

    return GARunResult(
        seed=seed,

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

        feasible_solution_count=len(
            evaluation_cache
        ),

        unique_route_count=len(
            observed_routes
        ),
    )
