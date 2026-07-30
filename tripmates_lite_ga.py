"""
Thermal exposure-aware bounded replanning for TripMates Lite.

For M_s = 1, this module directly uses Adaptive Without Thermal so
that the two planners are guaranteed to produce identical outputs.

For M_s > 1, WES is added as a fourth Pareto objective:

- maximise achieved utility;
- maximise completed visits;
- minimise executed travel time;
- minimise WES.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

from adaptive_ga_core import (
    AdaptiveGARunResult,
    AdaptiveProblem,
    REPLANNING_GA_CONFIG,
    decode_adaptive_chromosome,
    initialise_adaptive_population,
    run_adaptive_without_thermal_ga,
)

from exposure_core import (
    ExposureMetrics,
    calculate_exposure_metrics,
)

from ga_core import (
    GAConfig,
    Individual,
    order_crossover,
    swap_mutation,
    tournament_select,
)

from planning_core import (
    PlanningData,
    ScheduleResult,
)


# ================================================================
# Thermal solution structures
# ================================================================

@dataclass(frozen=True)
class ThermalEvaluatedSolution:
    chromosome: tuple[int, ...]
    route: tuple[int, ...]

    schedule: ScheduleResult

    achieved_utility: float
    completed_poi_count: int
    total_travel_min: float

    exposure: ExposureMetrics

    @property
    def wes(self) -> float:
        return (
            self.exposure
            .weather_weighted_exposure_score
        )


@dataclass(frozen=True)
class TripMatesGARunResult:
    seed: int
    problem: AdaptiveProblem

    selected_solution: (
        ThermalEvaluatedSolution
    )

    generations_executed: int
    convergence_generation: int
    stopped_by_stagnation: bool

    evaluated_chromosome_count: int
    unique_decoded_route_count: int

    thermal_objective_active: bool


# ================================================================
# Thermal Pareto dominance
# ================================================================

def thermal_dominates(
    first: ThermalEvaluatedSolution,
    second: ThermalEvaluatedSolution,
) -> bool:
    no_worse = (
        first.achieved_utility
        >= second.achieved_utility
        and first.completed_poi_count
        >= second.completed_poi_count
        and first.total_travel_min
        <= second.total_travel_min
        and first.wes
        <= second.wes
    )

    strictly_better = (
        first.achieved_utility
        > second.achieved_utility
        or first.completed_poi_count
        > second.completed_poi_count
        or first.total_travel_min
        < second.total_travel_min
        or first.wes
        < second.wes
    )

    return (
        no_worse
        and strictly_better
    )


def assign_thermal_pareto_ranks(
    population: list[Individual],
) -> list[list[int]]:
    population_size = len(
        population
    )

    dominated_indices = [
        []
        for _ in range(
            population_size
        )
    ]

    domination_count = [
        0
        for _ in range(
            population_size
        )
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

            if thermal_dominates(
                first_solution,
                second_solution,
            ):
                dominated_indices[
                    first_index
                ].append(
                    second_index
                )

            elif thermal_dominates(
                second_solution,
                first_solution,
            ):
                domination_count[
                    first_index
                ] += 1

        if domination_count[
            first_index
        ] == 0:
            population[
                first_index
            ].pareto_rank = 0

            first_front.append(
                first_index
            )

    fronts: list[list[int]] = []

    if first_front:
        fronts.append(
            first_front
        )

    current_front_index = 0

    while current_front_index < len(
        fronts
    ):
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
            fronts.append(
                next_front
            )

        current_front_index += 1

    return fronts


def assign_thermal_crowding_distance(
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

        lambda individual: (
            individual
            .solution
            .wes
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
        ].crowding_distance = (
            math.inf
        )

        population[
            sorted_indices[-1]
        ].crowding_distance = (
            math.inf
        )

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
                sorted_indices[
                    position
                ]
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


def rank_thermal_population(
    population: list[Individual],
) -> None:
    fronts = (
        assign_thermal_pareto_ranks(
            population
        )
    )

    for front in fronts:
        assign_thermal_crowding_distance(
            population,
            front,
        )


# ================================================================
# Thermal final-route selection
# ================================================================

def select_thermal_solution(
    population: list[Individual],
) -> ThermalEvaluatedSolution:
    """
    Use the first Pareto front.

    Eligibility:
        AU >= 90% of maximum-front utility
        N >= maximum-front count - 1

    Final priority:
        1. lowest WES
        2. lowest executed travel
        3. higher achieved utility
        4. lower POI-ID sequence
    """

    first_front = [
        individual.solution
        for individual in population
        if individual.pareto_rank == 0
    ]

    if not first_front:
        raise RuntimeError(
            "No first thermal Pareto front exists."
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
            "No thermal solution passed "
            "the final guardrails."
        )

    return min(
        eligible_solutions,
        key=lambda solution: (
            solution.wes,
            solution.total_travel_min,
            -solution.achieved_utility,
            solution.route,
        ),
    )


# ================================================================
# Thermal GA execution
# ================================================================

def run_tripmates_lite_ga(
    data: PlanningData,
    initial_schedule: ScheduleResult,
    problem: AdaptiveProblem,
    *,
    seed: int,
    config: GAConfig | None = None,
) -> TripMatesGARunResult:
    if config is None:
        config = (
            REPLANNING_GA_CONFIG
        )

    thermal_multiplier = (
        problem
        .scenario_instance
        .scenario
        .thermal_multiplier
    )

    # ------------------------------------------------------------
    # Thermal objective inactive
    # ------------------------------------------------------------

    if math.isclose(
        thermal_multiplier,
        1.0,
        abs_tol=1e-12,
    ):

        if (
            thermal_multiplier > 1.0
            and not problem
            .thermal_aware_candidate_window
        ):
            raise ValueError(
                "TripMates Lite requires the "
                "thermal-aware candidate window "
                "when the thermal objective is active."
            )

        non_thermal_result = (
            run_adaptive_without_thermal_ga(
                data=data,
                problem=problem,
                seed=seed,
                config=config,
            )
        )

        base_solution = (
            non_thermal_result
            .selected_solution
        )

        exposure = (
            calculate_exposure_metrics(
                data=data,
                initial_schedule=(
                    initial_schedule
                ),
                state=problem.state,
                repaired_schedule=(
                    base_solution.schedule
                ),
                thermal_multiplier=1.0,
            )
        )

        thermal_solution = (
            ThermalEvaluatedSolution(
                chromosome=(
                    base_solution
                    .chromosome
                ),

                route=(
                    base_solution.route
                ),

                schedule=(
                    base_solution.schedule
                ),

                achieved_utility=(
                    base_solution
                    .achieved_utility
                ),

                completed_poi_count=(
                    base_solution
                    .completed_poi_count
                ),

                total_travel_min=(
                    base_solution
                    .total_travel_min
                ),

                exposure=exposure,
            )
        )

        return TripMatesGARunResult(
            seed=seed,
            problem=problem,

            selected_solution=(
                thermal_solution
            ),

            generations_executed=(
                non_thermal_result
                .generations_executed
            ),

            convergence_generation=(
                non_thermal_result
                .convergence_generation
            ),

            stopped_by_stagnation=(
                non_thermal_result
                .stopped_by_stagnation
            ),

            evaluated_chromosome_count=(
                non_thermal_result
                .evaluated_chromosome_count
            ),

            unique_decoded_route_count=(
                non_thermal_result
                .unique_decoded_route_count
            ),

            thermal_objective_active=False,
        )

    # ------------------------------------------------------------
    # Thermal objective active
    # ------------------------------------------------------------

    random_generator = (
        random.Random(seed)
    )

    evaluation_cache: dict[
        tuple[int, ...],
        ThermalEvaluatedSolution,
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
            base_solution = (
                decode_adaptive_chromosome(
                    data=data,
                    problem=problem,
                    chromosome=(
                        chromosome
                    ),
                )
            )

            exposure = (
                calculate_exposure_metrics(
                    data=data,
                    initial_schedule=(
                        initial_schedule
                    ),
                    state=problem.state,
                    repaired_schedule=(
                        base_solution.schedule
                    ),
                    thermal_multiplier=(
                        thermal_multiplier
                    ),
                )
            )

            evaluation_cache[
                chromosome
            ] = (
                ThermalEvaluatedSolution(
                    chromosome=(
                        base_solution
                        .chromosome
                    ),

                    route=(
                        base_solution.route
                    ),

                    schedule=(
                        base_solution.schedule
                    ),

                    achieved_utility=(
                        base_solution
                        .achieved_utility
                    ),

                    completed_poi_count=(
                        base_solution
                        .completed_poi_count
                    ),

                    total_travel_min=(
                        base_solution
                        .total_travel_min
                    ),

                    exposure=exposure,
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

    rank_thermal_population(
        population
    )

    selected_solution = (
        select_thermal_solution(
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

        rank_thermal_population(
            population
        )

        selected_solution = (
            select_thermal_solution(
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
        select_thermal_solution(
            population
        )
    )

    return TripMatesGARunResult(
        seed=seed,
        problem=problem,

        selected_solution=(
            final_solution
        ),

        generations_executed=(
            generations_executed
        ),

        convergence_generation=(
            route_first_seen[
                final_solution.route
            ]
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

        thermal_objective_active=True,
    )