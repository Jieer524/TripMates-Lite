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
    BoundedRepairConfig,
    REPLANNING_GA_CONFIG,
    RepairRecord,
    build_repair_record,
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
class ThermalSelectionPolicy:
    """Configurable guardrail policy for thermal final selection.

    The frozen research policy is ``alpha=0.90`` with a visit margin of
    one, measured against the seed-matched non-thermal reference.  That
    exact combination is the default so existing callers reproduce the
    original result byte-for-byte.

    C2 sensitivity runs vary ``alpha`` and use the task-specified
    selection priority.  They never replace the frozen policy.
    """

    alpha: float = 0.90
    visit_guardrail_margin: int = 1

    # One of:
    #   "non_thermal_reference"        (frozen policy)
    #   "front_maximum"
    #   "front_maximum_plus_reference" (C2 default)
    utility_guardrail_baseline: str = (
        "non_thermal_reference"
    )

    # One of: "frozen", "c2"
    priority: str = "frozen"

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(
                "Utility-retention alpha must be in [0, 1]."
            )

        if self.visit_guardrail_margin < 0:
            raise ValueError(
                "Visit-guardrail margin cannot be negative."
            )

        valid_baselines = {
            "non_thermal_reference",
            "front_maximum",
            "front_maximum_plus_reference",
        }

        if (
            self.utility_guardrail_baseline
            not in valid_baselines
        ):
            raise ValueError(
                "Unknown utility-guardrail baseline: "
                f"{self.utility_guardrail_baseline!r}"
            )

        if self.priority not in {"frozen", "c2"}:
            raise ValueError(
                f"Unknown selection priority: "
                f"{self.priority!r}"
            )


DEFAULT_THERMAL_SELECTION_POLICY = (
    ThermalSelectionPolicy()
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
    guardrail_fallback_used: bool
    selection_status: str

    # ------------------------------------------------------------
    # Selection-policy audit trail (populated for the frozen policy
    # too, so the Pareto front can always be exported).
    # ------------------------------------------------------------

    # Policy used for the *final* route selection that produced
    # ``selected_solution``.  C2 exploratory runs record the requested
    # alpha here; the frozen research runs leave it at the default.
    final_selection_policy: ThermalSelectionPolicy = (
        DEFAULT_THERMAL_SELECTION_POLICY
    )

    # Explicit, alpha-independent policy that drives every *internal*
    # search decision: the generation-zero progress reference, the
    # in-loop stagnation comparison and the convergence generation.
    # It is always the frozen default, so the GA trajectory, the
    # stagnation stop, the convergence generation and the evaluated
    # Pareto front are identical across C2 alpha settings.
    search_selection_policy: ThermalSelectionPolicy = (
        DEFAULT_THERMAL_SELECTION_POLICY
    )

    first_front: tuple[
        ThermalEvaluatedSolution,
        ...,
    ] = ()

    front_utility_guardrail: tuple[bool, ...] = ()
    front_visit_guardrail: tuple[bool, ...] = ()
    front_guardrail_eligible: tuple[bool, ...] = ()
    front_selected: tuple[bool, ...] = ()

    u_star: float = 0.0
    reference_utility: float = 0.0
    front_max_utility: float = 0.0

    repair_record: RepairRecord | None = None

    @property
    def selection_policy(
        self,
    ) -> ThermalSelectionPolicy:
        """Backward-compatible alias for the final selection policy."""

        return self.final_selection_policy


@dataclass(frozen=True)
class ThermalSelectionOutcome:
    """Full, auditable record of one thermal final selection."""

    selected_solution: ThermalEvaluatedSolution
    guardrail_fallback_used: bool
    selection_status: str

    first_front: tuple[
        ThermalEvaluatedSolution,
        ...,
    ]

    utility_guardrail_passed: tuple[bool, ...]
    visit_guardrail_passed: tuple[bool, ...]
    guardrail_eligible: tuple[bool, ...]

    u_star: float
    reference_utility: float
    front_max_utility: float

    policy: ThermalSelectionPolicy


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

def _frozen_selection_key(
    solution: ThermalEvaluatedSolution,
) -> tuple:
    return (
        solution.wes,
        -solution.achieved_utility,
        solution.exposure.total_outdoor_exposure_min,
        solution.total_travel_min,
        solution.route,
    )


def _c2_selection_key(
    solution: ThermalEvaluatedSolution,
) -> tuple:
    return (
        solution.wes,
        solution.total_travel_min,
        -solution.achieved_utility,
        solution.route,
    )


def evaluate_thermal_selection(
    population: list[Individual],
    non_thermal_reference: ThermalEvaluatedSolution,
    policy: ThermalSelectionPolicy | None = None,
) -> ThermalSelectionOutcome:
    """Return the full thermal final-selection record.

    Eligibility is measured against a configurable utility-retention
    threshold ``alpha`` and a configurable visit-guardrail margin.

    ``U*`` is the highest achieved utility among the feasible
    alternatives considered for selection.  Its definition depends on
    ``policy.utility_guardrail_baseline``:

    - ``non_thermal_reference``: U* is the seed-matched non-thermal
      reference utility (the frozen contract).
    - ``front_maximum``: U* is the maximum utility on the first front.
    - ``front_maximum_plus_reference``: U* is the larger of the two.

    A route passes the utility guardrail when
    ``route.utility >= alpha * U*`` and the visit guardrail when
    ``route.completed_poi_count >= reference.count - margin``.

    If no first-front alternative passes both guardrails, the feasible
    non-thermal reference is returned and a fallback is reported.
    """

    active_policy = (
        policy or DEFAULT_THERMAL_SELECTION_POLICY
    )

    first_front = [
        individual.solution
        for individual in population
        if individual.pareto_rank == 0
    ]

    if not first_front:
        raise RuntimeError(
            "No first thermal Pareto front exists."
        )

    reference_utility = (
        non_thermal_reference.achieved_utility
    )

    front_max_utility = max(
        solution.achieved_utility
        for solution in first_front
    )

    if (
        active_policy.utility_guardrail_baseline
        == "non_thermal_reference"
    ):
        u_star = reference_utility

    elif (
        active_policy.utility_guardrail_baseline
        == "front_maximum"
    ):
        u_star = front_max_utility

    else:
        u_star = max(
            front_max_utility,
            reference_utility,
        )

    utility_passed = tuple(
        solution.achieved_utility
        >= active_policy.alpha * u_star
        for solution in first_front
    )

    visit_passed = tuple(
        solution.completed_poi_count
        >= non_thermal_reference.completed_poi_count
        - active_policy.visit_guardrail_margin
        for solution in first_front
    )

    eligible = tuple(
        utility_ok and visit_ok
        for utility_ok, visit_ok in zip(
            utility_passed,
            visit_passed,
        )
    )

    eligible_solutions = [
        solution
        for solution, is_eligible in zip(
            first_front,
            eligible,
        )
        if is_eligible
    ]

    selection_key = (
        _c2_selection_key
        if active_policy.priority == "c2"
        else _frozen_selection_key
    )

    if not eligible_solutions:
        return ThermalSelectionOutcome(
            selected_solution=(
                non_thermal_reference
            ),
            guardrail_fallback_used=True,
            selection_status=(
                "guardrail_fallback"
            ),
            first_front=tuple(first_front),
            utility_guardrail_passed=(
                utility_passed
            ),
            visit_guardrail_passed=(
                visit_passed
            ),
            guardrail_eligible=eligible,
            u_star=u_star,
            reference_utility=(
                reference_utility
            ),
            front_max_utility=(
                front_max_utility
            ),
            policy=active_policy,
        )

    selected_solution = min(
        eligible_solutions,
        key=selection_key,
    )

    return ThermalSelectionOutcome(
        selected_solution=selected_solution,
        guardrail_fallback_used=False,
        selection_status="thermal_admissible",
        first_front=tuple(first_front),
        utility_guardrail_passed=utility_passed,
        visit_guardrail_passed=visit_passed,
        guardrail_eligible=eligible,
        u_star=u_star,
        reference_utility=reference_utility,
        front_max_utility=front_max_utility,
        policy=active_policy,
    )


def select_thermal_solution(
    population: list[Individual],
    non_thermal_reference: ThermalEvaluatedSolution,
    policy: ThermalSelectionPolicy | None = None,
) -> tuple[ThermalEvaluatedSolution, bool]:
    """Backward-compatible wrapper around the auditable selector.

    Uses the first Pareto front.  Eligibility is measured against the
    seed-matched non-thermal reference route from the same disruption
    state and candidate window:

        AU >= alpha * U*
        N  >= reference count - margin

    Final priority (frozen policy):
        1. lowest WES
        2. higher achieved utility
        3. lower TOEM
        4. lower executed travel
        5. lower POI-ID sequence

    If no non-dominated thermal alternative passes both guardrails,
    return the feasible non-thermal reference and report a fallback.
    """

    outcome = evaluate_thermal_selection(
        population,
        non_thermal_reference,
        policy,
    )

    return (
        outcome.selected_solution,
        outcome.guardrail_fallback_used,
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
    selection_policy: ThermalSelectionPolicy | None = None,
) -> TripMatesGARunResult:
    if config is None:
        config = (
            REPLANNING_GA_CONFIG
        )

    final_selection_policy = (
        selection_policy
        or DEFAULT_THERMAL_SELECTION_POLICY
    )

    # The internal search must not depend on the requested alpha: only
    # the final route selection may vary.  Every internal GA progress
    # and stagnation comparison therefore uses one explicit, frozen,
    # alpha-independent policy.
    search_selection_policy = (
        DEFAULT_THERMAL_SELECTION_POLICY
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

        repair_record = build_repair_record(
            state=problem.state,
            problem=problem,
            repaired_route=(
                base_solution.route
            ),
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
            guardrail_fallback_used=False,
            selection_status="thermal_inactive",
            final_selection_policy=(
                final_selection_policy
            ),
            search_selection_policy=(
                search_selection_policy
            ),
            first_front=(thermal_solution,),
            front_utility_guardrail=(True,),
            front_visit_guardrail=(True,),
            front_guardrail_eligible=(True,),
            front_selected=(True,),
            u_star=thermal_solution.achieved_utility,
            reference_utility=(
                thermal_solution.achieved_utility
            ),
            front_max_utility=(
                thermal_solution.achieved_utility
            ),
            repair_record=repair_record,
        )

    # ------------------------------------------------------------
    # Thermal objective active
    # ------------------------------------------------------------

    non_thermal_result = (
        run_adaptive_without_thermal_ga(
            data=data,
            problem=problem,
            seed=seed,
            config=config,
        )
    )

    non_thermal_base_solution = (
        non_thermal_result.selected_solution
    )

    non_thermal_reference = ThermalEvaluatedSolution(
        chromosome=non_thermal_base_solution.chromosome,
        route=non_thermal_base_solution.route,
        schedule=non_thermal_base_solution.schedule,
        achieved_utility=(
            non_thermal_base_solution.achieved_utility
        ),
        completed_poi_count=(
            non_thermal_base_solution.completed_poi_count
        ),
        total_travel_min=(
            non_thermal_base_solution.total_travel_min
        ),
        exposure=calculate_exposure_metrics(
            data=data,
            initial_schedule=initial_schedule,
            state=problem.state,
            repaired_schedule=(
                non_thermal_base_solution.schedule
            ),
            thermal_multiplier=thermal_multiplier,
        ),
    )

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

    # Generation-zero progress reference.  Uses the alpha-independent
    # search policy so the stagnation baseline never depends on alpha.
    search_selected_solution = (
        evaluate_thermal_selection(
            population,
            non_thermal_reference,
            search_selection_policy,
        )
        .selected_solution
    )

    previous_selected_route = (
        search_selected_solution.route
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

        # Stagnation uses the explicit alpha-independent search policy.
        # The alpha-sensitive final selection policy is deliberately
        # NOT passed into this call.
        (
            search_selected_solution,
            _guardrail_fallback_used,
        ) = select_thermal_solution(
            population,
            non_thermal_reference,
            search_selection_policy,
        )

        if (
            search_selected_solution.route
            == previous_selected_route
        ):
            stagnant_generations += 1

        else:
            previous_selected_route = (
                search_selected_solution.route
            )

            stagnant_generations = 0

        if stagnant_generations >= (
            config.stagnation_limit
        ):
            stopped_by_stagnation = (
                True
            )

            break

    # Final selection may use the requested alpha; the search above is
    # unaffected by it.
    final_outcome = evaluate_thermal_selection(
        population,
        non_thermal_reference,
        final_selection_policy,
    )

    final_solution = (
        final_outcome.selected_solution
    )

    guardrail_fallback_used = (
        final_outcome.guardrail_fallback_used
    )

    selected_route = final_solution.route

    front_selected = tuple(
        solution.route == selected_route
        for solution in final_outcome.first_front
    )

    repair_record = build_repair_record(
        state=problem.state,
        problem=problem,
        repaired_route=selected_route,
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
            # Convergence is an alpha-independent search property: it
            # is the generation at which the search policy's selected
            # route first appeared, not the alpha-selected route.
            route_first_seen[
                search_selected_solution.route
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
        guardrail_fallback_used=(
            guardrail_fallback_used
        ),
        selection_status=(
            "guardrail_fallback"
            if guardrail_fallback_used
            else "thermal_admissible"
        ),
        final_selection_policy=(
            final_selection_policy
        ),
        search_selection_policy=(
            search_selection_policy
        ),
        first_front=(
            final_outcome.first_front
        ),
        front_utility_guardrail=(
            final_outcome
            .utility_guardrail_passed
        ),
        front_visit_guardrail=(
            final_outcome
            .visit_guardrail_passed
        ),
        front_guardrail_eligible=(
            final_outcome.guardrail_eligible
        ),
        front_selected=front_selected,
        u_star=final_outcome.u_star,
        reference_utility=(
            final_outcome.reference_utility
        ),
        front_max_utility=(
            final_outcome.front_max_utility
        ),
        repair_record=repair_record,
    )
