"""Category D - planner contracts.

These tests pin the properties the four planners promise to share:
one candidate window, one decoder, one thermal guardrail, an
alpha-independent search, reproducible baselines, isolated smoke output
and seeded determinism.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import run_latency_benchmark as latency_runner
import run_rq1_full_benchmark as rq1_runner
import run_thermal_tradeoff_sensitivity as c2_runner
from adaptive_ga_core import (
    REPLANNING_GA_CONFIG,
    decode_replanned_route,
    prepare_adaptive_problem,
    run_adaptive_without_thermal_ga,
)
from exposure_core import ExposureMetrics
from ga_core import GAConfig, Individual, run_static_utility_ga
from planning_core import decode_route
from scenario_core import simulate_replanning_state
from tripmates_lite_ga import (
    ThermalEvaluatedSolution,
    ThermalSelectionPolicy,
    run_tripmates_lite_ga,
    select_thermal_solution,
)

from conftest import make_scenario


SEED = 1001

# A small, deterministic budget.  The contract tests care about the
# shared search space and the shared decoder, not about solution
# quality, so a short budget keeps the suite fast.
SMALL_GA_CONFIG = GAConfig(
    population_size=10,
    maximum_generations=4,
    tournament_size=3,
    crossover_probability=0.90,
    mutation_probability=0.20,
    elitism_count=2,
    stagnation_limit=3,
)


# ================================================================
# Shared problem fixture
# ================================================================


@pytest.fixture()
def thermal_problem(planning_data, initial_schedule):
    """A post-disruption problem with the thermal objective active."""

    instance = make_scenario(
        trigger_offset_min=60,
        thermal_multiplier=1.8,
        closed_poi_ids=frozenset({3}),
    )

    state = simulate_replanning_state(
        data=planning_data,
        initial_schedule=initial_schedule,
        scenario_instance=instance,
    )

    problem = prepare_adaptive_problem(
        data=planning_data,
        initial_schedule=initial_schedule,
        state=state,
        scenario_instance=instance,
    )

    return problem, instance


# ================================================================
# D1. Both adaptive planners search the same candidate window
# ================================================================


def test_adaptive_and_tripmates_share_the_candidate_window(
    planning_data,
    initial_schedule,
    thermal_problem,
):
    problem, _ = thermal_problem

    adaptive = run_adaptive_without_thermal_ga(
        data=planning_data,
        problem=problem,
        seed=SEED,
        config=SMALL_GA_CONFIG,
    )

    tripmates = run_tripmates_lite_ga(
        data=planning_data,
        initial_schedule=initial_schedule,
        problem=problem,
        seed=SEED,
        config=SMALL_GA_CONFIG,
    )

    # Both planners consume the exact same prepared problem object.
    assert adaptive.problem is problem
    assert tripmates.problem is problem

    assert (
        adaptive.problem.candidate_window
        == tripmates.problem.candidate_window
    )

    window = set(problem.candidate_window.all_candidate_ids)
    assert window, "The candidate window must not be empty."

    # Every selected route is drawn from that window.
    assert set(adaptive.selected_solution.route) <= window
    assert set(tripmates.selected_solution.route) <= window

    # The bounded-repair policy is identical for both planners.
    assert (
        adaptive.problem.bounded_repair_config
        == tripmates.problem.bounded_repair_config
    )


# ================================================================
# D2. Both adaptive planners use the shared decoder
# ================================================================


def test_adaptive_and_tripmates_share_the_decoder(
    planning_data,
    initial_schedule,
    thermal_problem,
):
    problem, instance = thermal_problem

    results = (
        run_adaptive_without_thermal_ga(
            data=planning_data,
            problem=problem,
            seed=SEED,
            config=SMALL_GA_CONFIG,
        ),
        run_tripmates_lite_ga(
            data=planning_data,
            initial_schedule=initial_schedule,
            problem=problem,
            seed=SEED,
            config=SMALL_GA_CONFIG,
        ),
    )

    for result in results:
        solution = result.selected_solution

        redecoded = decode_replanned_route(
            data=planning_data,
            state=problem.state,
            scenario_instance=instance,
            route=solution.route,
            constraints=problem.constraints,
        )

        # Byte-for-byte identical schedule: same travel, same waiting,
        # same costs, same failure verdict.
        assert redecoded == solution.schedule

        assert redecoded.feasible is True
        assert redecoded.missing_mandatory_ids == ()


# ================================================================
# D3. Thermal selection respects both guardrails
# ================================================================


def _thermal_solution(
    *,
    route: tuple[int, ...],
    utility: float,
    visit_count: int,
    travel_min: float,
    wes: float,
) -> ThermalEvaluatedSolution:
    exposure = ExposureMetrics(
        prefix_walking_min=0.0,
        prefix_visit_min=0.0,
        repaired_walking_min=0.0,
        repaired_visit_min=0.0,
        outdoor_walking_min=0.0,
        outdoor_visit_min=0.0,
        total_outdoor_exposure_min=wes,
        weather_weighted_exposure_score=wes,
        thermal_multiplier=1.8,
    )

    return ThermalEvaluatedSolution(
        chromosome=route,
        route=route,
        schedule=None,  # type: ignore[arg-type]
        achieved_utility=utility,
        completed_poi_count=visit_count,
        total_travel_min=travel_min,
        exposure=exposure,
    )


def test_thermal_selection_respects_utility_and_visit_guardrails():
    reference = _thermal_solution(
        route=(1, 2, 3, 4, 5),
        utility=100.0,
        visit_count=5,
        travel_min=30.0,
        wes=120.0,
    )

    # Passes both guardrails and has a lower WES: it must be selected.
    eligible = _thermal_solution(
        route=(1, 10, 11, 12, 13),
        utility=95.0,
        visit_count=4,
        travel_min=25.0,
        wes=100.0,
    )

    # Fails the utility guardrail (89 < 0.90 * 100): must be rejected
    # even though its WES is by far the lowest.
    ineligible = _thermal_solution(
        route=(1, 14, 15),
        utility=89.0,
        visit_count=3,
        travel_min=5.0,
        wes=1.0,
    )

    selected, fallback = select_thermal_solution(
        [Individual(solution=ineligible), Individual(solution=eligible)],
        reference,
    )

    assert fallback is False
    assert selected is eligible

    # With only an ineligible candidate the selector falls back to the
    # non-thermal reference instead of violating the guardrail.
    selected, fallback = select_thermal_solution(
        [Individual(solution=ineligible)],
        reference,
    )

    assert fallback is True
    assert selected is reference


# ================================================================
# D4. Alpha changes selection, never the search
# ================================================================


def test_alpha_changes_selection_but_not_the_candidate_search(
    planning_data,
    initial_schedule,
    thermal_problem,
):
    problem, _ = thermal_problem

    frozen = run_tripmates_lite_ga(
        data=planning_data,
        initial_schedule=initial_schedule,
        problem=problem,
        seed=SEED,
        config=REPLANNING_GA_CONFIG,
        selection_policy=ThermalSelectionPolicy(alpha=0.90),
    )

    relaxed = run_tripmates_lite_ga(
        data=planning_data,
        initial_schedule=initial_schedule,
        problem=problem,
        seed=SEED,
        config=REPLANNING_GA_CONFIG,
        selection_policy=ThermalSelectionPolicy(alpha=0.85),
    )

    # The requested alpha is recorded, so the two runs are distinguishable.
    assert frozen.final_selection_policy.alpha == pytest.approx(0.90)
    assert relaxed.final_selection_policy.alpha == pytest.approx(0.85)

    # The internal search policy is always the frozen one.
    assert frozen.search_selection_policy == relaxed.search_selection_policy

    # The search itself is byte-identical.
    assert frozen.evaluated_chromosome_count == relaxed.evaluated_chromosome_count
    assert frozen.unique_decoded_route_count == relaxed.unique_decoded_route_count
    assert frozen.generations_executed == relaxed.generations_executed
    assert frozen.convergence_generation == relaxed.convergence_generation
    assert frozen.stopped_by_stagnation == relaxed.stopped_by_stagnation

    # ... and so is the Pareto front it produced.
    assert [solution.route for solution in frozen.first_front] == [
        solution.route for solution in relaxed.first_front
    ]

    # Only the final selection may differ, and it must stay inside the
    # front that both runs explored.
    front_routes = {solution.route for solution in frozen.first_front}
    assert relaxed.selected_solution.route in front_routes


# ================================================================
# D5. Static baseline reproducibility
# ================================================================


def test_static_baseline_is_reproducible(planning_data):
    first = run_static_utility_ga(
        data=planning_data,
        seed=SEED,
        config=SMALL_GA_CONFIG,
    )

    second = run_static_utility_ga(
        data=planning_data,
        seed=SEED,
        config=SMALL_GA_CONFIG,
    )

    assert first.selected_solution.route == second.selected_solution.route
    assert first.selected_solution.achieved_utility == pytest.approx(
        second.selected_solution.achieved_utility
    )
    assert first.selected_solution.total_travel_min == pytest.approx(
        second.selected_solution.total_travel_min
    )
    assert first.generations_executed == second.generations_executed
    assert first.convergence_generation == second.convergence_generation

    # The baseline never leaves the shared hard constraints.
    schedule = first.selected_solution.schedule
    assert schedule.feasible is True
    assert schedule.missing_mandatory_ids == ()
    assert schedule.total_cost_rm <= 160.0 + 1e-9
    assert schedule.finish_min <= 17 * 60 + 1e-9


# ================================================================
# D6. Smoke runs never overwrite official output
# ================================================================


RUNNERS = (
    pytest.param(latency_runner, id="latency"),
    pytest.param(rq1_runner, id="rq1"),
    pytest.param(c2_runner, id="c2"),
)


@pytest.mark.parametrize("module", RUNNERS)
def test_smoke_outputs_are_isolated_from_official_outputs(
    module,
    monkeypatch,
    tmp_path,
):
    """``--smoke`` must redirect every artefact into ``<out>/smoke/``."""

    # The set of module-level constants the isolator rewrites, read
    # straight from the function body so the test cannot drift out of
    # sync with the implementation.
    rewritten_names = [
        name
        for name in module.isolate_smoke_outputs.__code__.co_names
        if name.isupper()
        and name != "OUTPUT_DIRECTORY"
        and hasattr(module, name)
    ]

    assert rewritten_names, (
        f"{module.__name__}.isolate_smoke_outputs rewrites no constants."
    )

    sandbox = tmp_path / module.__name__
    sandbox.mkdir(parents=True, exist_ok=True)

    original_files: dict[str, Path] = {}

    for name in rewritten_names:
        original = Path(getattr(module, name))
        original_files[name] = original

        monkeypatch.setattr(module, name, sandbox / original.name)

    monkeypatch.setattr(module, "OUTPUT_DIRECTORY", sandbox)

    module.isolate_smoke_outputs()

    smoke_directory = Path(module.OUTPUT_DIRECTORY)

    assert smoke_directory == sandbox / "smoke"
    assert smoke_directory.is_dir()

    for name, original in original_files.items():
        redirected = Path(getattr(module, name))

        assert redirected.parent == smoke_directory, (
            f"{name} was not redirected into the smoke folder."
        )
        assert redirected.name == original.name, (
            f"{name} changed its file name during isolation."
        )


# ================================================================
# D7. Seeded determinism
# ================================================================


def test_seeded_runs_are_deterministic(
    planning_data,
    initial_schedule,
    thermal_problem,
):
    problem, _ = thermal_problem

    def run_adaptive():
        return run_adaptive_without_thermal_ga(
            data=planning_data,
            problem=problem,
            seed=SEED,
            config=SMALL_GA_CONFIG,
        )

    first = run_adaptive()
    second = run_adaptive()

    assert first.selected_solution.route == second.selected_solution.route
    assert first.selected_solution.achieved_utility == pytest.approx(
        second.selected_solution.achieved_utility
    )
    assert first.evaluated_chromosome_count == second.evaluated_chromosome_count
    assert first.generations_executed == second.generations_executed

    def run_thermal():
        return run_tripmates_lite_ga(
            data=planning_data,
            initial_schedule=initial_schedule,
            problem=problem,
            seed=SEED,
            config=SMALL_GA_CONFIG,
        )

    thermal_first = run_thermal()
    thermal_second = run_thermal()

    assert (
        thermal_first.selected_solution.route
        == thermal_second.selected_solution.route
    )
    assert thermal_first.u_star == pytest.approx(thermal_second.u_star)
    assert (
        thermal_first.guardrail_fallback_used
        == thermal_second.guardrail_fallback_used
    )


def test_redecoding_a_route_is_idempotent(planning_data, initial_schedule):
    """The decoder is a pure function of its inputs."""

    instance = make_scenario(trigger_offset_min=60)

    state = simulate_replanning_state(
        data=planning_data,
        initial_schedule=initial_schedule,
        scenario_instance=instance,
    )

    first = decode_replanned_route(
        data=planning_data,
        state=state,
        scenario_instance=instance,
        route=(4, 5),
    )

    second = decode_replanned_route(
        data=planning_data,
        state=state,
        scenario_instance=instance,
        route=(4, 5),
    )

    assert first == second
    assert first.feasible is True
