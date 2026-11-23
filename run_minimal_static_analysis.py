from pathlib import Path
from itertools import combinations
import pandas as pd

import run_rq1_full_benchmark as bench
from benchmark_common import StaticSuffixEvaluation, build_initial_schedules, load_initial_routes, route_utility
from planning_core import PlanningConstraints, decode_route, load_planning_data
from adaptive_ga_core import get_travel_adjustments

def minimal_edit_evaluator(*, data, state, scenario_instance, constraints):
    """Exact minimum-deletion fixed-order executor for the finite suffix."""
    adjustments = get_travel_adjustments(data, scenario_instance.scenario)
    original = tuple(state.original_remaining_poi_ids)
    unavailable = set(state.unavailable_poi_ids)
    candidates = []
    for keep_count in range(len(original), -1, -1):
        for kept in combinations(original, keep_count):
            if unavailable.intersection(kept):
                continue
            result = decode_route(
                data=data, visit_sequence=kept,
                start_poi_id=state.current_poi_id,
                start_time_min=state.replan_start_min,
                trip_end_min=constraints.trip_end_min,
                day_code=constraints.day_code,
                cost_budget_rm=constraints.cost_budget_rm,
                initial_cost_rm=state.cost_spent_rm,
                start_poi_is_visit=False,
                required_mandatory_ids=state.remaining_mandatory_ids,
                unavailable_poi_ids=state.unavailable_poi_ids,
                travel_adjustments=adjustments,
            )
            if result.feasible:
                utility = sum(data.pois[x].utility_score for x in kept)
                candidates.append((keep_count, utility, -result.total_travel_min, tuple(kept), result))
        if candidates:
            break
    if not candidates:
        return StaticSuffixEvaluation(False, "No feasible subsequence under minimum-deletion rule", tuple(), None, 0, tuple())
    _, _, _, route, final = max(candidates, key=lambda x: (x[0], x[1], x[2], tuple(-v for v in x[3])))
    skipped = tuple({"poi_id": int(x), "reason": "minimum-deletion fixed-order rule"} for x in original if x not in route)
    return StaticSuffixEvaluation(True, None, route, final, len(final.completed_sequence), skipped)

def main():
    data = load_planning_data(poi_file=bench.POI_FILE, matrix_file=bench.MATRIX_FILE)
    constraints = PlanningConstraints()
    routes = load_initial_routes()
    schedules = build_initial_schedules(data, routes)
    lookup = bench.build_scenario_lookup()
    initial_utils = {seed: route_utility(data, route) for seed, route in routes.items()}
    bench.evaluate_static_suffix_skip_and_continue = minimal_edit_evaluator
    rows=[]
    for iid in bench.ALL_INSTANCE_IDS:
        sc=lookup[iid]
        # Static Distance is included so the same minimum-deletion rule is
        # applied to every fixed-order baseline.
        if iid == bench.ALL_INSTANCE_IDS[0]:
            static_distance_route = bench.build_static_distance_route(data=data, start_poi_id=1)
            static_distance_schedule = decode_route(
                data=data, visit_sequence=static_distance_route,
                start_poi_id=1, start_time_min=constraints.start_time_min,
                trip_end_min=constraints.trip_end_min, day_code=constraints.day_code,
                cost_budget_rm=constraints.cost_budget_rm, start_poi_is_visit=True,
                required_mandatory_ids=data.mandatory_ids)
            static_distance_utility = route_utility(data, static_distance_route)
        rows.append(bench.static_planner_record(
            planner=bench.PLANNER_STATIC_DISTANCE, data=data, scenario_instance=sc,
            seed=None, initial_route=static_distance_route, initial_schedule=static_distance_schedule,
            initial_route_utility=static_distance_utility, constraints=constraints,
            deterministic=True, static_execution_mode="planned_skip_and_continue_execution"))
        for seed in bench.SEEDS:
            rows.append(bench.static_planner_record(
                planner=bench.PLANNER_STATIC_UTILITY, data=data, scenario_instance=sc,
                seed=seed, initial_route=routes[seed], initial_schedule=schedules[seed],
                initial_route_utility=initial_utils[seed], constraints=constraints,
                deterministic=True, static_execution_mode="planned_skip_and_continue_execution"))
    df=pd.DataFrame(rows)
    # static_planner_record dispatches through the skip-and-continue mode while
    # the evaluator above is temporarily replaced. Record the rule actually
    # used so downstream comparisons cannot confuse the two static protocols.
    df["static_execution_mode"] = "minimum_deletion_fixed_order_execution"
    out=Path("results/rq1_minimal_edit_static")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out/"minimal_static_utility_raw_runs.csv", index=False)
    print(df.groupby('scenario_id').agg(ICR=('itinerary_completion_ratio','mean'), OPR=('original_poi_retention','mean'), utility=('achieved_utility','mean'), travel=('total_travel_min','mean'), cost=('total_cost_rm','mean'), WES=('wes','mean')).to_string())
    print('\nOVERALL')
    print(df[['itinerary_completion_ratio','original_poi_retention','achieved_utility','total_travel_min','total_cost_rm','wes']].mean().to_string())
    print('\nFEASIBLE',df.feasible.mean())

if __name__ == '__main__': main()
