"""Run the controlled experimental-gap studies.

Usage: python run_gap_experiments.py --study all [--smoke]

The study definitions and shared execution contract live in
``ablation_common.py``.  This module is deliberately orchestration-only.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import asdict

import pandas as pd

from ablation_common import (
    ALL_PLANNERS, DENSITY_LEVELS, NOMINAL_INSTANCE_IDS, RESULTS_ROOT,
    SEVERITY_LEVELS, STUDY_DENSITY, STUDY_FULL_REPLANNING, STUDY_SEVERITY,
    STUDY_WEATHER, build_density_levels, build_full_replanning_state,
    build_manifest, load_study_inputs, now_iso, run_cell, seeds_for_run,
    study_directory, tag_records, weather_level, resolve_weather_profile,
    write_study_outputs,
)
from scenario_core import ScenarioInstance, ScenarioDefinition


def _scenario(base, *, scenario_id, thermal=None):
    definition = ScenarioDefinition(
        scenario_id=scenario_id, name=base.scenario.name,
        nominal_trigger_offset_min=base.scenario.nominal_trigger_offset_min,
        thermal_multiplier=(base.scenario.thermal_multiplier if thermal is None else thermal),
        global_travel_multiplier=base.scenario.global_travel_multiplier,
        closed_poi_ids=base.scenario.closed_poi_ids,
        congested_zone=base.scenario.congested_zone,
        congestion_multiplier=base.scenario.congestion_multiplier,
        purpose=base.scenario.purpose,
    )
    return ScenarioInstance(definition, "nominal", base.trigger_offset_min, base.trigger_absolute_min)


def run_study(study: str, smoke: bool) -> None:
    inputs = load_study_inputs()
    seeds = seeds_for_run(smoke)
    instances = ("S1_nominal",) if smoke else NOMINAL_INSTANCE_IDS
    rows = []
    started = time.perf_counter()

    if study == STUDY_SEVERITY:
        cells = [(x.cell_id, x.to_scenario(), x.to_dict()) for x in SEVERITY_LEVELS]
        factor_columns = ["family", "level", "cell_id"]
    elif study == STUDY_WEATHER:
        cells = []
        base = inputs.scenario_lookup["S1_nominal"]
        for level in (weather_level("normal"), weather_level("hot"), weather_level("very_hot"), weather_level("afternoon_escalation"), weather_level("missing_segment")):
            _, provenance = resolve_weather_profile(level)
            cells.append((level.level, _scenario(base, scenario_id=f"weather_{level.level}", thermal=level.thermal_multiplier), {**asdict(level), **provenance}))
        factor_columns = ["weather_level"]
    elif study == STUDY_DENSITY:
        factor_columns = ["density_level"]
        cells = [(level, None, {"density_level": level}) for level in DENSITY_LEVELS]
    elif study == STUDY_FULL_REPLANNING:
        factor_columns = ["replanning_scope"]
        cells = [("bounded_suffix", None, {"replanning_scope": "bounded_suffix", "replanning_policy": "bounded_suffix"}), ("full_itinerary", None, {"replanning_scope": "full_itinerary", "replanning_policy": "full_replanning_ablation"})]
    else:
        raise ValueError(study)

    for seed in seeds:
        base_route = inputs.route(seed)
        base_schedule = inputs.schedule(seed)
        for instance_id in instances:
            base_instance = inputs.scenario_lookup[instance_id]
            for cell, custom_scenario, factor in cells:
                scenario = custom_scenario or base_instance
                route, schedule = base_route, base_schedule
                state_override = None
                if study == STUDY_DENSITY:
                    density = build_density_levels(data=inputs.data, base_route=base_route, constraints=inputs.constraints)[cell]
                    route, schedule = density["route"], density["schedule"]
                if study == STUDY_FULL_REPLANNING and cell == "full_itinerary":
                    from scenario_core import simulate_replanning_state
                    state = simulate_replanning_state(data=inputs.data, initial_schedule=schedule, scenario_instance=base_instance)
                    state_override = build_full_replanning_state(data=inputs.data, state=state, initial_schedule=schedule)
                planners = ("Adaptive Without Thermal", "TripMates Lite") if state_override is not None else ALL_PLANNERS
                cell_rows = run_cell(data=inputs.data, constraints=inputs.constraints, initial_route=route, initial_schedule=schedule, scenario_instance=scenario, seed=seed, planners=planners, state_override=state_override)
                rows.extend(tag_records(cell_rows, **factor))

    frame = pd.DataFrame(rows)
    factor_spec = {"levels": [x[2] for x in cells], "identical_inputs": True, "study_started_utc": now_iso()}
    manifest = build_manifest(study=study, smoke=smoke, factors=factor_spec, seeds=seeds, planners=ALL_PLANNERS, run_count=len(frame), failure_count=int((~frame["feasible"].astype(bool)).sum()) if not frame.empty else 0, total_runtime_seconds=time.perf_counter()-started, extra={"source_script": "run_gap_experiments.py", "instances": list(instances), "output_root": str(RESULTS_ROOT)})
    files = write_study_outputs(study=study, smoke=smoke, frame=frame, factor_columns=factor_columns, manifest=manifest)
    print(f"{study}: {len(frame)} rows -> {files['directory'].resolve()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", choices=["all", STUDY_FULL_REPLANNING, STUDY_SEVERITY, STUDY_DENSITY, STUDY_WEATHER], default="all")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    studies = [STUDY_FULL_REPLANNING, STUDY_SEVERITY, STUDY_DENSITY, STUDY_WEATHER] if args.study == "all" else [args.study]
    for study in studies:
        run_study(study, args.smoke)


if __name__ == "__main__":
    main()
