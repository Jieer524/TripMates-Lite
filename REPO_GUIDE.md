# TripMates Lite — Code and Experiment Guide

> This is the **in-depth manual** for the repository: data contracts, module responsibilities, algorithm details, the design and artefacts of every experiment, how to read the results, and known issues.
> It is written for readers who need to modify code, reproduce experiments, or write the methodology chapter. If you only want to run the project, start with [README.md](README.md).

---

## Table of contents

1. [What this project is](#1-what-this-project-is)
2. [Environment and quick start](#2-environment-and-quick-start)
3. [Repository layout](#3-repository-layout)
4. [Data contracts](#4-data-contracts)
5. [Architecture and call chain](#5-architecture-and-call-chain)
6. [Shared engine](#6-shared-engine)
7. [The four planners](#7-the-four-planners)
8. [The web application](#8-the-web-application)
9. [Experiment inventory](#9-experiment-inventory)
10. [Results and interpretation](#10-results-and-interpretation)
11. [Test suite](#11-test-suite)
12. [How to extend](#12-how-to-extend)
13. [Known issues and caveats](#13-known-issues-and-caveats)
14. [Glossary](#14-glossary)

---

## 1. What this project is

TripMates Lite is a research prototype for **single-day urban walking-itinerary planning with post-disruption replanning**, using Kuala Lumpur and 50 POIs as the case study.

**Problem setting**

- A trip runs 09:00–17:00 with an admission budget of RM 160, three mandatory POIs (IDs 1 / 4 / 13), and an optional set of preferred POIs.
- The planner must produce a **feasible** walking itinerary: opening hours, budget, time horizon and mandatory visits all satisfied.
- Mid-trip, a disruption occurs (heavy rain, network delay, closure, congestion, heatwave, or a combination).
- Only the **unfinished suffix** is replanned; everything already completed is locked.

**Research question**

> If weather-weighted outdoor exposure (WES) is added as a fourth Pareto objective to replanning, can outdoor exposure be reduced without materially sacrificing itinerary quality (achieved utility, completed visits)?

**Control design** (this is the backbone of the whole repository)

| Planner | Adaptive | Stochastic | Uses thermal information |
|---|---|---|---|
| Static Distance | no | no | no |
| Static Utility | no | yes (initial route only) | no |
| Greedy Repair | yes | no | no |
| Adaptive Without Thermal | yes | yes | no |
| **TripMates Lite** | yes | yes | **yes (WES as 4th objective)** |

All adaptive planners share **the same candidate window** and **the same decoder**. Differences are only permitted in the objective function and in the final route selection — that is the precondition for a fair ablation.

---

## 2. Environment and quick start

**Dependencies** (`requirements.txt`)

```
pandas>=2.0
scipy>=1.11
streamlit>=1.40      # ships pydeck, used for the map
```

Development and validation environment: Python 3.12 / 3.13 on Windows 11. `dataset_validation_report.json` records Python 3.12.14, pandas 3.0.1, numpy 2.3.5.

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt

python static_utility_baseline.py     # required first: produces the 10 initial routes
streamlit run app.py                  # launch the web app
```

> `results/static_utility/initial_routes_by_seed.json` is the input for every downstream experiment. **Run the baseline before anything else.**

---

## 3. Repository layout

```

### Controlled experimental gaps

`run_gap_experiments.py` is the common entry point; named wrappers are provided for each study. The runners use deterministic seeds 1001–1010, matched inputs, the existing decoder, isolated output directories, and explicit failure rows. `--smoke` writes only to the corresponding `smoke/` directory.

The no-stability study is already implemented by `run_stability_ablation.py`. It reuses one final Pareto front and varies only the selection key and guardrail, so “no stability penalty” is a true selection-policy control. The full-replanning condition releases the committed prefix and is an ablation upper bound, not the proposed deployable policy. Severity uses exact factors in `ablation_common.py`: travel multipliers 1.15/1.50/2.00, closure sets {33}/{13}/{13,33}, and thermal multipliers 1.30/1.80/2.40. Density fixes the POI catalogue and matrix while using mandatory-only, frozen, and deterministic greedy-extended initial routes. Weather compares multipliers 1.0/1.8/2.4 and a missing-segment condition resolved through the documented predefined fallback; missing weather is never encoded as zero exposure.

Outputs are `results/ablation_studies/<study>/` with `raw_runs.csv`, `planner_summary.csv`, `failure_cases.csv`, and `run_config.json`; `analyze_gap_experiments.py` adds `analysis_summary.csv`, `paired_tests.csv`, and JSON metadata. The analysis reports feasibility, utility, retained POIs and percentage, WES, travel/walking time, changed activities, edit distance, runtime, failure rate, Wilcoxon paired tests, rank-biserial effects, and Benjamini–Hochberg q-values. Results are explicitly exploratory; descriptive differences are not claims of superiority. Limitations include the small fixed Kuala Lumpur dataset, ten repeated stochastic seeds, synthetic severity levels, and the infeasible physical assumption in the full-replanning upper bound.

### Thermal-exposure proxy contract

WES is an itinerary-planning proxy, not a medical, physiological, heatstroke, or health-risk score. For POI `j`, `indoor_ratio` is the whole-visit proportion expected indoors, while `shelter_ratio` is the conditional proportion of the remaining outdoor visit fraction covered or shaded by outdoor shelter. It is not a second whole-visit enclosure proportion. The exposed fraction is `X_j = (1 - indoor_ratio_j) * (1 - shelter_ratio_j)`. Walking exposure is `OWM = walking_minutes`; visit exposure is `OVM = visit_minutes * X_j`; total outdoor exposure is `TOEM = OWM + OVM`. Waiting is excluded from the primary proxy by definition, because the current model has no waiting-location exposure label. If a caller opts in, waiting exposure is `waiting_minutes * waiting_exposed_fraction` and is reported separately.

For an activity interval `[a,b]`, execution window `[s,e]`, and trigger `t`, the weighted interval is `overlap([a,b],[s,min(t,e)]) + M_s * overlap([a,b],[max(t,s),e])`. Consequently `WES = pre_trigger_exposure + M_s * post_trigger_exposure`; the multiplier never retroactively changes pre-trigger exposure. Time-of-day is represented by the schedule's absolute minute-of-day intervals and the trigger split; there is no hidden diurnal or solar model.

Weather profiles carry source provenance: `live`, `cached`, `predefined`, or `unavailable`, plus `fallback_reason`. Live profiles use weather code first for rain; otherwise temperature ≥35°C is hot, and temperature ≥32°C with relative humidity ≥70% is classified as hot-humid. If temperature or humidity is missing, the service does not assign zero exposure: it uses the documented predefined benchmark profile when supplied, or an explicit unavailable neutral profile when no fallback is supplied. External access remains isolated in `weather_service.py`; the planner has no weather dependency.

### Final reproducibility and audit package

`final_research_audit.py` validates official raw-run tables, configuration presence, duplicate run keys, required metrics, impossible values, and smoke/official separation. It writes JSON and Markdown reports under `output/research_final/`; critical failures return a non-zero exit code. `build_final_research_tables.py` creates derived comparison tables and focused figures without modifying raw results. The audit distinguishes official artefacts from smoke-only outputs and reports when a Phase 2 family has not yet received a full official run.

The completed evidence appendices are generated by `generate_evidence_appendices.py` and `analyze_thermal_alpha_decision.py`. They include route audits, failure categories and rates, the preregistered alpha decision table, utility-data provenance, and a validity-and-limitations table. These scripts are descriptive and do not tune thresholds or remove failures.

`run_rq1_full_benchmark.py` runs the originally planned skip-and-continue static rule and writes its recorded results under `results/rq1_full_benchmark_corrected_static_rule/`. The minimum-deletion primary comparison is assembled by `assemble_rq1_minimum_deletion_primary.py` from 198 separately evaluated fixed-order static rows and the 540 unchanged Greedy Repair and adaptive rows; its complete local output is under `results/rq1_full_benchmark_minimum_deletion_primary/`. This is an assembled reanalysis, not a single 738-row rerun. Use `--legacy-static` only to reproduce the archived legacy replay rule in `results/rq1_full_benchmark/`; `--corrected-static` remains a compatibility alias. `generate_latency_protocol.py` and `generate_latency_profile_summary.py` document the timing environment and cProfile bottlenecks.
TripMates-Lite/
├── poi_data_calculated_final.csv       # 50 POIs, 31 columns (utility, hours, exposure ratios)
├── walking_time_matrix_final.csv       # 50x50 symmetric walking-minute matrix
├── dataset_validation_report.json      # dataset audit report (produced by audit_checks.py)
│
├── planning_core.py                    # * data models + decode_route (the only feasibility decoder)
├── scenario_core.py                    # * six disruptions / 18 instances / trigger-state simulation
├── exposure_core.py                    # * outdoor exposure and WES
├── stability_metrics.py                # * deterministic itinerary-stability metrics + weighted edit distance
│
├── ga_core.py                          # static-utility GA + generic GA operators
├── adaptive_ga_core.py                 # bounded replanning GA + shared candidate window
├── greedy_repair.py                    # deterministic greedy repair baseline
├── tripmates_lite_ga.py                # thermal-aware GA (WES as 4th objective + guardrails)
│
├── static_utility_baseline.py          # experiment 1: 10 initial routes
├── static_distance_baseline.py         # experiment 2: distance-first baseline
├── run_weather_ablation_all_seeds.py   # experiment 3: main ablation (90 pairs)
├── analyze_weather_ablation.py         # experiment 3 statistics
├── run_rq1_full_benchmark.py           # experiment 4: RQ1 full benchmark (5 planners x 18 instances)
├── analyze_rq1_full_benchmark.py       # experiment 4 statistics
├── run_thermal_tradeoff_sensitivity.py # experiment 5: C2 alpha sensitivity (4 levels)
├── analyze_thermal_tradeoff_sensitivity.py
├── run_latency_benchmark.py            # experiment 6: replanning latency
├── audit_checks.py                     # experiment 7: dataset audit
├── diagnose_thermal_diversity.py       # diagnostic: why the thermal objective never changes the route
├── make_research_manifest.py           # provenance manifest for C2 v2
│
├── app.py                              # Streamlit front end
├── itinerary_service.py                # application service layer (validate / plan / replan)
├── weather_service.py                  # Open-Meteo adapter with three-level fallback
├── itinerary_history.py                # SQLite itinerary history
│
├── results/                            # all experiment artefacts (see section 9)
├── data/                               # runtime state: history.sqlite3, weather_profile.json
├── archive/overpass_cache/             # ARCHIVED: early Overpass/OSM responses, not used by the application
├── test_*.py                           # frozen-artefact contract / regression suites
├── tests/                              # pytest-native synthetic suite + legacy script wrappers
├── pytest.ini                          # pythonpath / markers
│
├── qa/            # [local] thesis docx proof sheets (622 PNGs)
├── documents/     # [local] four thesis drafts
├── output/        # [local] six PPT revisions + figures + imagegen assets
├── tools/         # [local] docx build scripts
├── ccfa-workfiles/# [local] presentation intermediate artefacts
└── .github/skills/# [local] academic-paper-reviewer skill
```

> `qa/`, `tools/`, `documents/`, `.github/skills/` and `reviews/` are excluded by `.gitignore`. They are local working files and are not tracked.

---

## 4. Data contracts

### 4.1 POI catalogue (`poi_data_calculated_final.csv`)

50 rows x 31 columns. `poi_id` must be exactly 1–50 in ascending order. The planner reads 17 of these columns (enforced by `load_planning_data`):

| Column | Meaning | Notes |
|---|---|---|
| `poi_id` | 1–50 | primary key |
| `name`, `category`, `zone` | name / type / zone (A/B/C) | `zone` drives the congestion scenario |
| `latitude`, `longitude` | coordinates | used by the map and by matrix verification |
| `mandatory` | `YES`/`NO` | mandatory POIs are 1, 4 and 13 |
| `off_day` | `NONE` or `MON,FRI` | closed days |
| `is_24_hours` | `YES`/`NO` | if yes, the window is `(0, 1440)` |
| `opening_time_24h`, `closing_time_24h` | `HH:MM` | converted to minutes |
| `closes_next_day` | `YES`/`NO` | if yes, `closing += 1440` |
| `visit_duration_min` | suggested visit length | fixed, never randomised |
| `admission_cost_rm` | admission fee | budget constraint |
| `utility_score` | utility, 6.41–9.80 | see 4.3 |
| `indoor_ratio`, `shelter_ratio` | whole-visit indoor share / conditional sheltered share of remaining outdoor time | exposure model inputs |

Dataset profile: zones B 27 / A 16 / C 7; 7 categories; mean `indoor_ratio` approx. 0.50; 4 POIs open 24 hours; 3 POIs have off days.

### 4.2 Walking matrix (`walking_time_matrix_final.csv`)

- 51 columns: `poi_id` plus `"1"` … `"50"` (column names must be string IDs, order matters);
- 50x50, **strictly symmetric** (`abs_tol=1e-9`), zero diagonal (`abs_tol=1e-12`);
- generated as equirectangular straight-line distance divided by 1.4 m/s and by 60, stored to 6 decimals.

> **Caveat:** "walking time" here contains **no street network** (no detours, no crossings). The matrix therefore provides connectivity and travel-time consistency only; it is not a physical-accessibility model and does not represent wheelchair access, step-free routing, stairs, lifts, slopes or barrier-aware paths. Treat it in the thesis as a lower bound or a straight-line proxy.

### 4.3 Utility formula (`U1_equal_rating_review_zone`)

```
rating_norm = 2 * google_rating                      # 0-10 scale
review_norm = 10 * log10(review_count + 1) / log10(max_review_count + 1)
zone_score  = A:10, B:8, C:6
utility     = (rating_norm + review_norm + zone_score) / 3
```

`audit_checks.py` recomputes this from the original high-precision dataset (git history `b38ee40:latest/poi_data_final_v2.csv`) with a tolerance of 1e-6.

### 4.4 Time semantics

- Everything is stored as absolute minutes after midnight (`09:30 -> 570`);
- `format_clock` renders past-midnight values as `D+1 HH:MM`;
- trigger time = `09:00 + offset`, where `offset = nominal x {0.95, 1.00, 1.05}`, computed with `Decimal` and `ROUND_HALF_UP` to avoid banker's rounding.

---

## 5. Architecture and call chain

```
CSV --> planning_core.load_planning_data --> PlanningData (pois + pre-expanded travel_lookup)
                                                   |
                                                   v
                                        planning_core.decode_route   <-- single feasibility gate
                                                   ^
                    +------------------------------+---------------------------+
                    |                              |                           |
              ga_core                    adaptive_ga_core              greedy_repair
        (static utility, initial plan)  (bounded replanning, 3 obj)   (deterministic greedy)
                    |                              |
                    |                     tripmates_lite_ga  <-- WES as 4th objective
                    |                     (reuses the same window and decoder)
                    v                              v
            results/static_utility      scenario_core.simulate_replanning_state
                                               + exposure_core.calculate_exposure_metrics
                                               + stability_metrics.compute_stability_metrics
                                                          |
                                                          v
                                              benchmark_common.build_metric_record
                                                          |
                                    +---------------------+---------------------+
                                    v                                           v
                          4 run_*.py experiment runners                 app.py (Streamlit)
```

**Actual call order of one replan**

```
ItineraryService.replan()
  |- simulate_replanning_state()          # lock the prefix, remaining mandatory, unavailable, cost spent
  |- prepare_adaptive_problem()           # mandatory backbone + candidate window (<=10) + locked travel
  |- dispatch:
  |    adaptive_without_thermal -> run_adaptive_without_thermal_ga()
  |    tripmates_lite          -> run_tripmates_lite_ga()
  |    greedy_repair           -> run_greedy_repair()
  |- calculate_exposure_metrics()         # TOEM / WES
  |- compute_stability_from_state()       # preserved / removed / added, order, shifts, edit distance
  \- build_replanning_explanation()       # natural-language explanation shown in the UI
```

---

## 5.1 Bounded replanning — formal definition

The replanner operates under an **enforceable bounded-repair contract** with ten rules:

| # | Rule | Enforcement mechanism |
|---|---|---|
| 1 | **Immutable prefix.** Visits completed before the trigger are locked and may not be reordered, removed or substituted. | `ReplanningState.completed_poi_ids` is excluded from the candidate window; the repaired route starts from the post-trigger state. |
| 2 | **Current-activity semantics.** If the traveller is mid-walking, waiting, or mid-visit at the trigger, the existing scenario rules apply consistently. | `simulate_replanning_state`: (a) walking leg → completed and locked, destination unvisited; (b) waiting → replanning starts at the trigger from the waiting POI; (c) visit → completed and locked. |
| 3 | **Suffix-only modification.** Only the unfinished suffix may be modified. | `decode_replanned_route` passes `start_poi_is_visit=False` and `initial_cost_rm=state.cost_spent_rm`; the prefix is never re-decoded. |
| 4 | **Bounded window.** An explicit `max_modifiable_suffix_activities` parameter caps the number of activities in the repaired suffix. | `BoundedRepairConfig` (default `MAXIMUM_CANDIDATE_COUNT` = 10), enforced in `build_shared_candidate_window` and `prepare_adaptive_problem`. |
| 5 | **Configurable and recorded.** The bound is configurable and appears in every experiment configuration and every repair record. | `BoundedRepairConfig.to_dict()` is serialised into run configs; `RepairRecord.bounded_repair_limit` is in every benchmark metric record. |
| 6 | **Decision audit.** Removed, skipped, shifted and replacement POIs are recorded with a reason. | `RepairDecision(poi_id, decision_type, reason)` entries in `RepairRecord.removed_activities` and `RepairRecord.replacements`. |
| 7 | **Explicit failure.** If no feasible bounded repair exists, an explicit failure result is returned. | `build_failure_repair_record` produces `failure_status="infeasible"` with a human-readable reason; no fabricated route is returned. |
| 8 | **Hard-constraint enforcement.** Opening hours, budget, mandatory POIs, route, duplication and end-time constraints are never bypassed. | The single decoder `planning_core.decode_route` is the only feasibility gate; every candidate is validated through it. |
| 9 | **Shared decoder.** All adaptive planners use the same decoder and hard constraints. | `decode_replanned_route` wraps `decode_route` for Adaptive Without Thermal, TripMates Lite and Greedy Repair. |
| 10 | **Planner parity.** Adaptive Without Thermal and TripMates Lite use identical candidate pools, search budgets and bounded-repair constraints. | Both call `prepare_adaptive_problem` with the same `BoundedRepairConfig`; the candidate window, mandatory backbone and `REPLANNING_GA_CONFIG` are shared. |

### Data structures

**`BoundedRepairConfig`** (frozen dataclass, `adaptive_ga_core.py`):

| Field | Type | Default | Description |
|---|---|---|---|
| `max_modifiable_suffix_activities` | `int` | `MAXIMUM_CANDIDATE_COUNT` (10) | Maximum number of POI visits the replanner may place in the repaired suffix. Must be ≥ 1 and ≤ 10. |

**`RepairDecision`** (frozen dataclass, `adaptive_ga_core.py`):

| Field | Type | Description |
|---|---|---|
| `poi_id` | `int` | The POI this decision concerns. |
| `decision_type` | `str` | `"retained"`, `"removed"`, `"shifted"` or `"replacement"`. |
| `reason` | `str` | Human-readable reason for the decision. |

**`RepairRecord`** (frozen dataclass, `adaptive_ga_core.py`):

| Field | Type | Description |
|---|---|---|
| `locked_prefix` | `tuple[int, ...]` | POI IDs completed before the trigger (immutable). |
| `current_activity` | `str` | What the traveller was doing at the trigger (`walking_leg`, `waiting`, `visit`, `none`). |
| `modified_suffix` | `tuple[int, ...]` | POI IDs in the repaired suffix, in execution order. |
| `retained_activities` | `tuple[int, ...]` | Original suffix POIs kept in the repair. |
| `removed_activities` | `tuple[RepairDecision, ...]` | Original suffix POIs removed, each with a reason. |
| `replacements` | `tuple[RepairDecision, ...]` | New POIs introduced, each with a reason. |
| `failure_status` | `str` | `"success"`, `"no_action"` or `"infeasible"`. |
| `failure_reason` | `str \| None` | Human-readable reason on failure, else `None`. |
| `bounded_repair_limit` | `int` | The configured `max_modifiable_suffix_activities`. |
| `actual_modification_count` | `int` | Number of suffix activities in the repair. |

### Worked example

```
Original itinerary (seed 1001, S1 nominal):
  1 -> 2 -> 12 -> 4 -> 33 -> 34 -> 36 -> 13

Disruption: Heavy Rain at 11:00 (trigger = 660 min)

Locked prefix (completed before trigger):
  [1, 2]

Current activity at trigger:
  visit (POI 12 in progress -> completed and locked)

Candidate repair window (max 10, mandatory first):
  mandatory backbone: [4, 13]
  optional candidates: [36, 12, 33, 34, 18, ...]
  (feasibility-gated, U/(T+V)-ranked)

Final itinerary:
  locked: [1, 2, 12]
  repaired suffix: [34, 4, 33, 36, 13]

RepairRecord:
  locked_prefix:              (1, 2, 12)
  current_activity:           visit
  modified_suffix:            (34, 4, 33, 36, 13)
  retained_activities:        (4, 33, 36, 13)
  removed_activities:         ()
  replacements:               (34,)
  failure_status:             success
  failure_reason:             None
  bounded_repair_limit:       10
  actual_modification_count:  5
```

### Benchmark audit fields

Every metric record produced by `benchmark_common.build_metric_record` now includes:

| Field | Source | Description |
|---|---|---|
| `bounded_repair_limit` | `RepairRecord` | Configured max suffix activities. |
| `actual_modification_count` | `RepairRecord` | Actual suffix size. |
| `locked_prefix` | `state.completed_poi_ids` | Immutable prefix, JSON-encoded. |
| `current_activity` | `state.committed_component` | Trigger-time activity. |
| `modified_suffix` | `repaired_route` | Repaired suffix, JSON-encoded. |
| `retained_activity_count` | `RepairRecord` | Count of retained original suffix POIs. |
| `removed_activity_count` | `RepairRecord` | Count of removed original suffix POIs. |
| `replacement_count` | `RepairRecord` | Count of replacement POIs. |
| `removed_pois_with_reasons` | `RepairRecord` | Full `RepairDecision` list for removed POIs. |
| `replacement_pois_with_reasons` | `RepairRecord` | Full `RepairDecision` list for replacements. |
| `repair_failure_status` | `RepairRecord` | `"success"`, `"no_action"` or `"infeasible"`. |
| `repair_failure_reason` | `RepairRecord` | Failure reason or `None`. |

### Tests

`test_bounded_repair.py` (17 tests) verifies:
- `BoundedRepairConfig` validation (zero, above-max, serialisation).
- Disruption before the first activity (no locked prefix).
- Disruption during walking (committed walking-leg semantics).
- Disruption during waiting (trigger-time replanning start).
- Disruption during a visit (visit completed and locked).
- Closure of a mandatory POI (unavailable handling, reason recorded).
- No feasible bounded repair (explicit failure, no fabricated route).
- Modification-limit exceeded (limit below mandatory count raises).
- Successful repair within the limit (repair record fields verified).
- Attempt to modify the locked prefix (prefix never in suffix).
- All optional POIs removed (tight limit = mandatory-only window).
- Preservation of mandatory POIs (no missing mandatory in repair).
- Duplicated replacement POIs (decoder rejects duplicates).
- Planner parity (Adaptive Without Thermal == TripMates Lite for M_s=1).
- Repair record auditability (all fields present and correctly typed).
- Greedy Repair produces a repair record.
- Hard constraints never silently bypassed (all 18 instances checked).

---

## 5.2 Itinerary stability — metric definitions

`stability_metrics.py` is the single implementation of the stability measurement. It is imported by `benchmark_common`, `itinerary_service`, the ablation runner and the sensitivity runner — **no planner computes its own stability numbers**.

### What is compared

```
original itinerary  = locked_prefix + state.original_remaining_poi_ids
replanned itinerary = locked_prefix + planner route
```

Both are observed from the same post-trigger state, so the comparison isolates the repair decision rather than the disruption itself.

### Exact metric definitions

All POI-level counts are computed on the **unlocked suffix**. Both sequences are first de-duplicated by first occurrence, so a malformed route with repeated IDs cannot inflate any count; repeated IDs are reported in `duplicate_*` fields.

| # | Field | Definition |
|---|---|---|
| 1 | `preserved_poi_count` | `\|original_suffix ∩ replanned_suffix\|` by POI identity, order-independent. |
| 2 | `preserved_poi_percentage` | `preserved_poi_count / \|original_suffix\|`; `None` when the original suffix is empty (ratio undefined). |
| 3 | `removed_pois` | Original-suffix POIs absent from the repair, listed in original execution order. |
| 4 | `added_pois` | Repaired-suffix POIs absent from the original suffix, listed in repaired execution order. These are the inserted / replacement POIs. |
| 5 | `changed_activity_count` | `len(removed) + len(added) + len(shifted)`; a POI counts once even if it is both reordered and shifted. When no replanned schedule exists, timing cannot be measured and only membership changes are counted — `timing_measurement_available` is then `False`, so the difference is never hidden. |
| 6 | `time_shift_count` | Preserved POIs whose `visit_start_min` differs between the original and replanned schedules by more than `TIME_SHIFT_TOLERANCE_MIN` (1e-6 min). `None` when either schedule is unavailable. |
| 6b | `total_time_shift_min` | Sum of the absolute shifts, in minutes. `None` under the same condition. |
| 7 | `order_deviation` | Kendall tau distance — the number of discordant pairs — between the two *retained subsequences* (each suffix restricted to the preserved POIs). `0` means relative order survives even when new POIs are inserted between retained ones. Range `0 .. n(n-1)/2`. |
| 8 | `edit_distance_score` | Weighted sum of the five components in the next subsection. |
| 9 | `modified_suffix_activity_count` | `len(replanned_suffix)` — the number of activities placed in the unlocked suffix. Identical to `RepairRecord.actual_modification_count` and bounded by `BoundedRepairConfig.max_modifiable_suffix_activities`. |
| 10 | `locked_prefix_changed` | `True` when the locked prefix was altered. See below. |
| 11 | `removed_poi_reasons` / `replacement_poi_reasons` | `RepairDecision` dictionaries copied from the `RepairRecord` when one is supplied; empty otherwise. |

### How the locked prefix is handled

The prefix is **verified, never corrected, and never counted as a permitted modification**.

Two independent checks produce a hard violation:

1. **Re-execution.** A POI from `state.completed_poi_ids` appearing in the repaired suffix means the replanner is re-running a completed visit. Reported in `prefix_reappearing_pois`.
2. **Final-route audit.** When a caller supplies a complete itinerary as `final_route`, its first `len(locked_prefix)` POIs must equal the locked prefix.

A violation sets `locked_prefix_changed = True`, `hard_stability_violation = True` and appends a readable entry to `violation_reasons`. The offending POI is **not** removed from `added_pois` and the route is **not** rewritten — the metric reports the breach and leaves the itinerary untouched, so a downstream reader sees exactly what the planner produced.

### How the edit distance is calculated

```
edit_distance_score =
      1.00 * len(removed_pois)
    + 1.00 * len(added_pois)
    + 0.50 * order_deviation
    + 0.25 * time_shift_count
    + 1000.00 * locked_prefix_changed
```

`StabilityWeightConfig` (frozen dataclass) holds every weight as a named constant:

| Field | Default | Rationale |
|---|---|---|
| `removed_poi_weight` | 1.00 | Base unit of change: one planned visit lost. |
| `inserted_poi_weight` | 1.00 | Symmetric unit: one itinerary slot differs. |
| `order_change_weight` | 0.50 | Reordering keeps every POI, so it costs half an identity change. |
| `time_shift_weight` | 0.25 | A retained POI that only moves in time is the mildest change. |
| `locked_prefix_violation_weight` | 1000.00 | A contract breach must dominate any permitted soft change. |

**Why 1000 is not arbitrary.** Inside the bounded window of at most `MAXIMUM_CANDIDATE_COUNT` = 10 suffix activities, the maximum soft cost is

```
n * 1.00  +  n * 1.00  +  [n(n-1)/2] * 0.50  +  n * 0.25
= 2.25n + 0.25n(n-1)  =  45.0   at n = 10
```

`stability_metrics.maximum_soft_edit_distance(n)` returns this bound, and `test_locked_prefix_penalty_dominates_soft_cost` asserts the dominance. Any conforming repair therefore scores below any violating repair at every window size the contract permits.

**Determinism.** Every component is an integer count; ordering follows execution order rather than set iteration; Kendall tau is computed by direct pairwise comparison; all floats are rounded to six decimals. Identical inputs always produce an identical `StabilityMetrics`, and `test_determinism` asserts it.

**Sensitivity analysis.** Pass a different `StabilityWeightConfig` to `compute_stability_metrics`, `build_metric_record(..., stability_weights=...)` or `ItineraryService.replan(..., stability_weights=...)`. The chosen configuration is serialised into `stability_weights` in every metric record and into the run config of every experiment.

### Where the columns appear

| Artefact | What is written |
|---|---|
| `benchmark_common.build_metric_record()` | The full 28-field stability block (all metrics plus components, reasons and duplicates). |
| `results/rq1_full_benchmark_corrected_static_rule/rq1_raw_runs.csv` | All of the above, one row per run. |
| `rq1_planner_scenario_summary.csv`, `rq1_overall_planner_summary.csv` | `mean_preserved_poi_count`, `mean_preserved_poi_percentage`, `mean_removed_poi_count`, `mean_added_poi_count`, `mean_changed_activity_count`, `mean_modified_suffix_activity_count`, `mean_time_shift_count`, `mean_order_deviation`, `mean_edit_distance_score`, `locked_prefix_violation_count`. |
| `results/weather_ablation/weather_ablation_raw_90_pairs.csv` | `adaptive_*` and `tripmates_*` blocks of the compact summary (11 columns each). |
| `weather_ablation_instance_summary.csv` | `median_adaptive_edit_distance`, `median_tripmates_edit_distance`, preserved-count medians, violation counts. |
| `results/thermal_tradeoff_sensitivity_alpha_independent_v2/thermal_tradeoff_raw_runs.csv` | `adaptive_*` / `tripmates_*` stability block plus `edit_distance_difference` (TripMates − Adaptive). |
| `thermal_tradeoff_instance_summary.csv` | Median edit distances and violation counts per instance and alpha. |
| `ItineraryService.replan()` → `ReplanningOutcome.stability` | A `StabilityMetrics` object; `app.py` renders a one-line caption and raises an error banner on a hard violation. |

All additions are **new columns**. Existing result files are never rewritten, so previously frozen CSVs remain valid; re-running a benchmark simply produces a wider schema.

### How to read the new columns

| Column | Interpretation |
|---|---|
| `preserved_poi_count` / `preserved_poi_percentage` | Plan retention. Higher = the repair stayed closer to the plan. |
| `removed_pois` / `added_pois` | JSON lists. `added_pois` is what was substituted in. |
| `changed_activity_count` | Total POI-level churn (identity + timing), each POI counted once. |
| `time_shift_count` | `None` when the repair was infeasible and no schedule exists — never silently zero. |
| `order_deviation` | Pure reordering. `0` together with non-zero `added_pois` means new stops were slotted in without disturbing retained order. |
| `edit_distance_score` | One comparable number across planners. `0.0` = the repair reproduced the original plan exactly. |
| `hard_stability_violation` | Must be `False` for every conforming run. Any `True` is a contract breach to investigate, not a soft result. |

### Tests

`test_stability_metrics.py` (17 tests) covers:

- no changes after replanning;
- one removed POI;
- one replacement POI;
- a shifted activity time (including a sub-tolerance shift that must not count);
- a reordered suffix (including a full reversal and an insertion that preserves relative order);
- an unchanged locked prefix;
- a locked-prefix violation (re-execution and final-route mismatch);
- an empty suffix (both "nothing planned" and "everything dropped");
- an infeasible replanning result (`time_shift_count is None`, no fabricated schedule);
- duplicate POIs (reported, never double-counted);
- identical original and replanned itineraries;
- weight-configuration validation and weight sensitivity;
- dominance of the locked-prefix penalty;
- determinism of repeated calls;
- integration with `build_metric_record`;
- a real-pipeline check over all 18 instances that a conforming repair never reports a violation.

---

## 6. Shared engine

### 6.1 `planning_core.py` (931 lines)

**Fixed constants:** `EXPECTED_POI_COUNT=50`, `TRIP_START_MIN=540` (09:00), `TRIP_END_MIN=1020` (17:00), `COST_BUDGET_RM=160.0`, `EXPERIMENT_DAY="WED"`.

**Parsing helpers**

| Function | Behaviour |
|---|---|
| `parse_yes_no` | `YES/TRUE/1` -> True, `NO/FALSE/0` -> False, anything else raises |
| `parse_clock` | `"HH:MM"` -> minutes, validates 0–23 / 0–59 |
| `format_clock` | minutes -> `HH:MM`, prefixes `D+` past midnight |
| `parse_off_days` | `NONE/N/A/empty` -> empty set; otherwise splits on `,;/` and validates day codes |

**Data models**

- `POI` (frozen): see 4.1. Core method `get_opening_window(day_code)` — returns `None` on a closed day, `(0, 1440)` for 24-hour POIs, and adds 1440 to closing when `closes_next_day`.
- `PlanningConstraints` (frozen): start POI, start/end minutes, day code, budget; `__post_init__` validates the interval.
- `PlanningData` (frozen):
  - `__post_init__` **pre-expands the 50x50 matrix into `_travel_lookup: dict[(from, to), float]`**. A GA insertion trial performs hundreds of lookups, so this avoids repeated pandas `.at` calls.
  - `with_mandatory_pois(ids)` / `with_preferred_pois(ids, multiplier)` return a **new** `PlanningData`, scoped to the current request, so benchmark utility values stay frozen.
  - `get_travel_time` enforces a zero diagonal and finite positive off-diagonal values.
- `TravelAdjustments`: `global_multiplier` (network-wide) plus `link_multipliers[(from, to)]` (zonal); `apply()` multiplies them.
- `ScheduleEntry`: index, POI, predecessor, travel, arrival/waiting/start/end (minutes and clock strings), visit duration, admission, cumulative cost.
- `ScheduleResult`: `feasible`, `failure_reason`, requested vs completed sequence, per-component time totals, finish time, total cost, missing and unavailable mandatory IDs.

**`load_planning_data()`** enforces the contract:

1. all 17 required columns present;
2. exactly 50 rows, `poi_id` exactly `1..50` ascending;
3. matrix columns exactly `["poi_id", "1", ..., "50"]`, rows in the same order;
4. shape 50x50, strictly symmetric, zero diagonal.

**`decode_route()` — the most important function in the repository**

```python
decode_route(
    data, visit_sequence, *,
    start_poi_id=1, start_time_min=540, trip_end_min=1020,
    day_code="WED", cost_budget_rm=160.0,
    initial_cost_rm=0.0,              # cost already spent, during replanning
    start_poi_is_visit=True,          # False = the start POI is just "current location"
    required_mandatory_ids=None,      # replanning only requires the *remaining* mandatory POIs
    unavailable_poi_ids=(),           # closure scenario
    travel_adjustments=None,          # delay / congestion
) -> ScheduleResult
```

Execution:

1. unknown IDs or duplicates -> infeasible;
2. with `start_poi_is_visit=True` the first element must equal the start POI; with `False` the start POI may appear first but counts only as the current location;
3. per stop: `travel` (possibly scaled) -> `arrival` -> `waiting = max(0, opening - arrival)` -> `visit_end = start + duration`;
4. four hard constraints: **not started after closing / not finished after closing / not past 17:00 / not over budget**;
5. final check that every available mandatory POI was completed.

On the first violation it returns `feasible=False` **immediately** but keeps the entries built so far, which makes failures traceable and countable ("POI 13 would finish after 17:00").

> Design intent: **every planner shares this one decoder**, so "feasible" has a single definition and no two planners can disagree about it.

### 6.2 `scenario_core.py` (744 lines)

**Six frozen disruptions**

| ID | Name | Nominal offset | Parameters | Purpose |
|---|---|---|---|---|
| S1 | Heavy Rain | +120 min | `thermal_multiplier=1.8` | reduce exposure in rain |
| S2 | Movement Delay | +90 | `global_travel_multiplier=1.5` | everything slower, no POI removed |
| S3 | Attraction Closure | +150 | closes POI 13 | remove and substitute |
| S4 | Congestion | +135 | zone-A internal links x1.3 | local slowdown |
| S5 | Heatwave | +180 | `thermal_multiplier=2.0` | reduce heat exposure |
| S6 | Combined | +120 | 1.8 + network x1.3 + closes POI 34 | simultaneous changes |

- `ScenarioDefinition.build_travel_adjustments()`: congestion applies only to directed links whose **origin and destination both lie in that zone**.
- `get_travel_adjustments()` is **memoised** (key = `id(data)` + scenario ID, holding a reference to `data` so a recycled id cannot return a stale entry). The adaptive decoder calls it once per insertion trial; without the cache, S4 would rebuild the 50x50 table every time.
- `build_scenario_instances()`: 6 x {early, nominal, late} = **18 instances**.

**`simulate_replanning_state()` — traveller state at the trigger**

| At the trigger the traveller is… | Handling | Replanning starts at |
|---|---|---|
| after a completed visit | locked | — |
| walking (in transit) | the leg is completed at its **original** duration and locked; the destination is **not yet visited** (`current_poi_requires_visit=True`) | arrival time |
| waiting for opening | stays at that POI | trigger time |
| inside a visit | that visit **counts as completed and locked** | visit end time |
| finished entirely | `route_finished_before_trigger=True` | trigger time (service returns `no_action`) |

It also derives `completed_poi_ids`, `original_remaining_poi_ids`, `remaining_mandatory_ids`, `unavailable_mandatory_ids`, `cost_spent_rm`, `utility_achieved`, `unavailable_poi_ids`, `thermal_multiplier` and `global_travel_multiplier`.

### 6.3 `exposure_core.py` (482 lines)

- exposed fraction: `X_j = (1 - indoor_ratio)(1 - shelter_ratio)`;
- `calculate_weighted_interval()`: weight 1 **before** the trigger and `M_s` **after** it, so a disruption cannot retroactively affect exposure that already happened;
- `calculate_prefix_exposure()`: from trip start to replan start, covering completed and in-transit components; **waiting time is excluded** (the chapter-3 model covers walking and visits only);
- `calculate_repaired_exposure()`: full walking time of the new route plus `duration x X_j` per POI, all multiplied by `M_s` later because everything occurs after the trigger;
- aggregates: `OWM` (outdoor walking), `OVM` (outdoor visiting), `TOEM = OWM + OVM`, `WES` (weighted sum).

---

## 7. The four planners

### 7.1 `ga_core.py` (1382 lines) — static utility GA

**`GAConfig` defaults:** population 80, at most 120 generations, tournament 3, crossover 0.90, mutation 0.20, elitism 2, stagnation limit 15.

**Mandatory backbone** `build_mandatory_backbone()`: three mandatory POIs with a fixed origin means only 2! orderings; it enumerates them exactly and sorts by (travel, finish time, route).

**A chromosome is a permutation of the optional POIs**, and `decode_chromosome()` performs **insertion decoding**:

```
route = mandatory backbone
for each optional POI in chromosome order:
    try every insertion position (position 0 is reserved for the origin)
    keep the feasible insertion with the smallest (travel increase, finish time, route)
    if no feasible position exists, skip the gene
```

This is the standard permutation-plus-greedy-insertion scheme: chromosomes are always legal, at the cost of a non-injective decoder (several chromosomes can map to the same route).

- `evaluate_initial_route_summary()` is a **fast feasibility pre-check** that computes only time and cost without building `ScheduleEntry` objects; it is used inside insertion trials.
- `dominates()` / `assign_pareto_ranks()` / `assign_crowding_distance()`: three-objective non-dominated sorting (utility up, count up, travel down) plus crowding distance.
- `select_final_solution()`: the **frozen guardrail** — first front only, `AU >= 0.90 * U*` and `N >= N* - 1`, then minimise (travel, -utility, route).
- Initialisation: a `U/(V+T)` utility-efficiency seed, a nearest-neighbour seed, and random permutations (deduplicated).
- Operators: tournament selection (rank, crowding distance, utility, count, travel, route), order crossover (OX), swap mutation.
- `run_static_utility_ga()`: evaluation cache (chromosome -> solution), distinct-route tracking, stagnation measured on the selected route, and `convergence_generation` = the generation when the final route first appeared.

### 7.2 `adaptive_ga_core.py` (1354 lines) — bounded replanning without thermal

- `REPLANNING_GA_CONFIG`: **40 individuals x 60 generations**, other settings unchanged; `MAXIMUM_CANDIDATE_COUNT = 10`.
- `decode_replanned_route()`: starts at `state.current_poi_id` and `replan_start_min`, carries `initial_cost_rm`, sets `start_poi_is_visit=False`, requires only the remaining mandatory POIs, and applies unavailable POIs and travel adjustments.

**Shared candidate window `build_shared_candidate_window()`** (the key to a fair ablation)

1. exclude completed and unavailable POIs; also exclude the current POI if it has already been visited;
2. include all remaining mandatory POIs directly;
3. every other POI must pass the gate "inserting it still allows all remaining mandatory POIs to be completed" (`candidate_has_feasible_insertion`, enumerating mandatory orders);
4. fill remaining slots by descending `U / (T_ij + V_j)`, breaking ties by ascending POI ID;
5. cap at 10; if the remaining mandatory POIs alone exceed 10 it raises (cannot happen with the current data).

> The window is built **without any thermal information**. Thermal data enters only the objective function and the final selection — that is the control-variable design.

- `decode_adaptive_chromosome()`: insertion at position 0 is allowed (the current location is not necessarily a visit); utility, count and travel must all include the **locked prefix** (`locked_utility`, `locked_completed_count`, `locked_travel_min`).
- `calculate_locked_travel_minutes()`: travel into completed visits plus the committed in-transit or waiting leg.
- `initialise_adaptive_population()`: utility order, nearest-neighbour order, then random; duplicates are allowed when too few optional POIs exist.
- `run_adaptive_without_thermal_ga()`: the three-objective version, all other mechanics identical.

### 7.3 `tripmates_lite_ga.py` (1395 lines) — thermal-aware

**Four objectives:** utility up, count up, travel down, **WES down** (`thermal_dominates` plus four-objective crowding distance).

**`ThermalSelectionPolicy`**

```python
alpha: float = 0.90                  # utility-retention threshold
visit_guardrail_margin: int = 1      # N >= N_ref - 1
utility_guardrail_baseline: str = "non_thermal_reference"
    # non_thermal_reference | front_maximum | front_maximum_plus_reference
priority: str = "frozen"             # frozen | c2
```

**`evaluate_thermal_selection()`** (auditable route selection)

1. take the first front of the four-objective ranking;
2. compute `U*` according to the baseline: by default the utility of the **seed-matched non-thermal reference**; `front_maximum` uses the front maximum; `front_maximum_plus_reference` uses the larger of the two;
3. two guardrails: `AU >= alpha * U*` and `N >= N_ref - 1`;
4. among eligible routes, minimise `_frozen_selection_key` (**WES, -utility, TOEM, travel, route**) or `_c2_selection_key` (WES, travel, -utility, route);
5. **if nothing is eligible, fall back to the non-thermal reference** and set `guardrail_fallback_used=True` with `selection_status="guardrail_fallback"`.

The whole process is retained on `TripMatesGARunResult`: `first_front`, `front_utility_guardrail`, `front_visit_guardrail`, `front_guardrail_eligible`, `front_selected`, `u_star`, `reference_utility`, `front_max_utility`.

**Two critical design decisions in `run_tripmates_lite_ga()`**

1. when `M_s == 1` it **delegates directly** to `run_adaptive_without_thermal_ga()`, guaranteeing bit-identical output in non-thermal scenarios;
2. when `M_s > 1` it first runs the non-thermal GA to obtain the reference solution, then runs the four-objective GA. **Every internal search decision (generation-zero progress reference, in-loop stagnation comparison, `convergence_generation`) is pinned to the default policy**; `alpha` affects only the final selection. This is the core of the C2 v2 fix (see 9.5).

### 7.4 `greedy_repair.py` (434 lines) — deterministic baseline

It uses the **same** candidate window and `decode_replanned_route`, but no GA, no randomness, no WES and no thermal multiplier.

```
route = []
loop:
    for each unused candidate in the window:
        feasible if (route + [c]) still allows all remaining mandatory POIs (enumerated exactly)
    pick (max U/(travel+visit), min travel, max utility, min ID)
    stop when no feasible candidate remains
```

Each step records a `GreedyRepairStep` (incremental travel, visit length, utility, utility per minute, mandatory flag, the feasible candidate set at that step). Missing mandatory POIs produce `status="infeasible"`.

### 7.5 `itinerary_service.py` (538 lines) — application service layer

- `TripRequest`: date, start POI, start/end minutes, budget, mandatory/preferred IDs, `seed=1001`.
- `validate_request()`: date type, start POI exists, time interval, non-negative budget, unknown IDs, duplicate IDs, **mandatory POIs closed on that weekday**.
- `build_instance()`: maps `date.weekday()` to a day code; mandatory set = user mandatory plus the start POI; preferred POIs get utility x **1.25** (`PREFERRED_UTILITY_MULTIPLIER`).
- `plan_initial_itinerary()`: calls `run_static_utility_ga` and converts exceptions into `status="infeasible"`.
- `replan()`: simulate state -> return `no_action` if already finished -> dispatch by `planner_id` -> time it -> compute exposure -> build the explanation.
- `build_replanning_explanation()`: a fixed template — "X occurred at HH:MM during <walking/visit/waiting>, removed or skipped …, introduced …, because …". The UI displays it directly.

---

## 8. The web application

`app.py` (559 lines) is a **single-file Streamlit app** with no templates or static assets; all styling is inlined.

| Area | Function | Notes |
|---|---|---|
| Styling | `inject_styles` | dark-green sidebar, gradient hero, route cards, metric cards |
| Map | `render_route_map` | slider reveals the route stop by stop; `PathLayer` for the path, `ScatterplotLayer` for stops (last stop orange), `TextLayer` for numbers; tooltip shows arrival/departure/duration/fee/utility; side card shows outdoor share `100 x (1 - indoor - shelter)` |
| Itinerary | `render_schedule` / `render_activity_timeline` | map + two-column timeline + four metrics (stops, walking, finish, admission); replanning labels the completed/committed prefix separately from the revised unfinished suffix |
| History | `render_saved_view` / `render_saved_summary` | pick a saved trip and restore map, timeline, inputs and metrics |
| Main flow | `main` | weather -> form -> plan -> replan -> save |

**Performance settings**

- `QUICK_INITIAL_CONFIG` / `QUICK_REPLANNING_CONFIG`: **12 x 12, stagnation 4** (the UI default "Quick preview", responds in seconds);
- "Research benchmark" uses the thesis configuration (80x120 / 40x60) and is noticeably slower in the browser (tens of seconds).

**Scenario selection:** with `Use live weather automatically`, `profile.category == "rain"` maps to S1 and both `"hot"` and `"hot_humid"` map to S5; otherwise no replanning is performed. S1–S6 and the trigger variant can also be chosen manually.

**Saving:** `ItineraryHistoryRepository.save()` writes `data/itinerary_history.sqlite3` (three tables: `itinerary_run` / `itinerary_stop` / `itinerary_metric`; trigger state and request parameters are stored as JSON). Initial plans and replans save different metric sets (the former includes `achieved_utility` and generation counts, the latter WES, exposure and latency).

**Supporting modules**

- `weather_service.py`: `WeatherProfile` with `source_type: live|cache|predefined`; `classify_weather()` maps rain codes to `("rain", 1.8)` and `T >= 35 C` to `("hot", 2.0)`, otherwise `("normal", 1.0)`; `get_profile()` degrades **live -> cache -> predefined** and records the reason in `fallback_reason`. The external API is fully isolated from the planning engine.
- `itinerary_history.py`: SQLite repository; `session()` context manager handles commit/rollback; `initialise()` includes `ALTER TABLE` statements for backward compatibility.

---

## 9. Experiment inventory

### 9.1 Experiment 1 — initial routes (`static_utility_baseline.py`)

- **What it does:** runs the 80x120 GA for seeds 1001–1010 and freezes one initial route per seed for reuse downstream.
- **Artefacts:** `results/static_utility/` — ten `seed_XXXX_schedule.csv` files, `static_utility_seed_summary.csv`, `initial_routes_by_seed.json`, `static_utility_ga_config.json`.
- **Result:** 8 stops, utility 66.96, walking 66.5 min, RM 157 admission, finishes 16:57. **Nine of ten seeds converge to the same route** `[1,2,12,4,33,34,36,13]`; one seed finds a higher-utility route `[1,2,12,4,34,36,15,13]` (68.35).
- **Implication:** the search space is narrow and the GA is stable — which also explains why the thermal objective has almost no room later.

### 9.2 Experiment 2 — distance baseline (`static_distance_baseline.py`)

- **What it does:** pure nearest-neighbour greedy (shortest walk, then higher utility, then lower ID), with a mandatory-feasibility gate at every step.
- **Artefacts:** `results/static_distance_normal_route.csv`, `..._summary.json`.
- **Result:** `[1,2,5,3,4,13]`, 6 stops, utility 49.50, walking 61.33 min, RM 142, finishes 16:46.

### 9.3 Experiment 3 — main ablation (`run_weather_ablation_all_seeds.py`)

- **What it does:** 9 weather instances (S1/S5/S6 x early/nominal/late) x 10 seeds = **90 paired comparisons**; each pair runs Adaptive Without Thermal and TripMates Lite over the same state, problem and candidate window.
- **Protocol:** only **hard constraints** raise (feasibility, mandatory POIs, duplicates, closed POIs, budget, 17:00, unknown IDs); guardrail failures are **recorded, never corrected**.
- **Artefacts:** `results/weather_ablation/` — raw 90 pairs, per-instance summary, run config.
- **Result:** **all 90 WES differences are zero**, zero route changes; `weather_ablation_statistical_summary.json` states `"No observed thermal advantage: every paired WES difference is zero."`
- **Statistics** (`analyze_weather_ablation.py`): paired Wilcoxon plus rank-biserial -> effect 0.0, p = 1.0.

### 9.4 Experiment 4 — RQ1 full benchmark (`run_rq1_full_benchmark.py`)

- **What it does:** 5 planners x 18 official instances = **738 runs** (Static Distance once per instance = 18; the other four planners x 10 seeds x 18 = 720).
- **Pairing rule:** for Static Utility and Greedy Repair the "seed" only selects the matching initial route; it is not an RNG seed. Both GAs use matched seeds 1001–1010.
- **Artefacts:** `results/rq1_full_benchmark_corrected_static_rule/` — `rq1_raw_runs.csv` (738 rows), `rq1_planner_scenario_summary.csv`, `rq1_overall_planner_summary.csv`, `rq1_failure_cases.csv`, `rq1_run_config.json` (SHA-256 of dataset, matrix and initial routes plus runtime metadata).
- **Statistics** (`analyze_rq1_full_benchmark.py`): 5 pairings x 5 metrics x (overall plus 18 instances), two-sided Wilcoxon with matched-pairs rank-biserial effect size.

### 9.5 Experiment 5 — C2 alpha sensitivity (`run_thermal_tradeoff_sensitivity.py`)

- **What it does:** 9 instances x 10 seeds x alpha in {1.00, 0.95, 0.90, 0.85} = **360 TripMates runs** plus 90 non-thermal baselines (reused per instance-seed pair).
- **C2 policy:** `U* = max(front maximum, non-thermal reference)`, `AU >= alpha * U*`, `N >= N_ref - 1`, priority `WES -> travel -> -utility`, structured fallback.
- **The v2 fix (important):** in v1 the **search accidentally depended on alpha** (generation zero used the requested alpha while the in-loop stagnation check fell back to the default). v2 pins **every internal search decision** to the default policy so alpha affects only the final selection. The v1 directory is preserved unchanged and marked `supersedes` in `experiment_manifest.json`.
- **Artefacts:** `results/thermal_tradeoff_sensitivity_alpha_independent_v2/` — 360 raw rows, the full first fronts (`pareto_fronts.csv`), per-instance summary, chart-ready CSVs, run config, manifest.
- **Statistics** (`analyze_thermal_tradeoff_sensitivity.py`): per-alpha primary test with **Holm** adjustment; explicitly flagged `is_confirmatory=false`.

### 9.6 Experiment 6 — latency (`run_latency_benchmark.py`)

- **What it does:** per scenario, **one untimed warm-up plus 30 timed repetitions** (fixed seed 1001, `perf_counter`, excluding file writing and plotting).
- **Three scopes** reported separately and never mixed: `engine_only` (data loaded, state pre-built), `end_to_end_request` (including state construction), `shared_preparation` (state construction only).
- **Two settings:** `research_benchmark` (frozen configuration) and `quick_preview` (12x8, explicitly not a research benchmark); Greedy is labelled `deterministic_no_ga`.
- **Artefacts:** `results/latency_benchmark/latency_raw_measurements.csv`, `latency_summary.csv`, `latency_run_config.json`.

### 9.7 Experiment 7 — dataset audit (`audit_checks.py`)

Twelve check groups, producing `dataset_validation_report.json` (report_version 2.0.0): file paths, loading (including recovery of the original high-precision CSV from git history), metadata, POI catalogue validation, derived-file validation, matrix validation, **matrix formula verification** (recomputed equirectangular distance / 1.4 m/s, tolerance 1e-6; tolerance 0.002 when coordinates are rounded to 6 decimals), matrix pass/fail, cross-file consistency, file integrity (SHA-256), overall verdict, JSON construction, write-out, and a console summary.

### 9.8 Diagnostic (`diagnose_thermal_diversity.py`)

- **What it does:** fixes seed 1001 and S1_nominal, prints the thermal profile of the candidate window (utility, visit length, indoor/shelter ratios, exposed visit minutes, travel from the current POI, a local WES proxy), then **samples 2000 chromosomes** to estimate the first front and the set eligible under the 0.90 guardrail — answering "why do both planners pick the same route?".
- **Status:** diagnostic only; it produces no official results and does not modify the frozen configuration.

### 9.9 Provenance (`make_research_manifest.py`)

Generates the C2 v2 `experiment_manifest.json`: what supersedes what, the search and final-selection policy modes, SHA-256 of the frozen inputs, and hashes of the frozen confirmatory result (`results/weather_ablation/`).

---

## 10. Results and interpretation

### 10.1 Recorded skip-and-continue RQ1 run (`results/rq1_full_benchmark_corrected_static_rule/rq1_overall_planner_summary.csv`)

| Planner | Runs | Feasible | Mandatory complete | Mean utility | Utility retention | Mean walking min | Mean WES | Median s | P95 s |
|---|---|---|---|---|---|---|---|---|---|
| Static Distance | 18 | 100.0% | 100.0% | 44.17 | 0.892 | 62.00 | 251.46 | 0.0004 | 0.0009 |
| Static Utility | 180 | 100.0% | 100.0% | 54.44 | 0.811 | 66.28 | 285.86 | 0.0004 | 0.0008 |
| Greedy Repair | 180 | 100% | 100% | 59.42 | 0.886 | 100.06 | 353.38 | 0.0116 | 0.0158 |
| Adaptive Without Thermal | 180 | 100% | 100% | 63.88 | 0.952 | 68.87 | 310.50 | 1.78 | 4.70 |
| **TripMates Lite** | 180 | 100% | 100% | **63.88** | **0.952** | **68.87** | **310.50** | 3.09 | 9.45 |

**How to read this**

- Under the originally planned skip-and-continue static rule, both static baselines remain feasible, while Adaptive Without Thermal and TripMates Lite achieve higher completion and utility than the static baselines. The minimum-deletion primary comparison below provides a stronger fixed-order comparator; the archived legacy replay results remain available for protocol comparison.
- Greedy is roughly 100x faster but walks 45% more and loses 7% utility — feasible, but not worthwhile.
- **TripMates equals Adaptive exactly**: under the frozen guardrail the thermal objective never changed the selected route (every run fell back to the non-thermal reference). This is the central negative result, not a defect.
- The cost: TripMates is about **74% slower** than Adaptive in this RQ1 run (3.09 s vs 1.78 s) because it runs two GAs (reference plus four-objective).

**Minimum-deletion primary comparison.** `results/rq1_full_benchmark_minimum_deletion_primary/` contains 738 rows: 198 minimum-deletion static rows and 540 unchanged Greedy Repair and adaptive rows. The configuration names and hashes both sources. The fixed-order executor chooses a feasible subsequence that retains the most original visits; it cannot reorder or insert a substitute. The original skip-and-continue run above remains the protocol-sensitivity result.

| Planner | Runs | Mean ICR | Mean original POI retention | Mean utility | Mean walking min |
|---|---:|---:|---:|---:|---:|
| Static Distance | 18 | 0.9167 | 0.9167 | 45.638 | 62.046 |
| Static Utility | 180 | 0.9375 | 0.9375 | 62.755 | 69.869 |
| Greedy Repair | 180 | 0.8958 | 0.8500 | 59.421 | 100.058 |
| Adaptive Without Thermal | 180 | 0.9583 | 0.9056 | 63.878 | 68.868 |
| TripMates Lite | 180 | 0.9583 | 0.9056 | 63.878 | 68.868 |

The adaptive-minus-Static-Utility ICR difference is 0.0208 overall. It occurs in S3 closure (family difference 0.1250); the other five families tie. Static Utility retains more original POIs overall (0.9375 versus 0.9056). The substitute in S3 cannot complete the unavailable mandatory POI, so completion ratio and mandatory completion must be interpreted separately.

### 10.2 Main ablation (`results/weather_ablation/`)

- All 90 paired WES differences are zero; zero route changes; effect size 0.0; p = 1.0.
- Every per-instance summary reports `utility_guardrail_pass_count=10` and `visit_guardrail_pass_count=10`. This is **guaranteed by construction for a fallback solution** and must not be read as evidence that thermal selection succeeded.

### 10.3 C2 alpha sensitivity (v2)

| alpha | WES advantage (positive = better) | Effect size | Adjusted p | Interpretation |
|---|---|---|---|---|
| 1.00 | **-14.11** (worse) | -1.00 | 4.1e-16 | the guardrail excludes the reference from the front, forcing a front member with higher WES |
| 0.95 | 0.00 | 0.00 | 1.0 | full fallback, identical to the baseline |
| 0.90 | 0.00 | 0.00 | 1.0 | full fallback (equals the frozen policy) |
| 0.85 | **+35.19** (better, median +53.13) | +1.00 | 1.7e-14 | relaxing utility retention actually buys exposure reduction |

> Suggested framing: **under the fixed policy (alpha = 0.90) the thermal objective does not change route selection; only relaxing utility retention to 0.85 produces a significant WES improvement** — i.e. exposure reduction is paid for with utility and visit count. That is precisely what "trade-off" means here.

### 10.4 Latency (S1_early, 30 repetitions, `engine_only`)

| Planner | Median | P95 |
|---|---|---|
| Greedy Repair | 3.0 ms | 3.7 ms |
| Adaptive Without Thermal | 513 ms | 978 ms |
| TripMates Lite | 3.48 s | 4.98 s |

(For S1_nominal the Adaptive median is about 1.68 s; TripMates is roughly double that.)

### 10.5 Dataset audit

`dataset_validation_report.json`: 50 rows x 31 columns, no missing values, valid coordinates, utility formula recomputed with zero error, matrix symmetric and consistent with the formula — all checks pass.

---

## 11. Test suite

```bash
pytest -q                 # 117 tests, includes 3 slow frozen-artefact replays
pytest -q -m "not slow"   # 114 tests, skips the 3 frozen-artefact replays
```

`pytest.ini` sets `pythonpath = .` (so `tests/` can import the top-level modules without an `__init__.py`), registers the `slow` marker and enables the short summary. `tests/conftest.py` inserts the repository root on `sys.path` as a belt-and-braces measure.

The suite is deliberately split into three layers.

### 11.1 Synthetic contract suite (`tests/`, 56 tests)

Fast, hermetic, and independent of both the 50-POI catalogue and every generated result file. `tests/conftest.py` builds a six-POI instance directly as a `PlanningData` object (the `load_planning_data` 50-POI validation is bypassed by construction), with a hand-written walking matrix and a fixed timeline:

```
09:00-09:30  POI 1 Origin Square   (24 h, mandatory)
09:40-10:40  POI 2 Heritage Museum (09:00-12:00, mandatory)
11:00-11:45  POI 3 Riverside Park  (09:00-17:00, optional)
11:55-13:00  waiting at POI 5      (opens 13:00)
13:00-13:30  POI 5 Late Gallery    (optional)
14:00-15:00  POI 4 Sky Tower       (10:00-18:00, mandatory)
```

Because the timeline is fixed, a trigger can be placed at an exact minute: offset 35 is mid-walk, 60 is mid-visit, 210 is mid-wait, 0 is at the trip start and −30 is before it.

| File | Tests | What it guards |
|---|---|---|
| `tests/test_feasibility.py` | 15 | every `decode_route` hard constraint, one test per rule: valid route, late arrival, closed before arrival, ending after closing, ending after the 17:00 horizon, non-overlap, duplicate, missing route, unavailable, budget, missing mandatory, unknown ID, closed-on-day, wrong start POI, unavailable-mandatory reporting |
| `tests/test_disruptions.py` | 11 | the four commit points (before start, walking, waiting, visit), locked prefix, in-transit leg semantics, closed-POI exclusion, delayed travel, zone-congestion scoping, combined disruption |
| `tests/test_repair_behavior.py` | 20 | Greedy Repair success and two independent failure modes, optional-removal, mandatory retention, locked-prefix immutability at four commit points, modification limits (valid and invalid), no duplicate replacements, the five stability shapes and a hard prefix violation |
| `tests/test_planner_contracts.py` | 10 | shared candidate window, shared decoder, thermal guardrails, alpha-independent search, static-baseline reproducibility, smoke isolation, seeded determinism, decoder idempotence |

Two design points are worth calling out:

**One scenario id per disruption.** `scenario_core.get_travel_adjustments` memoises on `(id(data), scenario_id)` and therefore assumes one id maps to exactly one `ScenarioDefinition`. `conftest.make_scenario` derives the id from the disruption parameters when the caller does not supply one, so two probes that differ in multiplier or closure can never share a stale travel-adjustment table.

**No live weather.** Nothing in the suite reaches Open-Meteo. `test_application_services` (wrapped, see below) injects a `fetcher` callable into `WeatherServiceAdapter`, and the synthetic tests never construct one at all.

### 11.2 Legacy script wrappers (`tests/test_legacy_script_checks.py`, 11 tests)

Eleven of the original `test_*.py` files are scripts exposing `main()` rather than test functions, so pytest never collected them. The wrapper imports each module, runs `main()` unchanged and converts a non-zero `SystemExit` into a pytest failure. Nothing is rewritten, so the assertions they make are still the original ones.

Each run happens inside a temporary working directory that holds copies of the two data CSVs and the two frozen result files the scripts read (`results/static_utility/initial_routes_by_seed.json`, `results/weather_ablation/weather_ablation_raw_90_pairs.csv`). A script that writes `results/...` therefore cannot touch the real artefacts, and the previous working directory plus any mutated module constants are restored afterwards.

`test_static_utility_reproducibility`, `test_benchmark_determinism` and `test_weather_ablation_seed` are marked `slow` because they replay the real catalogue.

### 11.3 Frozen-artefact regressions (root `test_*.py`, 43 tests)

Collected directly by pytest because they already expose test functions:

| File | What it guards |
|---|---|
| `test_bounded_repair.py` | bounded-repair contract: locked prefix, current-activity semantics, bounded window, repair-record auditability, planner parity, hard-constraint enforcement (17 tests) |
| `test_stability_metrics.py` | itinerary stability: all eleven metrics, locked-prefix violation reporting, edit-distance weights and determinism, `build_metric_record` integration, real-pipeline prefix safety (17 tests) |
| `test_thermal_tradeoff_policy.py` | three `U*` baselines, alpha changing eligibility, independent guardrails, C2 priority (5 tests) |
| `test_c2_alpha_independence.py` | search independent of alpha, selection dependent on alpha, default-policy backward compatibility, v2 smoke isolation (4 tests) |

The remaining script-style modules (`test_tripmates_lite.py`, `test_thermal_selection_contract.py`, `test_adaptive_without_thermal.py`, `test_greedy_repair.py`, `test_benchmark_common.py`, `test_benchmark_determinism.py`, `test_smoke_output_isolation.py`, `test_itinerary_service.py`, `test_application_services.py`, `test_scenario_states.py`, `test_static_utility_reproducibility.py`) are reached through §11.2.

### 11.4 Two implementation fixes made while building the suite

Writing small fixtures exposed two defects that the 50-POI catalogue hides. Both fixes are unreachable for the frozen experiments, and `test_static_utility_reproducibility`, `test_benchmark_determinism` and `test_weather_ablation_seed` confirm the published numbers still reproduce byte-for-byte.

1. **`ga_core.initialise_population` could loop forever.** It filled the population with `while len(chromosomes) < population_size`, adding only *distinct* permutations of the optional POIs. A catalogue with three optional POIs admits only `3! = 6` distinct chromosomes, so a population of 10 never filled and the loop never terminated. The loop is now capped at `min(population_size, len(optional_ids)!)`. With 47 optional POIs the cap is never reached, so the frozen runs are unaffected. `adaptive_ga_core.initialise_adaptive_population` already permitted duplicates and needed no change.

2. **`scenario_core.simulate_replanning_state` rejected a trigger before the trip start.** The scan matched a trigger to a walking leg, a waiting interval or a visit; a trigger earlier than 09:00 matched none and raised `RuntimeError`. It now applies rule 6: nothing is committed, the traveller is still at the origin with `current_poi_requires_visit=True`, and the whole itinerary is the modifiable suffix. Every official trigger is at least 85 minutes after the start, so the branch is unreachable in the 18 frozen instances — verified by regenerating `results/scenario_states/seed_1001_replanning_states.csv` byte-identically.

---

## 12. How to extend

**Adding a planner**

1. obtain a `problem` from `adaptive_ga_core.prepare_adaptive_problem()` (it already contains the candidate window, mandatory backbone and locked prefix);
2. evaluate candidates only through `decode_replanned_route()` — **do not** write your own feasibility logic;
3. return an `EvaluatedSolution` whose `achieved_utility`, `completed_poi_count` and `total_travel_min` **include the locked prefix**;
4. register it in the `planner_id` dispatch of `itinerary_service.replan()`;
5. add the name to `PLANNERS` in `run_rq1_full_benchmark.py` and write a matching `*_planner_record()`;
6. add a `test_*.py` that guards its contract.

Stability metrics are **not** the planner's job: `benchmark_common.build_metric_record()` and `ItineraryService.replan()` both call `stability_metrics` centrally, so a new planner inherits the identical definition without writing any measurement code. Only supply a `RepairRecord` if the planner records per-POI removal/replacement reasons.

**Adding a scenario**

Add a `ScenarioDefinition` to `scenario_core.SCENARIOS` (offset plus thermal multiplier / network multiplier / closed POIs / congested zone); the instance count becomes 21 automatically. Then check the downstream effect on `benchmark_common.ALL_INSTANCE_IDS`.

**Changing GA parameters**

`GAConfig` is a frozen dataclass, so construct a new instance and pass it in (that is exactly what the UI Quick preview does). **Do not** change the defaults — every published number depends on them.

---

## 13. Known issues and caveats

| # | Issue | Impact | Recommendation |
|---|---|---|---|
| 1 | Under the frozen guardrail the thermal objective never takes effect (TripMates == Adaptive) | the central negative result | state it directly; use C2 at alpha = 0.85 as evidence that a trade-off exists |
| 2 | The candidate window is capped at 10 and prioritises mandatory POIs, leaving very little room after a disruption | the four-objective front has almost no slack | widen the window or lower the mandatory share if you want the thermal objective to matter |
| 3 | The 10 "seeds" are repeated stochastic optimisations, not 10 real disruptions | easy to misread statistically | treat the **instance** as the primary unit; use `overall_pairs` for description only (the code already says so) |
| 4 | The walking matrix is straight-line distance / 1.4 m/s, with no street network | walking time is underestimated | state this explicitly in the limitations chapter |
| 5 | `ga_core.GARunResult.feasible_solution_count` is actually the **deduplicated chromosome-cache size** | misleading name | do not read it as "number of feasible solutions" |
| 6 | The module docstring of `adaptive_ga_core.py` does not describe the file (copied from the candidate-window section) | documentation-only defect | safe to fix, no logic impact |
| 7 | `archive/overpass_cache/*.json` holds early Overpass responses and is **not referenced by any code** | retained provenance only | leave archived; do not use as planner input |
| 8 | Choosing "Research benchmark" in the UI runs 80x120 / 40x60 | tens of seconds of waiting | use Quick preview for demos |
| 9 | `data/itinerary_history.sqlite3` is runtime state | grows with use | it is not ignored by git; avoid committing it |
| 10 | `results/thermal_tradeoff_sensitivity/` (v1) had an alpha-dependent search | superseded by v2 | cite v2 only; keep v1 as history |

---

## 14. Glossary

| Term | Meaning |
|---|---|
| **POI** | point of interest, a visitor attraction |
| **backbone** | the optimal ordering of mandatory POIs onto which optional POIs are inserted |
| **decode_route** | translates an ordered POI sequence into a timestamped itinerary and decides feasibility |
| **candidate window** | the at most 10 POIs allowed into the search during replanning |
| **WES** | weather-weighted exposure score |
| **TOEM** | total outdoor exposure minutes = OWM + OVM |
| **OWM / OVM** | outdoor walking minutes / outdoor visiting minutes |
| **M_s** | scenario thermal multiplier (S1 = 1.8, S5 = 2.0, otherwise 1.0) |
| **U\*** | utility guardrail baseline (default: utility of the seed-matched non-thermal reference) |
| **alpha** | utility-retention threshold (default 0.90) |
| **guardrail fallback** | returning the non-thermal reference when no front member passes both guardrails |
| **locked prefix** | the completed or in-transit part before the trigger, which replanning must not modify |
| **nominal / early / late** | the three trigger variants (x1.00 / 0.95 / 1.05) |

---

*This guide was written by reading every file in the repository at its current state (`main` branch); all figures come from the artefacts under `results/`.*

Candidate-window sensitivity uses 5, 10, 15, and all-eligible candidates while holding the maximum modifiable suffix at 10. Utility sensitivity is exploratory and perturbs the implemented composite `utility_score` because source rating/popularity/zone component columns are not present in the current dataset. Both preserve matched inputs, explicit failures, manifests, and smoke isolation.
