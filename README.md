# TripMates Lite

A research prototype for single-day walking-itinerary planning with post-disruption replanning in Kuala Lumpur. The core question: **if weather-weighted outdoor exposure (WES) is added as a fourth Pareto objective to replanning, can outdoor exposure be reduced without materially sacrificing itinerary quality?**

> Full documentation — data contracts, module-by-module walkthrough, every experiment, result interpretation, known issues — is in **[REPO_GUIDE.md](REPO_GUIDE.md)**.

## Controlled gap studies

The additional studies are exploratory and isolated under `results/ablation_studies/`:

| Study | Levels | Full-run cells |
|---|---|---:|
| No-stability-penalty | `run_stability_ablation.py`: frozen, unguarded, stability-guarded, stability-unguarded | 10 seeds × 6 nominal instances × 2 adaptive planners × 4 policies |
| Full-replanning ablation | bounded suffix vs full-replanning upper bound | 10 × 6 × 2 policies × 2 adaptive planners |
| Disruption severity | light / medium / severe for travel delay, closure, thermal escalation, and combined disruption | 10 × 6 nominal instances × 12 levels × 5 planners |
| Itinerary density | relaxed / moderate / dense | 10 × 6 × 3 × 5 planners |
| Weather profile | normal / hot / very hot / afternoon escalation / missing segment | 10 × 6 × 5 × 5 planners |

Run with `python run_gap_experiments.py --study all` (or `--smoke`). Each study writes CSV, JSON manifest, failure cases, input hashes, and exploratory paired-test outputs without touching official results.

## Final validation package

The reproducibility audit and derived thesis tables are generated with:

```text
python -m pytest -q
python audit_checks.py
python run_rq1_full_benchmark.py --smoke
python run_weather_ablation_all_seeds.py --smoke
python run_thermal_tradeoff_sensitivity.py --smoke
python run_gap_experiments.py --study all --smoke
python final_research_audit.py --allow-missing-official
python build_final_research_tables.py
```

`final_research_audit.py` writes `output/research_final/reproducibility_audit.json` and exits non-zero for critical schema/configuration problems unless `--allow-missing-official` is used. `build_final_research_tables.py` writes derived tables and focused figures to `output/research_final/`. Smoke artefacts are not official evidence; full official runs must be completed before final thesis claims.

---

## What it does

Given a 09:00–17:00 window, an admission budget of RM 160 and three mandatory POIs, the planner builds a feasible walking itinerary over 50 Kuala Lumpur POIs. When a disruption occurs mid-trip, only the **unfinished suffix** is replanned:

| Scenario | Disruption | Parameters |
|---|---|---|
| S1 | Heavy Rain | thermal multiplier 1.8 |
| S2 | Movement Delay | network-wide walking x1.5 |
| S3 | Attraction Closure | POI 13 closed |
| S4 | Congestion | zone-A internal links x1.3 |
| S5 | Heatwave | thermal multiplier 2.0 |
| S6 | Combined | 1.8 + x1.3 + POI 34 closed |

Each scenario x `early / nominal / late` trigger = **18 instances**.

**Five controlled planners**

| Planner | Adaptive | Stochastic | Uses thermal info | One-line summary |
|---|---|---|---|---|
| Static Distance | no | no | no | nearest-neighbour greedy, replays the plan after disruption |
| Static Utility | no | yes | no | 80x120 GA for the initial route, no adaptation afterwards |
| Greedy Repair | yes | no | no | greedy repair by utility per minute |
| Adaptive Without Thermal | yes | yes | no | bounded GA, 3 objectives |
| **TripMates Lite** | yes | yes | yes | bounded GA, **4 objectives (+WES)** with guardrailed selection |

All adaptive planners share **the same decoder** and **the same candidate window**. Differences are confined to the objective function and the final route selection — the precondition for a fair ablation.

---

## Bounded replanning — formal definition

When a disruption occurs mid-trip, the replanner operates under an **enforceable bounded-repair contract**.  The contract has ten rules:

| # | Rule | Enforcement |
|---|---|---|
| 1 | **Immutable prefix.** Visits completed before the trigger are locked. | `ReplanningState.completed_poi_ids` is excluded from the candidate window and the repaired route. |
| 2 | **Current-activity semantics.** If the traveller is walking, waiting or visiting at the trigger, the existing scenario rules apply. | `simulate_replanning_state` completes a committed walking leg; a committed visit is locked; waiting starts replanning at the trigger time. |
| 3 | **Suffix-only modification.** Only the unfinished suffix may be changed. | `decode_replanned_route` starts from the post-trigger state; the prefix is never re-decoded. |
| 4 | **Separate search window and modification cap.** `candidate_window_size` controls optimizer candidates; `max_modifiable_suffix_activities` caps the repaired suffix. | `BoundedRepairConfig` records and enforces both values; the default is 10 for each. |
| 5 | **Configurable and recorded.** The bound is in every experiment configuration and every `RepairRecord`. | `BoundedRepairConfig.to_dict()` is written to run configs; `RepairRecord.bounded_repair_limit` is in every metric record. |
| 6 | **Decision audit.** Removed, skipped, shifted and replacement POIs are recorded with a reason. | `RepairDecision` entries in `RepairRecord.removed_activities` and `RepairRecord.replacements`. |
| 7 | **Explicit failure.** No feasible bounded repair → structured failure, never a fabricated route. | `build_failure_repair_record` returns `failure_status="infeasible"` with a reason. |
| 8 | **Hard-constraint enforcement.** Opening hours, budget, mandatory POIs, route, duplication and end-time are never bypassed. | The single decoder `planning_core.decode_route` is the only feasibility gate. |
| 9 | **Shared decoder.** All adaptive planners use the same decoder and hard constraints. | `decode_replanned_route` wraps `decode_route` for every planner. |
| 10 | **Planner parity.** Adaptive Without Thermal and TripMates Lite use identical candidate pools, search budgets and bounded-repair constraints. | Both call `prepare_adaptive_problem` with the same `BoundedRepairConfig`. |

### Repair record structure

Every repair produces a `RepairRecord` with these fields:

| Field | Meaning |
|---|---|
| `locked_prefix` | POI IDs completed before the trigger (immutable). |
| `current_activity` | What the traveller was doing at the trigger (`walking_leg`, `waiting`, `visit`, `none`). |
| `modified_suffix` | POI IDs in the repaired suffix (in order). |
| `retained_activities` | Original suffix POIs kept in the repair. |
| `removed_activities` | Original suffix POIs removed, each with a `RepairDecision` reason. |
| `replacements` | New POIs introduced, each with a `RepairDecision` reason. |
| `failure_status` | `"success"`, `"no_action"` or `"infeasible"`. |
| `failure_reason` | Human-readable reason on failure, else `None`. |
| `bounded_repair_limit` | The configured `max_modifiable_suffix_activities`. |
| `actual_modification_count` | Number of suffix activities in the repair. |

### Worked example

```
Original itinerary (seed 1001, S1 nominal):
  1 -> 2 -> 12 -> 4 -> 33 -> 34 -> 36 -> 13

Disruption: Heavy Rain at 11:00 (trigger = 660 min)

Locked prefix (completed before trigger):
  [1, 2]

Current activity at trigger:
  visit (POI 12 is in progress, completed and locked)

Candidate repair window (max 10, mandatory first):
  mandatory: [4, 13]
  optional:  [36, 12, 33, 34, 18, ...]  (top-scored, feasibility-gated)

Final itinerary:
  locked: [1, 2, 12]
  repaired suffix: [34, 4, 33, 36, 13]

RepairRecord:
  locked_prefix:              (1, 2, 12)
  current_activity:           visit
  modified_suffix:            (34, 4, 33, 36, 13)
  retained_activities:        (4, 33, 36, 13)
  removed_activities:         ()  — no original suffix POI was dropped
  replacements:               (34,)  — re-inserted from candidate window
  failure_status:             success
  bounded_repair_limit:       10
  actual_modification_count:  5
```

---

## Itinerary stability — how far did the repair move the plan?

Utility and WES describe *what the repaired itinerary achieves*. `stability_metrics.py` describes *how much it deviates from the original plan*. Every planner is measured by the same function, so the comparison is like-for-like.

The comparison is always **original suffix → repaired suffix** from the same post-trigger state:

```
original itinerary  = locked_prefix + state.original_remaining_poi_ids
replanned itinerary = locked_prefix + planner route
```

| # | Metric | Exact definition |
|---|---|---|
| 1 | `preserved_poi_count` | `\|original_suffix ∩ replanned_suffix\|` by POI identity (order-independent). |
| 2 | `preserved_poi_percentage` | `preserved_poi_count / \|original_suffix\|`; `None` when the original suffix is empty. |
| 3 | `removed_pois` | Original-suffix POIs absent from the repair, in original order. |
| 4 | `added_pois` | Repaired-suffix POIs absent from the original suffix, in repaired order (insertions / replacements). |
| 5 | `changed_activity_count` | `len(removed) + len(added) + len(shifted)`; each POI counts once. |
| 6 | `time_shift_count` | Preserved POIs whose `visit_start_min` moved by more than 1e-6 min; `None` when no schedule exists. |
| 7 | `order_deviation` | Kendall tau distance (discordant pairs) between the two *retained subsequences*; `0` when relative order survives insertions. |
| 8 | `edit_distance_score` | Weighted sum of the five components below. |
| 9 | `modified_suffix_activity_count` | `len(replanned_suffix)` — the quantity capped by `BoundedRepairConfig`. |
| 10 | `locked_prefix_changed` | `True` when a completed visit reappears in the repair, or a supplied final route has a different prefix. |
| 11 | `removed_poi_reasons` / `replacement_poi_reasons` | `RepairDecision` reasons from the `RepairRecord`, when the planner recorded them. |

### How the locked prefix is handled

The prefix is **verified, never corrected, and never counted as a modification**. If a repaired suffix re-executes a completed POI, the result carries `locked_prefix_changed = True`, `hard_stability_violation = True` and a readable `violation_reasons` entry — and the offending POI still appears in `added_pois`. Nothing is silently repaired.

### How the edit distance is calculated

```
edit_distance = 1.00 * removed_pois
              + 1.00 * added_pois
              + 0.50 * order_deviation
              + 0.25 * time_shift_count
              + 1000 * locked_prefix_changed
```

| Component | Weight | Rationale |
|---|---|---|
| removed POI | 1.00 | Base unit: one planned visit lost. |
| inserted POI | 1.00 | Symmetric unit: one itinerary slot differs. |
| order change | 0.50 | Reordering keeps every POI, so it costs half an identity change. |
| time shift | 0.25 | A retained POI that only moves in time is the mildest change. |
| locked-prefix violation | 1000.00 | Must dominate any permitted soft change. |

The weights live in `StabilityWeightConfig` (frozen dataclass, named constants), are recorded in every run config and every metric record, and can be swapped for a sensitivity analysis. The violation weight is justified rather than arbitrary: inside the 10-activity window the maximum soft cost is `2.25n + 0.25n(n-1) = 45.0` (`maximum_soft_edit_distance`), so 1000 always dominates.

The score is deterministic: every component is an integer count, ordering follows execution order (never set iteration), and all floats are rounded to six decimals.

### Reading the new benchmark columns

- **`preserved_poi_count` / `preserved_poi_percentage`** — plan retention. Higher is more faithful.
- **`removed_pois` / `added_pois`** — JSON lists; `added_pois` is what was substituted in.
- **`order_deviation`** — pure reordering, independent of insertions. `0` with non-zero `added_pois` means new stops were slotted in without disturbing the retained order.
- **`time_shift_count` / `total_time_shift_min`** — `None` when the repair was infeasible and no schedule exists. Never silently zero.
- **`edit_distance_score`** — single comparable number across planners. `0.0` = the repair reproduced the original plan exactly.
- **`hard_stability_violation`** — must be `False` for every conforming run; any `True` is a contract breach to investigate, not a soft result.

---

## Key results (TL;DR)

- **Under the frozen policy (alpha = 0.90) the thermal objective never changes the selected route.** All 90 paired ablation runs show a WES difference of exactly zero, and across 180 RQ1 runs TripMates and Adaptive produce identical utility, walking time and WES (the guardrail falls back to the non-thermal reference every time).
- **A trade-off appears only when utility retention is relaxed.** In the C2 sensitivity run, alpha = 0.85 reduces WES by **35.2 min on average** (median 53.1, effect size +1.00, Holm-adjusted p ~ 1.7e-14) at the cost of utility and visit count; alpha = 1.00 is actually 14.1 min worse.
- **Adaptive replanning improves recovery quality.** Under the predeclared skip-and-continue static rule, both static baselines remain feasible, but Adaptive Without Thermal and TripMates Lite achieve higher itinerary completion and utility in the matched benchmark.
- **The cost is latency.** Greedy ~3 ms, Adaptive ~0.5–1.7 s, TripMates ~3.5 s (S1_early, 30 repetitions).

---

## Quick start

```bash
pip install -r requirements.txt          # pandas>=2.0  scipy>=1.11  streamlit>=1.40

streamlit run app.py                     # web app (Quick preview by default, seconds)
```

Running the experiments from the command line:

```bash
python static_utility_baseline.py        # required first: produces the 10 initial routes
python static_distance_baseline.py
python run_weather_ablation_all_seeds.py # main ablation: 90 pairs
python analyze_weather_ablation.py
python run_rq1_full_benchmark.py         # RQ1: 738 runs (slow)
python analyze_rq1_full_benchmark.py
python run_thermal_tradeoff_sensitivity.py
python analyze_thermal_tradeoff_sensitivity.py
python run_latency_benchmark.py
python audit_checks.py
```

Every runner supports `--smoke`, and **smoke output goes to `results/<exp>/smoke/`** so official results are never overwritten.

---

## Project structure

```
.
├── poi_data_calculated_final.csv        # 50 POIs x 31 columns
├── walking_time_matrix_final.csv        # 50x50 symmetric walking-minute matrix
│
├── planning_core.py                     # data models + decode_route (single feasibility decoder)
├── scenario_core.py                     # six disruptions / 18 instances / trigger-state simulation
├── exposure_core.py                     # outdoor exposure and WES
├── stability_metrics.py                 # deterministic itinerary-stability metrics + edit distance
│
├── ga_core.py                           # static-utility GA + generic operators
├── adaptive_ga_core.py                  # bounded replanning GA + shared candidate window
├── greedy_repair.py                     # deterministic greedy baseline
├── tripmates_lite_ga.py                 # thermal-aware GA (WES as 4th objective)
│
├── app.py                               # Streamlit front end
├── itinerary_service.py                 # application service layer
├── weather_service.py                   # Open-Meteo with three-level fallback
├── itinerary_history.py                 # SQLite itinerary history
│
├── run_*.py / analyze_*.py / static_*_baseline.py / audit_checks.py
├── test_*.py                            # frozen-artefact contract / regression suites
├── tests/                               # pytest-native synthetic suite (conftest + 4 modules + legacy wrappers)
├── pytest.ini                           # rootdir / sys.path / slow marker
└── results/                             # all experiment artefacts
```

---

## Experiments

| Script | Design | Scale | Key artefacts |
|---|---|---|---|
| `static_utility_baseline.py` | 10 seeds x 80x120 GA, frozen initial routes | 10 | `results/static_utility/` |
| `static_distance_baseline.py` | nearest neighbour + mandatory-feasibility gate | 1 | `static_distance_normal_*` |
| `run_weather_ablation_all_seeds.py` | S1/S5/S6 x 3 triggers x 10 seeds, paired planners | 90 pairs | `results/weather_ablation/` |
| `run_rq1_full_benchmark.py` | 5 planners x 18 instances | 738 runs | `results/rq1_full_benchmark_corrected_static_rule/` |
| `run_thermal_tradeoff_sensitivity.py` | 9 instances x 10 seeds x 4 alphas | 360 runs | `results/thermal_tradeoff_sensitivity_alpha_independent_v2/` |
| `run_latency_benchmark.py` | 1 warm-up + 30 timed repetitions, 3 scopes | 3 planners x 18 | `results/latency_benchmark/` |
| `audit_checks.py` | 12 dataset checks (schema / formula / matrix / hashes) | — | `dataset_validation_report.json` |
| `diagnose_thermal_diversity.py` | why the thermal objective never changes the route (2000 sampled chromosomes) | — | console |

---

## Results summary

**RQ1 full benchmark** (`results/rq1_full_benchmark_corrected_static_rule/rq1_overall_planner_summary.csv`)

| Planner | Feasible | Mean utility | Mean walking min | Mean WES | Median runtime |
|---|---|---|---|---|---|
| Static Distance | 100.0% | 44.17 | 62.00 | 251.46 | 0.4 ms |
| Static Utility | 100.0% | 54.44 | 66.28 | 285.86 | 0.4 ms |
| Greedy Repair | 100% | 59.42 | 100.06 | 353.38 | 11.6 ms |
| Adaptive Without Thermal | 100% | 63.88 | 68.87 | 310.50 | 1.78 s |
| TripMates Lite | 100% | 63.88 | 68.87 | 310.50 | 3.09 s |

**C2 alpha sensitivity** (WES advantage, positive = TripMates better)

| alpha | WES advantage | Effect size | Adjusted p |
|---|---|---|---|
| 1.00 | -14.11 | -1.00 | 4.1e-16 |
| 0.95 | 0.00 | 0.00 | 1.0 |
| 0.90 | 0.00 | 0.00 | 1.0 |
| 0.85 | **+35.19** | +1.00 | 1.7e-14 |

---

## Web application

```bash
streamlit run app.py
```

- The sidebar shows live Open-Meteo weather (rain maps to S1; temperature at least 35 C or temperature at least 32 C with relative humidity at least 70% maps to S5; a scenario can also be chosen manually).
- After planning: a pydeck map with a slider that reveals the route stop by stop, a two-column timeline, and metric cards. After replanning, the completed/committed portion and revised unfinished portion are labelled separately.
- Choose `Quick preview` (12x12, seconds) or `Research benchmark` (80x120 / 40x60, tens of seconds).
- Trips can be named and saved to `data/itinerary_history.sqlite3`, then reopened or deleted under "Saved itineraries".

---

## Data and formulas

- **Utility** `U1 = (2*rating + 10*log10(n+1)/log10(max_n+1) + zone_score) / 3`, with `zone_score = A:10 / B:8 / C:6`.
- **Walking time** = equirectangular straight-line distance / 1.4 m/s / 60 (**no street network**; a lower-bound proxy). The repository checks matrix shape, symmetry, finite positive connections and route-level connectivity; it does **not** implement physical accessibility features such as wheelchair access, step-free routing, stairs, lifts, slopes or barrier-aware pedestrian navigation.
- **Thermal-exposure proxy (WES):** for POI `j`, `indoor_ratio_j` is the whole-visit indoor proportion and `shelter_ratio_j` is the conditional proportion of the remaining outdoor visit fraction that is sheltered or shaded. The exposed fraction is `X_j = (1 - indoor_ratio_j)(1 - shelter_ratio_j)`. Walking exposure is walking minutes; visit exposure is `visit_minutes × X_j`. Waiting is explicitly excluded by default (and remains available as a separate optional metric), rather than being silently treated as outdoor exposure. For any activity interval, `weighted = pre_trigger_minutes + M_s × post_trigger_minutes`, so `WES = pre_trigger_exposure + M_s × post_trigger_exposure`. This is a planning proxy based on researcher-assigned itinerary metadata, not a medically validated, physiological, or health-risk model.
- Weather provenance is recorded as `live`, `cached`, `predefined`, or `unavailable`, with a fallback reason. Temperature and, when present, relative humidity can classify a profile as hot; missing values use the deterministic predefined benchmark profile or explicit unavailable neutral mode, never zero exposure. `exposure_core.py` also exposes total, high-heat, maximum-continuous, travel, visit, and optional waiting exposure metrics.
- All data is verified by `audit_checks.py`; see `dataset_validation_report.json`.

---

## Known limitations

1. **The thermal objective has no effect under the frozen policy** (a finding, not a defect); the candidate window is capped at 10 and prioritises mandatory POIs, leaving the front almost no slack.
2. **The 10 "seeds" are repeated stochastic optimisations, not 10 real disruptions**; the instance is the primary statistical unit.
3. The walking matrix contains no street network, so walking time is underestimated.
4. Early Overpass snapshots are retained for provenance under `archive/overpass_cache/` and are not referenced by the application.
5. The repository contains no `LICENSE` file; contact the author before academic reuse.

---

## Tests

```bash
pytest -q                 # full suite: 117 tests, includes 3 slow frozen-artefact replays
pytest -q -m "not slow"   # 114 tests, skips the 3 frozen-artefact replays
```

`pytest.ini` puts the repository root on `sys.path` and registers the `slow` marker. The suite has three layers:

| Layer | Location | Count | What it is |
|---|---|---|---|
| Synthetic contract suite | `tests/test_feasibility.py`, `test_disruptions.py`, `test_repair_behavior.py`, `test_planner_contracts.py` | 56 | pytest-native, six-POI toy instance, no file or network I/O, runs in under a second |
| Legacy script wrappers | `tests/test_legacy_script_checks.py` | 11 | runs each original `main()` unchanged inside a throw-away working directory |
| Frozen-artefact regressions | `test_bounded_repair.py`, `test_stability_metrics.py`, `test_c2_alpha_independence.py`, `test_thermal_tradeoff_policy.py` | 43 | recompute against the real 50-POI catalogue and the frozen result files |

**Category A — feasibility** (15 tests): valid itinerary, late arrival, closed before arrival, visit ending after closing, visit ending after the 17:00 horizon, non-overlapping activities, duplicate POI, missing route, unavailable POI, budget exceeded, missing mandatory POI, unknown POI ID, closed-on-day, wrong start POI, unavailable-mandatory reporting.

**Category B — disruption and replanning state** (11 tests): trigger before the first activity, at trip start, during walking, during waiting, during a visit, locked prefix, in-transit leg semantics, closed POI excluded from the suffix, delayed travel, zone congestion scoping, combined disruption.

**Category C — repair behaviour** (20 tests): greedy feasible replacement, explicit failure (two independent causes), all optional POIs removed, mandatory POIs never silently dropped, locked-prefix immutability across four commit points, modification limit (valid and invalid values), no duplicate replacements, and the five stability-metric shapes (unchanged / removed / inserted / shifted / reordered) plus a hard prefix violation.

**Category D — planner contracts** (10 tests): shared candidate window, shared decoder, thermal utility+visit guardrails, alpha-independent search, static-baseline reproducibility, smoke-output isolation, seeded determinism, decoder idempotence.

The synthetic fixture (`tests/conftest.py`) is a six-POI catalogue with a fixed timeline, so a trigger can be placed at an exact minute (09:35 is mid-walk, 12:30 is mid-wait, 10:00 is mid-visit). No test needs live Open-Meteo access; the weather adapter is exercised with an injected `fetcher`.

```bash
python test_bounded_repair.py            # bounded-repair contract (17 tests)
python test_stability_metrics.py         # itinerary stability + edit distance (17 tests)
python test_c2_alpha_independence.py     # alpha-independence regression
python test_thermal_tradeoff_policy.py   # selection-policy contract
python test_greedy_repair.py             # greedy determinism
python test_smoke_output_isolation.py    # smoke must not overwrite official results
```

---

## Documentation

- **[REPO_GUIDE.md](REPO_GUIDE.md)** — architecture, function-level module notes, experiment design and artefacts, result interpretation, extension guide
- `documents/` — thesis drafts (local, not tracked)
- `output/` — presentation decks and figures (local, not tracked)

### Exploratory sensitivity studies

Candidate-window sensitivity is available with `python run_candidate_window_sensitivity.py --smoke`; levels 5, 10, 15, and all eligible candidates are written to `results/ablation_studies/candidate_window_sensitivity/`. The modification limit remains independently fixed at 10, so 10 remains the proposed configuration and 15/all-eligible are exploratory search-space conditions. Utility-weight sensitivity is available with `python run_utility_weight_sensitivity.py --smoke` and writes to `results/sensitivity/utility_weights/`. These studies test search-space access and robustness to researcher-assigned utility assumptions; neither replaces the confirmatory benchmark or licenses retroactive tuning.

Full official sensitivity outputs are now available in the corresponding result directories. The planned skip-and-continue static execution rule is the canonical RQ1 protocol. The archived legacy replay results remain available under `results/rq1_full_benchmark/` for comparison. Evidence appendices are under `output/research_final/`.
