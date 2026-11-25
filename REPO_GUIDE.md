# TripMates Lite: code and experiment guide

This guide connects the FYP2 study to the implementation and archived experiment records. [README.md](README.md) is the short project entry point. TripMates Lite is a bounded itinerary-repair workflow for one-day walking trips in Kuala Lumpur. Its research contribution is the integration of realised-state reconstruction, prefix preservation, bounded suffix search, shared feasibility checking, auditable repair decisions, and a weather-exposure selection policy.

## 1. Research design

The study asks how bounded repair compares with fixed-order execution in completion and original-POI retention; how frozen and relaxed WES-based selection affects route choice, modelled exposure, utility, and visits; and what latency the tested GA and finite-domain search configurations exhibit. The experiments are controlled computational comparisons, not a traveller study.

The benchmark uses 50 curated Kuala Lumpur POIs, a coordinate-derived 50 × 50 walking-time matrix, a Wednesday 09:00–17:00 horizon, an RM160 compulsory-admission budget, and origin POI 1. The original mandatory set is `{1, 4, 13}`. Initial routes are generated with ten seeds, `1001`–`1010`; Static Distance has one deterministic route. Six scripted disruption families have early, nominal, and late trigger variants, producing 18 scenario instances. The six families are the design-level interpretation units; variants and seeds are repeated computational tests.

| Family | Nominal trigger | Post-trigger change |
|---|---:|---|
| S1 Heavy Rain | 120 min | Thermal multiplier 1.8 |
| S2 Movement Delay | 90 min | Remaining walking times ×1.5 |
| S3 Attraction Closure | 150 min | Mandatory POI 13 unavailable |
| S4 Zone A Congestion | 135 min | Eligible Zone A links ×1.3 |
| S5 Heatwave | 180 min | Thermal multiplier 2.0 |
| S6 Combined | 120 min | Rain ×1.8, movement ×1.3, optional POI 34 unavailable |

The early and late triggers use 0.95 and 1.05 times each nominal offset. Scenario generation and the traveller's post-trigger state are implemented in [`scenario_core.py`](scenario_core.py).

## 2. Data and feasibility contracts

| Input | Role |
|---|---|
| [`poi_data_calculated_final.csv`](poi_data_calculated_final.csv) | POI identity, coordinates, hours, cost, duration, utility components, and exposure annotations |
| [`walking_time_matrix_final.csv`](walking_time_matrix_final.csv) | Shared symmetric walking-time proxy |
| [`results/static_utility/initial_routes_by_seed.json`](results/static_utility/initial_routes_by_seed.json) | Seed-matched initial routes used in the comparative experiments |

The utility score is the stored composite of a normalised rating term, a logarithmic review-count term, and a zone score. The catalogue retains source rating and review-count fields as well as the derived score; [`audit_checks.py`](audit_checks.py) recomputes and checks the formula. The movement matrix uses an equirectangular coordinate-distance approximation at 1.4 m/s. It does not represent crossings, barriers, slopes, pedestrian-network detours, or public transport.

[`planning_core.decode_route`](planning_core.py) is the common feasibility gate. It checks the daily horizon, opening hours, compulsory-admission budget, available mandatory visits, unique POIs, and valid matrix links. A candidate route is accepted only after this decoder returns a feasible schedule. When a mandatory original POI closes, original-mandatory completion can fall even if a feasible optional substitute preserves executable route length.

## 3. Bounded repair workflow

1. Generate and decode a feasible initial route before departure.
2. At a disruption trigger, [`scenario_core.simulate_replanning_state`](scenario_core.py) reconstructs location, elapsed time, completed visits, any committed walking leg or visit, and the remaining route.
3. Freeze completed or committed progress. [`adaptive_ga_core.prepare_adaptive_problem`](adaptive_ga_core.py) forms a shared candidate window, prioritises remaining mandatory POIs, and prescreens optional candidates for mandatory-backbone feasibility.
4. Search and decode routes for the unfinished suffix. The default candidate window and maximum modifiable suffix are both ten activities. [`adaptive_ga_core.decode_replanned_route`](adaptive_ga_core.py) uses the common decoder from the reconstructed state.
5. Apply the route-selection policy and return a [`RepairRecord`](adaptive_ga_core.py) containing the locked prefix, current activity, retained and removed activities, replacements, modification count, and an explicit failure status when no feasible repair exists.

The GA proposes candidate orderings; the deterministic decoder decides whether each proposed route is executable. Initial utility search uses a population of 80 and at most 120 generations. The research replanning configuration uses a population of 40 and at most 60 generations, with tournament selection, order crossover, swap mutation, elitism, and a 15-generation stagnation limit. The recorded run configuration contains the exact parameters.

### Planner permissions

| Planner | Initial route | Permitted post-trigger edits | Search |
|---|---|---|---|
| Static Distance | Deterministic distance-based route | Delete unfinished entries; retain original order | Minimum-deletion fixed-order execution |
| Static Utility | Seed-matched utility GA route | Delete unfinished entries; retain original order | Minimum-deletion fixed-order execution |
| Greedy Repair | Same seed-matched utility route | Select, reorder, and substitute within the bounded window | Deterministic utility-per-added-minute rule |
| Single-route AILS adaptation | Same seed-matched utility route | Select, reorder, and substitute within the bounded window | Feasible insertion, exchange/reversal, and seeded shaking |
| Adaptive Without Thermal | Same seed-matched utility route | Select, reorder, and substitute within the bounded window | GA without thermal selection |
| TripMates Lite | Same seed-matched utility route | Select, reorder, and substitute within the bounded window | Same GA configuration with thermal selection |

The fixed-order comparator keeps the largest feasible order-preserving subsequence of the unfinished original route; ties favour higher utility, lower travel time, then lower POI ID. It does not reorder or insert substitutes. Static-versus-adaptive results compare complete mechanisms with different edit permissions. The matched comparison between the two GA planners isolates the thermal selection rule because they share the state, initial route, window, decoder, GA settings, and seed. The AILS condition adapts a published method family to a single post-trigger route; it is not a reproduction of the original multi-route algorithm. Its implementation and configuration are in [`ils_repair_baseline.py`](ils_repair_baseline.py) and [`results/rq1_ils_external_comparison/run_config.json`](results/rq1_ils_external_comparison/run_config.json).

### Stability records

[`stability_metrics.py`](stability_metrics.py) compares the original unfinished suffix with the repaired suffix. Retention counts original POIs present after repair, irrespective of order. The edit-distance score adds one point per removed or inserted POI, 0.5 per retained-POI order-deviation pair, 0.25 per shifted visit, and a 1,000-point locked-prefix violation term. The stability ablation changes the post-search selection key; the published primary planners do not use an active stability penalty.

## 4. Exposure and route selection

[`exposure_core.py`](exposure_core.py) calculates modelled outdoor walking and visit exposure. A POI's exposed visit fraction is `(1 - indoor_ratio) × (1 - shelter_ratio)`, where shelter is conditional on the outdoor share. WES weights exposure before the trigger by 1 and exposure after the trigger by the scenario thermal multiplier. Waiting exposure is excluded from the primary proxy because the catalogue has no waiting-location annotation.

Under normal conditions, the two adaptive GA planners use the same non-thermal objective vector. Under adverse thermal conditions, TripMates Lite adds WES to feasible-route evaluation and applies utility and completed-visit guardrails before final selection. The frozen primary utility guardrail uses `α = 0.90` relative to the seed-matched non-thermal reference route. The finite-domain front sensitivity uses the maximum utility on its finite thermal front as a separate anchor. These two analyses should not be combined as though their `α` values had the same reference.

WES is a comparative planning proxy based on researcher-assigned annotations and scenario multipliers. It is not PET, UTCI, a physiological risk score, or a validated measure of route safety.

## 5. Evaluation files and results

### Primary route comparison

[`results/rq1_full_benchmark_minimum_deletion_primary/`](results/rq1_full_benchmark_minimum_deletion_primary/) contains 738 primary rows: 198 separately evaluated minimum-deletion static rows and 540 unchanged Greedy Repair and adaptive rows. Its [`rq1_run_config.json`](results/rq1_full_benchmark_minimum_deletion_primary/rq1_run_config.json) records both source artifacts and identifies the output as an assembled reanalysis. [`results/rq1_ils_external_comparison/`](results/rq1_ils_external_comparison/) adds 180 matched single-route AILS rows, giving 918 combined rows. All 918 combined rows are feasible.

| Planner | Runs | Mean ICR | Mean OPR | Mean utility | Mean travel (min) | Mean cost (RM) |
|---|---:|---:|---:|---:|---:|---:|
| Static Distance | 18 | 0.9167 | 0.9167 | 45.638 | 62.046 | 117.500 |
| Static Utility | 180 | 0.9375 | 0.9375 | 62.755 | 69.869 | 135.417 |
| Greedy Repair | 180 | 0.8958 | 0.8500 | 59.421 | 100.058 | 135.833 |
| Adaptive Without Thermal | 180 | 0.9583 | 0.9056 | 63.878 | 68.868 | 128.889 |
| TripMates Lite | 180 | 0.9583 | 0.9056 | 63.878 | 68.868 | 128.889 |
| Single-route AILS adaptation | 180 | 0.9236 | 0.7556 | 63.425 | 77.647 | 140.833 |

ICR is the itinerary-completion ratio and credits a feasible substitute; OPR is original-POI retention. The adaptive planners' mean ICR is 0.0208 above minimum-deletion Static Utility, entirely due to S3 closure (family difference 0.1250). The other five families tie in ICR. Static Utility retains more original POIs overall (0.9375 versus 0.9056), and the unavailable original mandatory POI remains incomplete in S3 even when an optional substitute is inserted.

### Thermal selection and finite-domain search

The frozen `α = 0.90` policy changes none of the 90 weather-related matched primary routes. [`results/exact_front_thermal_ablation/`](results/exact_front_thermal_ablation/) enumerates the finite ten-POI domain for the recorded weather-related instances and reports separate front-anchored guardrail sensitivity:

| Front `α` | Changed routes | Median WES reduction | Median utility loss | Median visit loss |
|---:|---:|---:|---:|---:|
| 0.90 | 0/90 | 0.00% | 0.00% | 0 |
| 0.85 | 80/90 | 14.01% | 11.18% | 1 |
| 0.80 | 80/90 | 14.01% | 12.41% | 1 |

The predefined joint criterion requires at least 15% median WES reduction, at most 10% median utility loss, and no more than one lost visit. The relaxed settings meet the visit limit but miss the WES and utility limits. The finite-domain enumerator selects the same routes as the GA in all 90 tested comparisons within this bounded window; this does not establish relative scalability at larger windows. The separate exploratory GA alpha sensitivity is in [`results/thermal_tradeoff_sensitivity_alpha_independent_v2/`](results/thermal_tradeoff_sensitivity_alpha_independent_v2/).

### Replanning latency

The dedicated benchmark in [`results/latency_benchmark/`](results/latency_benchmark/) times 30 repetitions per planner and scenario instance after a warm-up, excluding data loading and using a pre-built disruption state. The research-configuration engine-only results are:

| Planner | Median | P95 |
|---|---:|---:|
| Adaptive Without Thermal | 361.710 ms | 2,161.575 ms |
| TripMates Lite | 714.746 ms | 2,413.600 ms |

Both medians meet the 1,000 ms engineering target; both P95 values exceed 2,000 ms. In the paired finite-domain search comparison, median runtimes are 37.996 ms for the enumerator and 876.510 ms for the GA. The dedicated research benchmark and finite-domain timing are different protocols, so their figures are not a paired speed ratio.

### Exploratory analyses

[`results/ablation_studies/`](results/ablation_studies/) contains stability selection, full replanning, disruption-severity, itinerary-density, weather-profile, and candidate-window analyses. They describe configuration sensitivity and are interpreted separately from the primary route comparison. The full-replanning condition releases the realised prefix; it is a different planning scope, not the deployable bounded-repair rule. Candidate windows of 5, 10, 15, and all eligible POIs are explored under a fixed GA budget, so a lower utility at a larger window does not prove the ten-POI window is an exact optimum.

## 6. Application and code map

| Module | Responsibility |
|---|---|
| [`app.py`](app.py) | Streamlit trip input, map, timeline, replanning controls, and saved-itinerary views |
| [`itinerary_service.py`](itinerary_service.py) | Application request validation and planning/replanning service |
| [`planning_core.py`](planning_core.py) | POI data models, constraints, data loading, and the shared feasibility decoder |
| [`scenario_core.py`](scenario_core.py) | Disruption definitions, trigger variants, and realised-state reconstruction |
| [`adaptive_ga_core.py`](adaptive_ga_core.py) | Bounded candidate window, adaptive GA, repair decoding, and `RepairRecord` |
| [`tripmates_lite_ga.py`](tripmates_lite_ga.py) | Thermal Pareto evaluation, guardrails, and route selection |
| [`exposure_core.py`](exposure_core.py) | Outdoor exposure and WES calculation |
| [`stability_metrics.py`](stability_metrics.py) | Retention and itinerary-edit metrics |
| [`itinerary_history.py`](itinerary_history.py) | Local saved-itinerary storage |

The local browser create → replan → save → reopen workflow is recorded in [`results/web_e2e_evaluation/functional_report.md`](results/web_e2e_evaluation/functional_report.md). That check demonstrates those functions in one tested local configuration; it is not a usability study.

## 7. Run and verify

Install dependencies and start the web application:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Check the catalogue, matrix, and code contracts:

```bash
python audit_checks.py
python -m pytest -q
```

The experiment runners are [`run_rq1_full_benchmark.py`](run_rq1_full_benchmark.py), [`run_minimal_static_analysis.py`](run_minimal_static_analysis.py), [`assemble_rq1_minimum_deletion_primary.py`](assemble_rq1_minimum_deletion_primary.py), [`run_ils_external_comparison.py`](run_ils_external_comparison.py), [`run_weather_ablation_all_seeds.py`](run_weather_ablation_all_seeds.py), [`run_exact_front_thermal_ablation.py`](run_exact_front_thermal_ablation.py), and [`run_latency_benchmark.py`](run_latency_benchmark.py). To reconstruct the reported primary comparison, generate the initial routes, generate the static minimum-deletion rows, and run the assembly script against its recorded nonstatic source file. Each result directory contains the relevant configuration and raw or summary records. Some full runners overwrite their documented result files; inspect their output paths before rerunning an archive.

The [versioned research archive](https://github.com/Jieer524/TripMates-Lite/tree/research-artifacts-full-raw-2026-11-23) contains the published source and evidence package. The archived results use controlled inputs, a coordinate-derived movement proxy, and scripted disruptions; they do not establish pedestrian-network travel times, physiological heat safety, or performance across cities and user populations.
