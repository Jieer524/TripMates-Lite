# TripMates Lite

TripMates Lite is a research prototype for planning and repairing one-day walking itineraries in Kuala Lumpur. Its central contribution is a bounded, auditable replanning workflow: after a disruption, it reconstructs the traveller's realised state, preserves completed or committed activities, and repairs the unfinished part of the route. A shared feasibility decoder checks opening hours, the daily time limit, admission cost, mandatory visits, POI uniqueness, and travel links.

The FYP2 study evaluates three aspects of that workflow: itinerary completion and original-POI retention after bounded repair, the effect of weather-exposure selection on route choice, and replanning latency within the tested search window.

## How replanning works

1. Build an initial feasible itinerary from the POI catalogue and trip constraints.
2. At a disruption trigger, reconstruct the current location, elapsed time, completed visits, and any committed activity.
3. Lock realised progress and form a candidate window for the unfinished suffix, prioritising remaining mandatory POIs.
4. Search for feasible suffix routes with the shared decoder. Under adverse weather, TripMates Lite includes Weather-Exposure Score (WES) in route selection subject to utility and visit guardrails.
5. Return a revised route and a `RepairRecord` that identifies retained, removed, and replacement activities or explains infeasibility.

The default candidate window and maximum modifiable suffix each contain at most ten activities. WES compares modelled outdoor walking and visit exposure; it is not a physiological heat-risk measure.

## Evaluation setup

The controlled benchmark contains 50 curated Kuala Lumpur points of interest (POIs), a coordinate-derived 50 × 50 walking-time matrix, a 09:00–17:00 planning window, and an RM160 compulsory-admission budget. POI 1 is the fixed origin; POIs 1, 4, and 13 form the original mandatory set. The walking matrix uses an equirectangular distance approximation and a walking speed of 1.4 m/s. It is a reproducible movement proxy, not a pedestrian street network.

Six scripted disruption families are each tested at early, nominal, and late trigger times:

| Family | Disruption |
|---|---|
| S1 | Heavy rain; thermal multiplier 1.8 |
| S2 | Network-wide movement delay; remaining walking times ×1.5 |
| S3 | Attraction closure; mandatory POI 13 becomes unavailable |
| S4 | Zone A congestion; eligible links ×1.3 |
| S5 | Heatwave; thermal multiplier 2.0 |
| S6 | Combined rain, movement delay, and optional POI 34 closure |

### Comparison methods

The study uses **minimum-deletion execution** for its fixed-order baselines. After freezing realised progress, this rule keeps the largest feasible order-preserving subsequence of the unfinished original route. It cannot reorder or insert a substitute. Bounded repair methods may select, reorder, and substitute POIs in the shared ten-POI candidate window.

| Planner | Role after a disruption |
|---|---|
| Static Distance | Deterministic distance-based initial route, then minimum-deletion execution |
| Static Utility | Seed-matched GA initial route, then minimum-deletion execution |
| Greedy Repair | Deterministic bounded repair using utility per added minute |
| Single-route AILS adaptation | Bounded local-search comparator adapted from an iterated-local-search method |
| Adaptive Without Thermal | Bounded GA repair without WES-based route selection |
| TripMates Lite | The same bounded GA configuration with thermal-aware route selection |

The two GA repair planners share their initial routes, candidate window, decoder, operators, configuration, and seeds. Their matched comparison isolates the thermal selection policy. Comparisons between fixed-order and adaptive planners evaluate complete repair mechanisms with different edit permissions.

## Main findings

The evaluation characterises the bounded repair workflow under scripted disruptions. Its completion benefit over fixed-order execution is concentrated in one closure family, while the thermal and search analyses identify how route selection and runtime behave in the tested configuration.

The five-planner primary comparison contains **738 runs**: 198 minimum-deletion static rows and 540 Greedy Repair and adaptive rows. An additional **180** matched runs evaluate the single-route AILS adaptation, giving **918** combined rows. Static Distance runs once per trigger variant; the other planners use ten seed-matched initial routes per variant. All 918 combined rows are feasible. The six disruption families are the result-interpretation units; trigger variants and seeds are repeated computational tests.

| Planner | Runs | Mean ICR | Mean OPR | Mean utility | Mean travel (min) |
|---|---:|---:|---:|---:|---:|
| Static Distance | 18 | 0.9167 | 0.9167 | 45.638 | 62.046 |
| Static Utility (minimum-deletion) | 180 | 0.9375 | 0.9375 | 62.755 | 69.869 |
| Greedy Repair | 180 | 0.8958 | 0.8500 | 59.421 | 100.058 |
| Adaptive Without Thermal | 180 | 0.9583 | 0.9056 | 63.878 | 68.868 |
| TripMates Lite | 180 | 0.9583 | 0.9056 | 63.878 | 68.868 |
| Single-route AILS adaptation | 180 | 0.9236 | 0.7556 | 63.425 | 77.647 |

ICR is the itinerary-completion ratio; it credits an executable substitute. OPR measures retention of original POIs. The adaptive planners' **0.0208** mean ICR difference over minimum-deletion Static Utility occurs entirely in S3 attraction closure. A substitute restores executable route length there, but the unavailable original mandatory POI remains incomplete. Static Utility has higher overall OPR (**0.9375** versus **0.9056**), so completion and original-itinerary retention are distinct outcomes.

The frozen thermal policy (`α = 0.90`) changed **0 of 90** weather-related matched routes relative to Adaptive Without Thermal. In the separate finite-domain front analysis, relaxing `α` to **0.85** changed 80 of 90 routes and reduced median Weather-Exposure Score (WES) by **14.01%**, with **11.18%** median utility loss and one fewer visit. The predefined joint targets of at least 15% WES reduction and at most 10% utility loss were not met. The primary GA policy and finite-front sensitivity use different utility anchors; their results are reported separately.

In the dedicated engine-only latency benchmark, Adaptive Without Thermal had a median/P95 of **361.710/2,161.575 ms** and TripMates Lite **714.746/2,413.600 ms**. Both met the 1,000 ms median target and exceeded the 2,000 ms P95 target. Within the tested ten-POI window, a memoised finite-domain enumerator selected the same routes as the GA in **90/90** comparisons, with lower paired median runtime (**37.996** versus **876.510 ms**). These timing results describe the recorded machine and protocols.

WES is a comparative planning proxy based on walking and visit durations, researcher-assigned indoor and shelter ratios, and scenario multipliers. It is not PET, UTCI, a physiological heat-risk measure, or evidence that a route is medically safer.

## Run the application

```bash
pip install -r requirements.txt
streamlit run app.py
```

The Streamlit interface accepts trip inputs, displays an itinerary on a map and timeline, triggers replanning, and can save and reopen a run. One local create → replan → save → reopen check is documented in [`results/web_e2e_evaluation/functional_report.md`](results/web_e2e_evaluation/functional_report.md).

## Research files and reproduction

| Purpose | File or directory |
|---|---|
| POI catalogue and walking matrix | [`poi_data_calculated_final.csv`](poi_data_calculated_final.csv), [`walking_time_matrix_final.csv`](walking_time_matrix_final.csv) |
| Primary minimum-deletion comparison | [`results/rq1_full_benchmark_minimum_deletion_primary/`](results/rq1_full_benchmark_minimum_deletion_primary/) |
| Additional AILS comparison and combined summary | [`results/rq1_ils_external_comparison/`](results/rq1_ils_external_comparison/) |
| Finite-domain thermal analysis | [`results/exact_front_thermal_ablation/`](results/exact_front_thermal_ablation/) |
| Dedicated latency benchmark | [`results/latency_benchmark/`](results/latency_benchmark/) |
| Exploratory ablations and sensitivity analyses | [`results/ablation_studies/`](results/ablation_studies/) |
| Detailed architecture and experiment guide | [`REPO_GUIDE.md`](REPO_GUIDE.md) |

The [versioned research archive](https://github.com/Jieer524/TripMates-Lite/tree/research-artifacts-full-raw-2026-11-23) records the source, inputs, configurations, generation scripts, and raw results used for the study. The minimum-deletion primary package is an assembled analysis: `run_minimal_static_analysis.py` produces the 198 fixed-order rows, and `assemble_rq1_minimum_deletion_primary.py` combines them with 540 unchanged Greedy Repair and adaptive rows. The separate AILS runner adds 180 matched rows. The run configurations record this provenance.

For a local data and code check:

```bash
python audit_checks.py
python -m pytest -q
```

Full experiment scripts can be computationally expensive and may write to their documented result directories. See [`REPO_GUIDE.md`](REPO_GUIDE.md) and the configuration files in each result directory before regenerating archived runs.
