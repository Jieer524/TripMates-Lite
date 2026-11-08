"""Shared fixtures for the pytest suite.

Everything in this module is synthetic and deterministic.  The suite
never reads the 50-POI research catalogue, never reads a generated
result file and never touches the network, so it stays fast and stays
independent of the frozen experiment artefacts.

The synthetic world is a six-POI Kuala Lumpur toy instance:

    id  name               zone  mandatory  opening      visit  cost
    --  -----------------  ----  ---------  -----------  -----  ----
     1  Origin Square      A     yes        24 hours       30    0.0
     2  Heritage Museum    A     yes        09:00-12:00    60   10.0
     3  Riverside Park     B     no         09:00-17:00    45    0.0
     4  Sky Tower          A     yes        10:00-18:00    60   20.0
     5  Late Gallery       B     no         13:00-17:00    30    5.0
     6  Night Market       B     no         closed WED     30    0.0

POI 6 is deliberately closed on Wednesday (the experiment day) so the
"closed on the selected day" branch is reachable.

The canonical initial itinerary is ``(1, 2, 3, 5, 4)``.  Its timeline is
fixed and known, which lets every disruption test place a trigger at an
exact minute:

    09:00-09:30  POI 1
    09:40-10:40  POI 2      (10 min walk from POI 1)
    11:00-11:45  POI 3      (20 min walk from POI 2)
    11:55-13:00  waiting at POI 5 (opens at 13:00)
    13:00-13:30  POI 5
    14:00-15:00  POI 4      (30 min walk from POI 5)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# ------------------------------------------------------------------
# Make the repository importable no matter where pytest is invoked
# from.  ``tests/`` has no ``__init__.py``, so pytest would otherwise
# put only ``tests/`` on ``sys.path``.
# ------------------------------------------------------------------

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from planning_core import (  # noqa: E402
    POI,
    PlanningConstraints,
    PlanningData,
    ScheduleResult,
    decode_route,
)
from scenario_core import (  # noqa: E402
    ReplanningState,
    ScenarioDefinition,
    ScenarioInstance,
    simulate_replanning_state,
)


# ================================================================
# Synthetic catalogue
# ================================================================

TRIP_START_MIN = 9 * 60
TRIP_END_MIN = 17 * 60
EXPERIMENT_DAY = "WED"
COST_BUDGET_RM = 160.0

INITIAL_ROUTE: tuple[int, ...] = (1, 2, 3, 5, 4)

MANDATORY_POI_IDS: tuple[int, ...] = (1, 2, 4)


def _poi(
    poi_id: int,
    name: str,
    zone: str,
    *,
    mandatory: bool,
    opening: int,
    closing: int,
    visit_duration_min: float,
    admission_cost_rm: float,
    utility_score: float,
    is_24_hours: bool = False,
    off_days: frozenset[str] = frozenset(),
    indoor_ratio: float = 0.5,
    shelter_ratio: float = 0.3,
) -> POI:
    return POI(
        poi_id=poi_id,
        name=name,
        category="Synthetic",
        latitude=3.14 + poi_id * 0.001,
        longitude=101.69 + poi_id * 0.001,
        zone=zone,
        mandatory=mandatory,
        off_days=off_days,
        is_24_hours=is_24_hours,
        opening_minute=opening,
        closing_minute=closing,
        closes_next_day=False,
        visit_duration_min=visit_duration_min,
        admission_cost_rm=admission_cost_rm,
        utility_score=utility_score,
        indoor_ratio=indoor_ratio,
        shelter_ratio=shelter_ratio,
    )


def build_synthetic_pois() -> dict[int, POI]:
    pois = [
        _poi(
            1,
            "Origin Square",
            "A",
            mandatory=True,
            opening=0,
            closing=23 * 60 + 59,
            visit_duration_min=30.0,
            admission_cost_rm=0.0,
            utility_score=8.0,
            is_24_hours=True,
            indoor_ratio=0.2,
            shelter_ratio=0.1,
        ),
        _poi(
            2,
            "Heritage Museum",
            "A",
            mandatory=True,
            opening=9 * 60,
            closing=12 * 60,
            visit_duration_min=60.0,
            admission_cost_rm=10.0,
            utility_score=7.0,
            indoor_ratio=0.8,
        ),
        _poi(
            3,
            "Riverside Park",
            "B",
            mandatory=False,
            opening=9 * 60,
            closing=17 * 60,
            visit_duration_min=45.0,
            admission_cost_rm=0.0,
            utility_score=6.0,
            indoor_ratio=0.1,
        ),
        _poi(
            4,
            "Sky Tower",
            "A",
            mandatory=True,
            opening=10 * 60,
            closing=18 * 60,
            visit_duration_min=60.0,
            admission_cost_rm=20.0,
            utility_score=9.0,
            indoor_ratio=0.7,
        ),
        _poi(
            5,
            "Late Gallery",
            "B",
            mandatory=False,
            opening=13 * 60,
            closing=17 * 60,
            visit_duration_min=30.0,
            admission_cost_rm=5.0,
            utility_score=5.0,
            indoor_ratio=0.6,
        ),
        _poi(
            6,
            "Night Market",
            "B",
            mandatory=False,
            opening=9 * 60,
            closing=16 * 60,
            visit_duration_min=30.0,
            admission_cost_rm=0.0,
            utility_score=4.0,
            off_days=frozenset({EXPERIMENT_DAY}),
            indoor_ratio=0.0,
            shelter_ratio=0.1,
        ),
    ]

    return {poi.poi_id: poi for poi in pois}


# Symmetric hand-written walking times.  Zone A -> A links are short,
# B -> B links are short, and cross-zone links are longer.  Every value
# is distinct enough that no two routes tie by accident.
TRAVEL_MINUTES: dict[tuple[int, int], float] = {
    (1, 2): 10.0,
    (1, 3): 20.0,
    (1, 4): 25.0,
    (1, 5): 30.0,
    (1, 6): 20.0,
    (2, 3): 20.0,
    (2, 4): 15.0,
    (2, 5): 30.0,
    (2, 6): 25.0,
    (3, 4): 25.0,
    (3, 5): 10.0,
    (3, 6): 10.0,
    (4, 5): 30.0,
    (4, 6): 25.0,
    (5, 6): 10.0,
}


def build_walking_matrix() -> pd.DataFrame:
    poi_ids = sorted(build_synthetic_pois())

    values = []
    for from_id in poi_ids:
        row = []
        for to_id in poi_ids:
            if from_id == to_id:
                row.append(0.0)
            else:
                key = (min(from_id, to_id), max(from_id, to_id))
                row.append(TRAVEL_MINUTES[key])
        values.append(row)

    return pd.DataFrame(
        values,
        index=poi_ids,
        columns=[str(poi_id) for poi_id in poi_ids],
    )


# ================================================================
# Fixtures
# ================================================================


@pytest.fixture(scope="session")
def planning_data() -> PlanningData:
    """A deterministic six-POI planning instance."""

    return PlanningData(
        pois=build_synthetic_pois(),
        walking_matrix=build_walking_matrix(),
    )


@pytest.fixture(scope="session")
def constraints() -> PlanningConstraints:
    return PlanningConstraints(
        start_poi_id=1,
        start_time_min=TRIP_START_MIN,
        trip_end_min=TRIP_END_MIN,
        day_code=EXPERIMENT_DAY,
        cost_budget_rm=COST_BUDGET_RM,
    )


@pytest.fixture(scope="session")
def initial_schedule(
    planning_data: PlanningData,
    constraints: PlanningConstraints,
) -> ScheduleResult:
    """The canonical, feasible ``(1, 2, 3, 5, 4)`` itinerary."""

    schedule = decode_route(
        data=planning_data,
        visit_sequence=INITIAL_ROUTE,
        start_poi_id=constraints.start_poi_id,
        start_time_min=constraints.start_time_min,
        trip_end_min=constraints.trip_end_min,
        day_code=constraints.day_code,
        cost_budget_rm=constraints.cost_budget_rm,
        start_poi_is_visit=True,
        required_mandatory_ids=MANDATORY_POI_IDS,
    )

    assert schedule.feasible, schedule.failure_reason

    return schedule


# ================================================================
# Scenario helpers
# ================================================================


def make_scenario(
    *,
    scenario_id: str | None = None,
    name: str = "Synthetic probe",
    trigger_offset_min: int = 60,
    trigger_variant: str = "nominal",
    thermal_multiplier: float = 1.0,
    global_travel_multiplier: float = 1.0,
    closed_poi_ids: frozenset[int] | tuple[int, ...] = frozenset(),
    congested_zone: str | None = None,
    congestion_multiplier: float = 1.0,
) -> ScenarioInstance:
    """Build a one-off ``ScenarioInstance`` on the synthetic timeline.

    ``trigger_offset_min`` is relative to 09:00, so offset 35 means a
    trigger at 09:35.

    ``scenario_id`` defaults to a string derived from the disruption
    parameters.  This matters because ``scenario_core`` memoises
    ``build_travel_adjustments`` under ``(id(data), scenario_id)`` and
    therefore assumes one id maps to exactly one definition.  Two probes
    that share an id but differ in parameters would silently share the
    first probe's travel adjustments.
    """

    closed = frozenset(int(poi_id) for poi_id in closed_poi_ids)

    if scenario_id is None:
        scenario_id = (
            f"TX_t{thermal_multiplier:g}"
            f"_g{global_travel_multiplier:g}"
            f"_c{'-'.join(str(poi_id) for poi_id in sorted(closed)) or 'none'}"
            f"_z{congested_zone or 'none'}{congestion_multiplier:g}"
            f"_o{trigger_offset_min}"
        )

    definition = ScenarioDefinition(
        scenario_id=scenario_id,
        name=name,
        nominal_trigger_offset_min=trigger_offset_min,
        thermal_multiplier=thermal_multiplier,
        global_travel_multiplier=global_travel_multiplier,
        closed_poi_ids=closed,
        congested_zone=congested_zone,
        congestion_multiplier=congestion_multiplier,
    )

    return ScenarioInstance(
        scenario=definition,
        trigger_variant=trigger_variant,  # type: ignore[arg-type]
        trigger_offset_min=trigger_offset_min,
        trigger_absolute_min=TRIP_START_MIN + trigger_offset_min,
    )


def replanning_state_at(
    planning_data: PlanningData,
    initial_schedule: ScheduleResult,
    trigger_offset_min: int,
    **scenario_kwargs,
) -> tuple[ReplanningState, ScenarioInstance]:
    """Simulate the traveller's state at a given trigger offset."""

    instance = make_scenario(
        trigger_offset_min=trigger_offset_min,
        **scenario_kwargs,
    )

    state = simulate_replanning_state(
        data=planning_data,
        initial_schedule=initial_schedule,
        scenario_instance=instance,
    )

    return state, instance


# ================================================================
# Timings used by the tests (absolute minutes after midnight)
# ================================================================

TIMELINE = {
    "poi_1_visit_start": 9 * 60,
    "poi_1_visit_end": 9 * 60 + 30,
    "poi_2_arrival": 9 * 60 + 40,
    "poi_2_visit_end": 10 * 60 + 40,
    "poi_3_arrival": 11 * 60,
    "poi_3_visit_end": 11 * 60 + 45,
    "poi_5_arrival": 11 * 60 + 55,
    "poi_5_visit_start": 13 * 60,
    "poi_5_visit_end": 13 * 60 + 30,
    "poi_4_arrival": 14 * 60,
    "poi_4_visit_end": 15 * 60,
}

# Trigger offsets (minutes after 09:00) for the four commit points.
TRIGGER_BEFORE_START = -30
TRIGGER_DURING_FIRST_VISIT = 0
TRIGGER_DURING_WALKING = 35
TRIGGER_DURING_VISIT = 60
TRIGGER_DURING_WAITING = 210
