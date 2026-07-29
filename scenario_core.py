"""
Controlled disruption scenarios and trigger-state simulation.

This module:

1. Defines the six frozen disruption families.
2. Generates early, nominal and late trigger instances.
3. Simulates the traveller's state at each trigger.
4. Locks completed visits.
5. Completes an in-progress walking leg or visit before replanning.
6. Calculates the remaining mandatory POIs.
7. Produces travel adjustments and unavailable-POI sets.

No replanning algorithm is implemented in this file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Literal

from planning_core import (
    PlanningData,
    ScheduleEntry,
    ScheduleResult,
    TravelAdjustments,
    TRIP_START_MIN,
    format_clock,
)


TriggerVariant = Literal[
    "early",
    "nominal",
    "late",
]


# ================================================================
# Scenario definitions
# ================================================================

@dataclass(frozen=True)
class ScenarioDefinition:
    scenario_id: str
    name: str

    nominal_trigger_offset_min: int

    thermal_multiplier: float = 1.0
    global_travel_multiplier: float = 1.0

    closed_poi_ids: frozenset[int] = frozenset()

    congested_zone: str | None = None
    congestion_multiplier: float = 1.0

    purpose: str = ""

    def build_travel_adjustments(
        self,
        data: PlanningData,
    ) -> TravelAdjustments:
        """
        Construct the post-trigger travel adjustments.

        Network-wide movement delay and the combined scenario
        use a global multiplier.

        Congestion applies only to directed links whose origin
        and destination both belong to the configured zone.
        """

        link_multipliers: dict[
            tuple[int, int],
            float,
        ] = {}

        if self.congested_zone is not None:
            zone = (
                self.congested_zone
                .strip()
                .upper()
            )

            for from_poi in data.pois.values():
                for to_poi in data.pois.values():
                    if from_poi.poi_id == to_poi.poi_id:
                        continue

                    if (
                        from_poi.zone == zone
                        and to_poi.zone == zone
                    ):
                        link_multipliers[
                            (
                                from_poi.poi_id,
                                to_poi.poi_id,
                            )
                        ] = self.congestion_multiplier

        return TravelAdjustments(
            global_multiplier=(
                self.global_travel_multiplier
            ),
            link_multipliers=link_multipliers,
        )


SCENARIOS: dict[str, ScenarioDefinition] = {
    "S1": ScenarioDefinition(
        scenario_id="S1",
        name="Heavy Rain",
        nominal_trigger_offset_min=120,
        thermal_multiplier=1.8,
        purpose=(
            "Test avoidance of exposed walking "
            "during rainfall."
        ),
    ),

    "S2": ScenarioDefinition(
        scenario_id="S2",
        name="Network-wide Movement Delay",
        nominal_trigger_offset_min=90,
        global_travel_multiplier=1.5,
        purpose=(
            "Test a broad movement disruption "
            "without claiming a specific transit model."
        ),
    ),

    "S3": ScenarioDefinition(
        scenario_id="S3",
        name="Attraction Closure",
        nominal_trigger_offset_min=150,
        closed_poi_ids=frozenset({13}),
        purpose=(
            "Test removal and substitution of "
            "an infeasible attraction."
        ),
    ),

    "S4": ScenarioDefinition(
        scenario_id="S4",
        name="Congestion",
        nominal_trigger_offset_min=135,
        congested_zone="A",
        congestion_multiplier=1.3,
        purpose=(
            "Test local movement delay without "
            "removing a POI."
        ),
    ),

    "S5": ScenarioDefinition(
        scenario_id="S5",
        name="Heatwave",
        nominal_trigger_offset_min=180,
        thermal_multiplier=2.0,
        purpose=(
            "Test heat-related exposure reduction."
        ),
    ),

    "S6": ScenarioDefinition(
        scenario_id="S6",
        name="Combined",
        nominal_trigger_offset_min=120,
        thermal_multiplier=1.8,
        global_travel_multiplier=1.3,
        closed_poi_ids=frozenset({34}),
        purpose=(
            "Test simultaneous changes to exposure, "
            "movement time and attraction availability."
        ),
    ),
}


# ================================================================
# Scenario instances
# ================================================================

@dataclass(frozen=True)
class ScenarioInstance:
    scenario: ScenarioDefinition
    trigger_variant: TriggerVariant

    trigger_offset_min: int
    trigger_absolute_min: int

    @property
    def instance_id(self) -> str:
        return (
            f"{self.scenario.scenario_id}_"
            f"{self.trigger_variant}"
        )

    @property
    def trigger_clock(self) -> str:
        return format_clock(
            self.trigger_absolute_min
        )


def round_half_up(value: Decimal) -> int:
    """
    Round to the nearest whole minute using half-up rounding.

    This avoids Python's built-in banker's rounding.
    """

    return int(
        value.quantize(
            Decimal("1"),
            rounding=ROUND_HALF_UP,
        )
    )


def calculate_trigger_offset(
    nominal_offset_min: int,
    variant: TriggerVariant,
) -> int:
    factors = {
        "early": Decimal("0.95"),
        "nominal": Decimal("1.00"),
        "late": Decimal("1.05"),
    }

    if variant not in factors:
        raise ValueError(
            f"Unknown trigger variant: {variant}"
        )

    value = (
        Decimal(nominal_offset_min)
        * factors[variant]
    )

    return round_half_up(value)


def build_scenario_instances(
) -> tuple[ScenarioInstance, ...]:
    instances: list[ScenarioInstance] = []

    variants: tuple[TriggerVariant, ...] = (
        "early",
        "nominal",
        "late",
    )

    for scenario in SCENARIOS.values():
        for variant in variants:
            offset = calculate_trigger_offset(
                scenario.nominal_trigger_offset_min,
                variant,
            )

            instances.append(
                ScenarioInstance(
                    scenario=scenario,
                    trigger_variant=variant,
                    trigger_offset_min=offset,
                    trigger_absolute_min=(
                        TRIP_START_MIN
                        + offset
                    ),
                )
            )

    return tuple(instances)


# ================================================================
# Replanning state
# ================================================================

@dataclass(frozen=True)
class ReplanningState:
    instance_id: str
    scenario_id: str
    scenario_name: str
    trigger_variant: str

    trigger_offset_min: int
    trigger_absolute_min: int
    trigger_clock: str

    committed_component: str

    current_poi_id: int
    current_poi_requires_visit: bool

    replan_start_min: float
    replan_start_clock: str

    completed_poi_ids: tuple[int, ...]
    original_remaining_poi_ids: tuple[int, ...]

    remaining_mandatory_ids: tuple[int, ...]
    unavailable_mandatory_ids: tuple[int, ...]

    cost_spent_rm: float
    utility_achieved: float

    unavailable_poi_ids: tuple[int, ...]

    thermal_multiplier: float
    global_travel_multiplier: float

    route_finished_before_trigger: bool

    def to_dict(self) -> dict:
        return asdict(self)


# ================================================================
# Trigger-state simulation
# ================================================================

def simulate_replanning_state(
    data: PlanningData,
    initial_schedule: ScheduleResult,
    scenario_instance: ScenarioInstance,
) -> ReplanningState:
    """
    Simulate the traveller's state at the trigger.

    Rules:

    1. Visits completed before the trigger remain locked.
    2. If a walking leg has started, it is completed using its
       original pre-trigger duration. Replanning starts from its
       destination, but the destination visit remains unfinished.
    3. If a visit has started, it is completed and locked.
    4. If the traveller is waiting, replanning starts at the
       trigger from the POI at which the traveller is waiting.
    5. Post-trigger closures and travel multipliers apply only
       to the unfinished route.
    """

    if not initial_schedule.feasible:
        raise ValueError(
            "The initial schedule must be feasible."
        )

    entries = list(
        initial_schedule.entries
    )

    if not entries:
        raise ValueError(
            "The initial schedule contains no visits."
        )

    trigger_min = (
        scenario_instance
        .trigger_absolute_min
    )

    completed_entries: list[
        ScheduleEntry
    ] = []

    committed_component = "none"

    current_poi_id = (
        initial_schedule.start_poi_id
    )

    current_poi_requires_visit = False

    replan_start_min = float(
        trigger_min
    )

    remaining_start_index = len(
        entries
    )

    route_finished = False

    tolerance = 1e-9

    for index, entry in enumerate(entries):
        # The visit was fully completed before or exactly
        # at the trigger.
        if (
            entry.visit_end_min
            <= trigger_min + tolerance
        ):
            completed_entries.append(
                entry
            )

            current_poi_id = (
                entry.poi_id
            )

            continue

        previous_visit_end = (
            entries[index - 1].visit_end_min
            if index > 0
            else initial_schedule.finish_min
            - initial_schedule.time_used_min
        )

        travel_start_min = (
            previous_visit_end
        )

        # --------------------------------------------------------
        # Trigger during an in-progress walking leg
        # --------------------------------------------------------

        if (
            entry.travel_min > 0
            and travel_start_min
            <= trigger_min + tolerance
            and trigger_min
            < entry.arrival_min - tolerance
        ):
            committed_component = (
                "walking_leg"
            )

            # The walking leg is completed and locked.
            # Its destination visit has not yet started.
            current_poi_id = (
                entry.poi_id
            )

            current_poi_requires_visit = (
                True
            )

            replan_start_min = (
                entry.arrival_min
            )

            remaining_start_index = index
            break

        # --------------------------------------------------------
        # Trigger while waiting for the POI to open
        # --------------------------------------------------------

        if (
            entry.arrival_min
            <= trigger_min + tolerance
            and trigger_min
            < entry.visit_start_min - tolerance
        ):
            committed_component = "waiting"

            current_poi_id = (
                entry.poi_id
            )

            current_poi_requires_visit = (
                True
            )

            replan_start_min = float(
                trigger_min
            )

            remaining_start_index = index
            break

        # --------------------------------------------------------
        # Trigger during a visit
        # --------------------------------------------------------

        if (
            entry.visit_start_min
            <= trigger_min + tolerance
            and trigger_min
            < entry.visit_end_min - tolerance
        ):
            committed_component = "visit"

            # The in-progress visit is completed and locked.
            completed_entries.append(
                entry
            )

            current_poi_id = (
                entry.poi_id
            )

            current_poi_requires_visit = (
                False
            )

            replan_start_min = (
                entry.visit_end_min
            )

            remaining_start_index = (
                index + 1
            )

            break

        raise RuntimeError(
            "The trigger could not be mapped to the "
            "initial schedule timeline."
        )

    else:
        # Every planned visit was completed before the trigger.
        route_finished = True

        current_poi_id = (
            entries[-1].poi_id
        )

        current_poi_requires_visit = False

        replan_start_min = float(
            trigger_min
        )

        remaining_start_index = len(
            entries
        )

    completed_poi_ids = tuple(
        entry.poi_id
        for entry in completed_entries
    )

    original_remaining_poi_ids = tuple(
        entry.poi_id
        for entry in entries[
            remaining_start_index:
        ]
    )

    scenario = (
        scenario_instance.scenario
    )

    unavailable_poi_ids = set(
        scenario.closed_poi_ids
    )

    completed_id_set = set(
        completed_poi_ids
    )

    unavailable_mandatory_ids = (
        set(data.mandatory_ids)
        & unavailable_poi_ids
        - completed_id_set
    )

    remaining_mandatory_ids = (
        set(data.mandatory_ids)
        - completed_id_set
        - unavailable_poi_ids
    )

    cost_spent_rm = sum(
        entry.admission_cost_rm
        for entry in completed_entries
    )

    utility_achieved = sum(
        data.pois[
            poi_id
        ].utility_score
        for poi_id in completed_poi_ids
    )

    return ReplanningState(
        instance_id=(
            scenario_instance.instance_id
        ),

        scenario_id=(
            scenario.scenario_id
        ),

        scenario_name=(
            scenario.name
        ),

        trigger_variant=(
            scenario_instance
            .trigger_variant
        ),

        trigger_offset_min=(
            scenario_instance
            .trigger_offset_min
        ),

        trigger_absolute_min=(
            trigger_min
        ),

        trigger_clock=format_clock(
            trigger_min
        ),

        committed_component=(
            committed_component
        ),

        current_poi_id=(
            current_poi_id
        ),

        current_poi_requires_visit=(
            current_poi_requires_visit
        ),

        replan_start_min=(
            replan_start_min
        ),

        replan_start_clock=format_clock(
            replan_start_min
        ),

        completed_poi_ids=(
            completed_poi_ids
        ),

        original_remaining_poi_ids=(
            original_remaining_poi_ids
        ),

        remaining_mandatory_ids=tuple(
            sorted(
                remaining_mandatory_ids
            )
        ),

        unavailable_mandatory_ids=tuple(
            sorted(
                unavailable_mandatory_ids
            )
        ),

        cost_spent_rm=round(
            cost_spent_rm,
            6,
        ),

        utility_achieved=round(
            utility_achieved,
            6,
        ),

        unavailable_poi_ids=tuple(
            sorted(
                unavailable_poi_ids
            )
        ),

        thermal_multiplier=(
            scenario.thermal_multiplier
        ),

        global_travel_multiplier=(
            scenario
            .global_travel_multiplier
        ),

        route_finished_before_trigger=(
            route_finished
        ),
    )