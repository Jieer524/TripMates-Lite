from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import math
import pandas as pd


# ================================================================
# Fixed experiment settings
# ================================================================

EXPECTED_POI_COUNT = 50

# Absolute minutes after midnight
TRIP_START_MIN = 9 * 60
TRIP_END_MIN = 17 * 60

COST_BUDGET_RM = 160.0
EXPERIMENT_DAY = "WED"

VALID_DAYS = {
    "MON",
    "TUE",
    "WED",
    "THU",
    "FRI",
    "SAT",
    "SUN",
}


# ================================================================
# Parsing helpers
# ================================================================


def parse_yes_no(value: object, field_name: str) -> bool:
    text = str(value).strip().upper()

    if text in {"YES", "TRUE", "1"}:
        return True

    if text in {"NO", "FALSE", "0"}:
        return False

    raise ValueError(f"{field_name} must use YES or NO. Received: {value!r}")


def parse_clock(value: object) -> int:
    """
    Convert HH:MM into absolute minutes after midnight.

    Example:
        09:30 -> 570
    """

    text = str(value).strip()
    parts = text.split(":")

    if len(parts) != 2:
        raise ValueError(f"Invalid HH:MM value: {value!r}")

    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except ValueError as error:
        raise ValueError(f"Invalid HH:MM value: {value!r}") from error

    if not 0 <= hour <= 23:
        raise ValueError(f"Invalid hour: {hour}")

    if not 0 <= minute <= 59:
        raise ValueError(f"Invalid minute: {minute}")

    return hour * 60 + minute


def format_clock(absolute_minute: float) -> str:
    """
    Convert an absolute minute value into HH:MM.

    Values after midnight are displayed with D+1.
    """

    rounded = int(round(absolute_minute))
    day_offset, minute_of_day = divmod(
        rounded,
        24 * 60,
    )

    hour, minute = divmod(
        minute_of_day,
        60,
    )

    if day_offset == 0:
        return f"{hour:02d}:{minute:02d}"

    return f"D+{day_offset} {hour:02d}:{minute:02d}"


def parse_off_days(value: object) -> frozenset[str]:
    text = str(value).strip().upper()

    if text in {
        "",
        "NONE",
        "N/A",
        "NA",
        "NAN",
    }:
        return frozenset()

    text = text.replace("/", ",").replace(";", ",")

    days = frozenset(part.strip() for part in text.split(",") if part.strip())

    invalid_days = days - VALID_DAYS

    if invalid_days:
        raise ValueError(f"Invalid off-day values: {sorted(invalid_days)}")

    return days


# ================================================================
# Data models
# ================================================================


@dataclass(frozen=True)
class POI:
    poi_id: int
    name: str
    category: str
    latitude: float
    longitude: float
    zone: str

    mandatory: bool
    off_days: frozenset[str]

    is_24_hours: bool
    opening_minute: int
    closing_minute: int
    closes_next_day: bool

    visit_duration_min: float
    admission_cost_rm: float
    utility_score: float

    indoor_ratio: float
    shelter_ratio: float

    def get_opening_window(
        self,
        day_code: str,
    ) -> tuple[int, int] | None:
        """
        Return opening and closing minutes.

        None means that the POI is closed on the chosen day.
        """

        day = day_code.strip().upper()

        if day not in VALID_DAYS:
            raise ValueError(f"Invalid day code: {day_code}")

        if day in self.off_days:
            return None

        if self.is_24_hours:
            return 0, 24 * 60

        opening = self.opening_minute
        closing = self.closing_minute

        if self.closes_next_day:
            closing += 24 * 60

        return opening, closing


@dataclass(frozen=True)
class PlanningData:
    pois: dict[int, POI]
    walking_matrix: pd.DataFrame

    @property
    def mandatory_ids(self) -> frozenset[int]:
        return frozenset(poi.poi_id for poi in self.pois.values() if poi.mandatory)

    def get_travel_time(
        self,
        from_poi_id: int,
        to_poi_id: int,
    ) -> float:
        try:
            value = float(
                self.walking_matrix.loc[
                    from_poi_id,
                    str(to_poi_id),
                ]
            )
        except KeyError as error:
            raise KeyError(
                f"Walking-matrix value is missing for {from_poi_id} -> {to_poi_id}."
            ) from error

        if from_poi_id == to_poi_id:
            if not math.isclose(
                value,
                0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("Walking-matrix diagonal must be zero.")

            return 0.0

        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"Invalid walking time for {from_poi_id} -> {to_poi_id}: {value}"
            )

        return value


@dataclass(frozen=True)
class TravelAdjustments:
    """
    Used later for disruption scenarios.

    Examples:
        Network-wide delay:
            global_multiplier = 1.5

        Specific congestion links:
            link_multipliers[(1, 2)] = 1.3
    """

    global_multiplier: float = 1.0

    link_multipliers: Mapping[
        tuple[int, int],
        float,
    ] = field(default_factory=dict)

    def apply(
        self,
        from_poi_id: int,
        to_poi_id: int,
        base_minutes: float,
    ) -> float:
        if self.global_multiplier <= 0:
            raise ValueError("Global travel multiplier must be positive.")

        link_multiplier = float(
            self.link_multipliers.get(
                (from_poi_id, to_poi_id),
                1.0,
            )
        )

        if link_multiplier <= 0:
            raise ValueError("Link travel multiplier must be positive.")

        return base_minutes * self.global_multiplier * link_multiplier


@dataclass(frozen=True)
class ScheduleEntry:
    sequence_index: int

    poi_id: int
    poi_name: str
    previous_poi_id: int | None

    travel_min: float

    arrival_min: float
    arrival_clock: str

    waiting_min: float

    visit_start_min: float
    visit_start_clock: str

    visit_end_min: float
    visit_end_clock: str

    visit_duration_min: float

    admission_cost_rm: float
    cumulative_cost_rm: float


@dataclass(frozen=True)
class ScheduleResult:
    feasible: bool
    failure_reason: str | None

    start_poi_id: int
    requested_sequence: tuple[int, ...]
    completed_sequence: tuple[int, ...]

    entries: tuple[ScheduleEntry, ...]

    total_travel_min: float
    total_waiting_min: float
    total_visit_min: float
    time_used_min: float

    finish_min: float
    finish_clock: str

    total_cost_rm: float

    required_mandatory_ids: tuple[int, ...]
    unavailable_mandatory_ids: tuple[int, ...]
    missing_mandatory_ids: tuple[int, ...]

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "entries": [asdict(entry) for entry in self.entries],
        }


# ================================================================
# Data loading
# ================================================================


def load_planning_data(
    poi_file: str | Path,
    matrix_file: str | Path,
) -> PlanningData:
    poi_path = Path(poi_file)
    matrix_path = Path(matrix_file)

    if not poi_path.exists():
        raise FileNotFoundError(f"POI file not found: {poi_path.resolve()}")

    if not matrix_path.exists():
        raise FileNotFoundError(f"Walking matrix not found: {matrix_path.resolve()}")

    poi_df = pd.read_csv(poi_path)

    poi_df["poi_id"] = pd.to_numeric(
        poi_df["poi_id"],
        errors="raise",
    ).astype(int)

    poi_df = poi_df.sort_values("poi_id").reset_index(drop=True)

    required_columns = {
        "poi_id",
        "name",
        "category",
        "latitude",
        "longitude",
        "zone",
        "mandatory",
        "off_day",
        "is_24_hours",
        "opening_time_24h",
        "closing_time_24h",
        "closes_next_day",
        "visit_duration_min",
        "admission_cost_rm",
        "utility_score",
        "indoor_ratio",
        "shelter_ratio",
    }

    missing_columns = required_columns - set(poi_df.columns)

    if missing_columns:
        raise ValueError(f"POI file is missing columns: {sorted(missing_columns)}")

    if len(poi_df) != EXPECTED_POI_COUNT:
        raise ValueError(f"Expected {EXPECTED_POI_COUNT} POIs, found {len(poi_df)}.")

    expected_ids = list(range(1, EXPECTED_POI_COUNT + 1))

    if poi_df["poi_id"].tolist() != expected_ids:
        raise ValueError("POI IDs must be ordered exactly from 1 to 50.")

    pois: dict[int, POI] = {}

    for row in poi_df.itertuples(index=False):
        poi = POI(
            poi_id=int(row.poi_id),
            name=str(row.name).strip(),
            category=str(row.category).strip(),
            latitude=float(row.latitude),
            longitude=float(row.longitude),
            zone=str(row.zone).strip().upper(),
            mandatory=parse_yes_no(
                row.mandatory,
                "mandatory",
            ),
            off_days=parse_off_days(row.off_day),
            is_24_hours=parse_yes_no(
                row.is_24_hours,
                "is_24_hours",
            ),
            opening_minute=parse_clock(row.opening_time_24h),
            closing_minute=parse_clock(row.closing_time_24h),
            closes_next_day=parse_yes_no(
                row.closes_next_day,
                "closes_next_day",
            ),
            visit_duration_min=float(row.visit_duration_min),
            admission_cost_rm=float(row.admission_cost_rm),
            utility_score=float(row.utility_score),
            indoor_ratio=float(row.indoor_ratio),
            shelter_ratio=float(row.shelter_ratio),
        )

        pois[poi.poi_id] = poi

    matrix_df = pd.read_csv(matrix_path)

    expected_matrix_columns = ["poi_id"] + [str(poi_id) for poi_id in expected_ids]

    if list(matrix_df.columns) != expected_matrix_columns:
        raise ValueError("Walking-matrix columns do not align with POI IDs 1–50.")

    matrix_df["poi_id"] = pd.to_numeric(
        matrix_df["poi_id"],
        errors="raise",
    ).astype(int)

    if matrix_df["poi_id"].tolist() != expected_ids:
        raise ValueError("Walking-matrix row IDs do not align with POI IDs 1–50.")

    matrix_df = matrix_df.set_index("poi_id")

    matrix_values = matrix_df.to_numpy(dtype=float)

    if matrix_values.shape != (
        EXPECTED_POI_COUNT,
        EXPECTED_POI_COUNT,
    ):
        raise ValueError("Walking matrix must have shape 50 × 50.")

    if not math.isclose(
        float(abs(matrix_values - matrix_values.T).max()),
        0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("Walking matrix is not symmetric.")

    for index in range(EXPECTED_POI_COUNT):
        if not math.isclose(
            float(matrix_values[index, index]),
            0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Walking-matrix diagonal must be zero.")

    return PlanningData(
        pois=pois,
        walking_matrix=matrix_df,
    )


# ================================================================
# Schedule decoder
# ================================================================


def decode_route(
    data: PlanningData,
    visit_sequence: Iterable[int],
    *,
    start_poi_id: int = 1,
    start_time_min: float = TRIP_START_MIN,
    trip_end_min: float = TRIP_END_MIN,
    day_code: str = EXPERIMENT_DAY,
    cost_budget_rm: float = COST_BUDGET_RM,
    initial_cost_rm: float = 0.0,
    start_poi_is_visit: bool = True,
    required_mandatory_ids: Iterable[int] | None = None,
    unavailable_poi_ids: Iterable[int] = (),
    allow_unavailable_mandatory: bool = True,
    travel_adjustments: TravelAdjustments | None = None,
) -> ScheduleResult:
    """
    Decode an ordered POI sequence into a schedule.

    Initial itinerary example:
        start_poi_id=1
        visit_sequence=[1, 4, 13, ...]
        start_poi_is_visit=True

    Replanning example:
        start_poi_id=current_location_id
        visit_sequence=[remaining POIs only]
        start_poi_is_visit=False
        start_time_min=current simulated time
        initial_cost_rm=cost already spent

    During replanning, pass only the remaining mandatory
    POIs through required_mandatory_ids.
    """

    sequence = tuple(int(poi_id) for poi_id in visit_sequence)

    unavailable_ids = {int(poi_id) for poi_id in unavailable_poi_ids}

    if required_mandatory_ids is None:
        required_ids = set(data.mandatory_ids)
    else:
        required_ids = {int(poi_id) for poi_id in required_mandatory_ids}

    unavailable_mandatory = required_ids & unavailable_ids

    entries: list[ScheduleEntry] = []

    current_poi_id = start_poi_id
    current_time = float(start_time_min)
    current_cost = float(initial_cost_rm)

    adjustments = travel_adjustments or TravelAdjustments()

    def build_result(
        feasible: bool,
        failure_reason: str | None,
    ) -> ScheduleResult:
        completed_ids = tuple(entry.poi_id for entry in entries)

        missing_ids = required_ids - set(completed_ids) - unavailable_mandatory

        return ScheduleResult(
            feasible=feasible,
            failure_reason=failure_reason,
            start_poi_id=start_poi_id,
            requested_sequence=sequence,
            completed_sequence=completed_ids,
            entries=tuple(entries),
            total_travel_min=sum(entry.travel_min for entry in entries),
            total_waiting_min=sum(entry.waiting_min for entry in entries),
            total_visit_min=sum(entry.visit_duration_min for entry in entries),
            time_used_min=(current_time - start_time_min),
            finish_min=current_time,
            finish_clock=format_clock(current_time),
            total_cost_rm=current_cost,
            required_mandatory_ids=tuple(sorted(required_ids)),
            unavailable_mandatory_ids=tuple(sorted(unavailable_mandatory)),
            missing_mandatory_ids=tuple(sorted(missing_ids)),
        )

    if start_poi_id not in data.pois:
        raise KeyError(f"Unknown start POI: {start_poi_id}")

    unknown_ids = [poi_id for poi_id in sequence if poi_id not in data.pois]

    if unknown_ids:
        return build_result(
            False,
            f"Unknown POI IDs: {sorted(set(unknown_ids))}",
        )

    if len(sequence) != len(set(sequence)):
        return build_result(
            False,
            "The route contains duplicate POI visits.",
        )

    if unavailable_mandatory and not allow_unavailable_mandatory:
        return build_result(
            False,
            f"Mandatory POIs are unavailable: {sorted(unavailable_mandatory)}",
        )

    if start_poi_is_visit:
        if not sequence:
            return build_result(
                False,
                "The initial route must contain the starting POI.",
            )

        if sequence[0] != start_poi_id:
            return build_result(
                False,
                "The first route ID must equal "
                "start_poi_id when "
                "start_poi_is_visit=True.",
            )

    elif start_poi_id in sequence and sequence[0] != start_poi_id:
        return build_result(
            False,
            "The current-location POI may appear only "
            "as the first visit during bounded replanning.",
        )

    for sequence_index, poi_id in enumerate(sequence):
        poi = data.pois[poi_id]

        if poi_id in unavailable_ids:
            return build_result(
                False,
                f"POI {poi_id} ({poi.name}) is unavailable.",
            )

        opening_window = poi.get_opening_window(day_code)

        if opening_window is None:
            return build_result(
                False,
                f"POI {poi_id} ({poi.name}) is closed on {day_code}.",
            )

        if sequence_index == 0 and poi_id == start_poi_id:
            # Initial planning:
            #     POI 1 is the first visit.
            #
            # Replanning:
            #     The traveller may already be located at an
            #     unvisited POI after completing a committed
            #     walking leg or while waiting for it to open.
            previous_poi_id = None if start_poi_is_visit else start_poi_id

            travel_min = 0.0

        else:
            previous_poi_id = current_poi_id

            base_travel_min = data.get_travel_time(
                previous_poi_id,
                poi_id,
            )

            travel_min = adjustments.apply(
                previous_poi_id,
                poi_id,
                base_travel_min,
            )

        arrival_min = current_time + travel_min

        opening_min, closing_min = opening_window

        waiting_min = max(
            0.0,
            opening_min - arrival_min,
        )

        visit_start_min = arrival_min + waiting_min

        visit_end_min = visit_start_min + poi.visit_duration_min

        proposed_cost = current_cost + poi.admission_cost_rm

        if visit_start_min >= closing_min:
            return build_result(
                False,
                f"POI {poi_id} ({poi.name}) cannot be started before closing.",
            )

        if visit_end_min > closing_min + 1e-9:
            return build_result(
                False,
                f"POI {poi_id} ({poi.name}) would finish after closing.",
            )

        if visit_end_min > trip_end_min + 1e-9:
            return build_result(
                False,
                f"POI {poi_id} ({poi.name}) would exceed the 17:00 trip limit.",
            )

        if proposed_cost > cost_budget_rm + 1e-9:
            return build_result(
                False,
                f"POI {poi_id} ({poi.name}) "
                "would exceed the cost budget: "
                f"RM{proposed_cost:.2f} > "
                f"RM{cost_budget_rm:.2f}.",
            )

        entry = ScheduleEntry(
            sequence_index=sequence_index,
            poi_id=poi_id,
            poi_name=poi.name,
            previous_poi_id=previous_poi_id,
            travel_min=travel_min,
            arrival_min=arrival_min,
            arrival_clock=format_clock(arrival_min),
            waiting_min=waiting_min,
            visit_start_min=visit_start_min,
            visit_start_clock=format_clock(visit_start_min),
            visit_end_min=visit_end_min,
            visit_end_clock=format_clock(visit_end_min),
            visit_duration_min=(poi.visit_duration_min),
            admission_cost_rm=(poi.admission_cost_rm),
            cumulative_cost_rm=(proposed_cost),
        )

        entries.append(entry)

        current_poi_id = poi_id
        current_time = visit_end_min
        current_cost = proposed_cost

    completed_ids = {entry.poi_id for entry in entries}

    missing_mandatory = required_ids - completed_ids - unavailable_mandatory

    if missing_mandatory:
        return build_result(
            False,
            "The route does not contain all available "
            "mandatory POIs: "
            f"{sorted(missing_mandatory)}",
        )

    return build_result(
        True,
        None,
    )


# ================================================================
# Console display
# ================================================================


def print_schedule(
    result: ScheduleResult,
) -> None:
    print("=" * 90)
    print("SCHEDULE RESULT")
    print("=" * 90)

    print(f"Feasible: {result.feasible}")

    print(f"Failure reason: {result.failure_reason}")

    print(f"Requested route: {list(result.requested_sequence)}")

    print(f"Completed route: {list(result.completed_sequence)}")

    print(f"Finish time: {result.finish_clock}")

    print(f"Travel time: {result.total_travel_min:.2f} min")

    print(f"Waiting time: {result.total_waiting_min:.2f} min")

    print(f"Visit time: {result.total_visit_min:.2f} min")

    print(f"Total time used: {result.time_used_min:.2f} min")

    print(f"Total cost: RM{result.total_cost_rm:.2f}")

    print(f"Missing mandatory POIs: {list(result.missing_mandatory_ids)}")

    if not result.entries:
        return

    schedule_df = pd.DataFrame([asdict(entry) for entry in result.entries])

    display_columns = [
        "sequence_index",
        "poi_id",
        "poi_name",
        "previous_poi_id",
        "travel_min",
        "arrival_clock",
        "waiting_min",
        "visit_start_clock",
        "visit_end_clock",
        "admission_cost_rm",
        "cumulative_cost_rm",
    ]

    print()
    print(schedule_df[display_columns].to_string(index=False))


# ================================================================
# Integration test
# ================================================================


def main() -> None:
    data = load_planning_data(
        poi_file=("poi_data_calculated_final.csv"),
        matrix_file=("walking_time_matrix_final.csv"),
    )

    mandatory_ids = sorted(data.mandatory_ids)

    print(
        "Mandatory POIs:",
        mandatory_ids,
    )

    # Origin must remain first.
    test_route = [1] + [poi_id for poi_id in mandatory_ids if poi_id != 1]

    result = decode_route(
        data=data,
        visit_sequence=test_route,
        start_poi_id=1,
        start_time_min=TRIP_START_MIN,
        trip_end_min=TRIP_END_MIN,
        day_code="WED",
        cost_budget_rm=160.0,
        start_poi_is_visit=True,
        required_mandatory_ids=(mandatory_ids),
    )

    print_schedule(result)

    if not result.feasible:
        raise SystemExit("Shared planning-core test failed.")


if __name__ == "__main__":
    main()
