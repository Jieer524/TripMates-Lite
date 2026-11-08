"""SQLite-backed local itinerary history and traceability records."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping
from uuid import uuid4

from planning_core import ScheduleEntry, ScheduleResult


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ItineraryRunRecord:
    planner_id: str
    seed: int | None
    scenario_id: str | None
    catalogue_id: str
    matrix_id: str
    utility_rule_version: str
    configuration_id: str
    trigger_state: Mapping[str, Any]
    request_parameters: Mapping[str, Any] = field(default_factory=dict)
    itinerary_name: str = "Untitled itinerary"
    fallback_status: str | None = None
    run_id: str = ""
    created_at: str = ""


@dataclass(frozen=True)
class StoredItinerary:
    record: ItineraryRunRecord
    stops: tuple[dict[str, Any], ...]
    metrics: dict[str, float]
    schedule: ScheduleResult


class ItineraryHistoryRepository:
    """Persist generated itineraries locally with parameterised SQLite calls."""

    def __init__(self, database_file: str | Path) -> None:
        self.database_file = Path(database_file)
        self.database_file.parent.mkdir(parents=True, exist_ok=True)
        self.initialise()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_file)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def session(self):
        connection = self.connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialise(self) -> None:
        with self.session() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS itinerary_run (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    itinerary_name TEXT NOT NULL DEFAULT 'Untitled itinerary',
                    planner_id TEXT NOT NULL,
                    seed INTEGER,
                    scenario_id TEXT,
                    catalogue_id TEXT NOT NULL,
                    matrix_id TEXT NOT NULL,
                    utility_rule_version TEXT NOT NULL,
                    configuration_id TEXT NOT NULL,
                    trigger_state_json TEXT NOT NULL,
                    request_json TEXT NOT NULL DEFAULT '{}',
                    schedule_json TEXT,
                    fallback_status TEXT
                );

                CREATE TABLE IF NOT EXISTS itinerary_stop (
                    run_id TEXT NOT NULL,
                    sequence_index INTEGER NOT NULL,
                    poi_id INTEGER NOT NULL,
                    poi_name TEXT NOT NULL,
                    previous_poi_id INTEGER,
                    travel_min REAL NOT NULL,
                    arrival_min REAL NOT NULL,
                    arrival_clock TEXT NOT NULL,
                    waiting_min REAL NOT NULL,
                    visit_start_min REAL NOT NULL,
                    visit_start_clock TEXT NOT NULL,
                    visit_end_min REAL NOT NULL,
                    visit_end_clock TEXT NOT NULL,
                    visit_duration_min REAL NOT NULL,
                    admission_cost_rm REAL NOT NULL,
                    cumulative_cost_rm REAL NOT NULL,
                    PRIMARY KEY (run_id, sequence_index),
                    FOREIGN KEY (run_id) REFERENCES itinerary_run(run_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS itinerary_metric (
                    run_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    PRIMARY KEY (run_id, metric_name),
                    FOREIGN KEY (run_id) REFERENCES itinerary_run(run_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_itinerary_run_created_at
                    ON itinerary_run(created_at DESC);
                """
            )

            existing_columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(itinerary_run)"
                ).fetchall()
            }
            if "request_json" not in existing_columns:
                connection.execute(
                    "ALTER TABLE itinerary_run "
                    "ADD COLUMN request_json TEXT NOT NULL DEFAULT '{}'"
                )
            if "schedule_json" not in existing_columns:
                connection.execute(
                    "ALTER TABLE itinerary_run ADD COLUMN schedule_json TEXT"
                )
            if "itinerary_name" not in existing_columns:
                connection.execute(
                    "ALTER TABLE itinerary_run ADD COLUMN itinerary_name "
                    "TEXT NOT NULL DEFAULT 'Untitled itinerary'"
                )

    def save(
        self,
        *,
        record: ItineraryRunRecord,
        schedule: ScheduleResult,
        metrics: Mapping[str, float],
    ) -> ItineraryRunRecord:
        if not schedule.feasible:
            raise ValueError("Only feasible itineraries can be saved.")

        run_id = record.run_id or str(uuid4())
        created_at = record.created_at or utc_now_iso()
        persisted_record = ItineraryRunRecord(
            **{
                **asdict(record),
                "run_id": run_id,
                "created_at": created_at,
            }
        )

        with self.session() as connection:
            connection.execute(
                """
                INSERT INTO itinerary_run (
                    run_id, created_at, itinerary_name, planner_id, seed, scenario_id,
                    catalogue_id, matrix_id, utility_rule_version,
                    configuration_id, trigger_state_json, request_json,
                    schedule_json, fallback_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    persisted_record.run_id,
                    persisted_record.created_at,
                    persisted_record.itinerary_name,
                    persisted_record.planner_id,
                    persisted_record.seed,
                    persisted_record.scenario_id,
                    persisted_record.catalogue_id,
                    persisted_record.matrix_id,
                    persisted_record.utility_rule_version,
                    persisted_record.configuration_id,
                    json.dumps(
                        dict(persisted_record.trigger_state),
                        sort_keys=True,
                    ),
                    json.dumps(
                        dict(persisted_record.request_parameters),
                        sort_keys=True,
                    ),
                    json.dumps(schedule.to_dict(), sort_keys=True),
                    persisted_record.fallback_status,
                ),
            )

            connection.executemany(
                """
                INSERT INTO itinerary_stop (
                    run_id, sequence_index, poi_id, poi_name, previous_poi_id,
                    travel_min, arrival_min, arrival_clock, waiting_min,
                    visit_start_min, visit_start_clock, visit_end_min,
                    visit_end_clock, visit_duration_min, admission_cost_rm,
                    cumulative_cost_rm
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        entry.sequence_index,
                        entry.poi_id,
                        entry.poi_name,
                        entry.previous_poi_id,
                        entry.travel_min,
                        entry.arrival_min,
                        entry.arrival_clock,
                        entry.waiting_min,
                        entry.visit_start_min,
                        entry.visit_start_clock,
                        entry.visit_end_min,
                        entry.visit_end_clock,
                        entry.visit_duration_min,
                        entry.admission_cost_rm,
                        entry.cumulative_cost_rm,
                    )
                    for entry in schedule.entries
                ],
            )

            connection.executemany(
                """
                INSERT INTO itinerary_metric (
                    run_id, metric_name, metric_value
                ) VALUES (?, ?, ?)
                """,
                [
                    (run_id, str(name), float(value))
                    for name, value in metrics.items()
                ],
            )

        return persisted_record

    def list_runs(self) -> tuple[ItineraryRunRecord, ...]:
        with self.session() as connection:
            rows = connection.execute(
                "SELECT * FROM itinerary_run ORDER BY created_at DESC, run_id DESC"
            ).fetchall()

        return tuple(self.row_to_record(row) for row in rows)

    def load(self, run_id: str) -> StoredItinerary | None:
        with self.session() as connection:
            row = connection.execute(
                "SELECT * FROM itinerary_run WHERE run_id = ?",
                (run_id,),
            ).fetchone()

            if row is None:
                return None

            stop_rows = connection.execute(
                """
                SELECT * FROM itinerary_stop
                WHERE run_id = ?
                ORDER BY sequence_index ASC
                """,
                (run_id,),
            ).fetchall()

            metric_rows = connection.execute(
                """
                SELECT metric_name, metric_value FROM itinerary_metric
                WHERE run_id = ?
                ORDER BY metric_name ASC
                """,
                (run_id,),
            ).fetchall()

        stop_dicts = tuple(dict(stop_row) for stop_row in stop_rows)
        schedule = self.restore_schedule(row, stop_dicts)

        return StoredItinerary(
            record=self.row_to_record(row),
            stops=stop_dicts,
            metrics={
                str(metric_row["metric_name"]): float(
                    metric_row["metric_value"]
                )
                for metric_row in metric_rows
            },
            schedule=schedule,
        )

    @staticmethod
    def restore_schedule(
        run_row: sqlite3.Row,
        stops: tuple[dict[str, Any], ...],
    ) -> ScheduleResult:
        """Restore an exact saved schedule, with legacy-row compatibility."""

        schedule_json = run_row["schedule_json"]
        if schedule_json:
            payload = json.loads(str(schedule_json))
            payload["entries"] = tuple(
                ScheduleEntry(**entry) for entry in payload["entries"]
            )
            for key in (
                "requested_sequence",
                "completed_sequence",
                "required_mandatory_ids",
                "unavailable_mandatory_ids",
                "missing_mandatory_ids",
            ):
                payload[key] = tuple(payload[key])
            return ScheduleResult(**payload)

        entries = tuple(
            ScheduleEntry(
                **{
                    key: value
                    for key, value in stop.items()
                    if key != "run_id"
                }
            )
            for stop in stops
        )
        if not entries:
            raise ValueError("The stored itinerary contains no stops.")

        start_min = entries[0].arrival_min
        finish_min = entries[-1].visit_end_min
        completed = tuple(entry.poi_id for entry in entries)
        return ScheduleResult(
            feasible=True,
            failure_reason=None,
            start_poi_id=entries[0].poi_id,
            requested_sequence=completed,
            completed_sequence=completed,
            entries=entries,
            total_travel_min=sum(entry.travel_min for entry in entries),
            total_waiting_min=sum(entry.waiting_min for entry in entries),
            total_visit_min=sum(entry.visit_duration_min for entry in entries),
            time_used_min=finish_min - start_min,
            finish_min=finish_min,
            finish_clock=entries[-1].visit_end_clock,
            total_cost_rm=entries[-1].cumulative_cost_rm,
            required_mandatory_ids=(),
            unavailable_mandatory_ids=(),
            missing_mandatory_ids=(),
        )

    def delete(self, run_id: str) -> bool:
        with self.session() as connection:
            cursor = connection.execute(
                "DELETE FROM itinerary_run WHERE run_id = ?",
                (run_id,),
            )
            deleted = cursor.rowcount == 1

        return deleted

    @staticmethod
    def row_to_record(row: sqlite3.Row) -> ItineraryRunRecord:
        return ItineraryRunRecord(
            run_id=str(row["run_id"]),
            created_at=str(row["created_at"]),
            itinerary_name=str(row["itinerary_name"]),
            planner_id=str(row["planner_id"]),
            seed=(
                int(row["seed"])
                if row["seed"] is not None
                else None
            ),
            scenario_id=(
                str(row["scenario_id"])
                if row["scenario_id"] is not None
                else None
            ),
            catalogue_id=str(row["catalogue_id"]),
            matrix_id=str(row["matrix_id"]),
            utility_rule_version=str(row["utility_rule_version"]),
            configuration_id=str(row["configuration_id"]),
            trigger_state=json.loads(str(row["trigger_state_json"])),
            request_parameters=json.loads(str(row["request_json"] or "{}")),
            fallback_status=(
                str(row["fallback_status"])
                if row["fallback_status"] is not None
                else None
            ),
        )
