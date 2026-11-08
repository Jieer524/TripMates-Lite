"""Focused tests for weather fallback and local itinerary history."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from itinerary_history import (
    ItineraryHistoryRepository,
    ItineraryRunRecord,
)
from planning_core import decode_route, load_planning_data
from weather_service import (
    WeatherServiceAdapter,
    predefined_weather_profile,
)


ROOT = Path(__file__).resolve().parent


def main() -> None:
    with TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        cache_file = temporary_path / "weather_cache.json"

        def live_response(_url: str, _timeout: float) -> dict:
            return {
                "current": {
                    "weather_code": 63,
                    "temperature_2m": 28.5,
                }
            }

        adapter = WeatherServiceAdapter(
            cache_file=cache_file,
            fetcher=live_response,
        )
        profile = adapter.get_profile(
            latitude=3.1390,
            longitude=101.6869,
            applicable_at="2026-09-10T09:00:00+08:00",
            fallback_profile=predefined_weather_profile(),
        )

        if (
            profile.source_type != "live"
            or profile.category != "rain"
            or profile.thermal_multiplier != 1.8
            or not cache_file.exists()
        ):
            raise RuntimeError("Live weather profile was not normalised or cached.")

        def unavailable(_url: str, _timeout: float) -> dict:
            raise OSError("offline")

        predefined_adapter = WeatherServiceAdapter(
            cache_file=temporary_path / "missing_cache.json",
            fetcher=unavailable,
        )
        predefined_profile = predefined_adapter.get_profile(
            latitude=3.1390,
            longitude=101.6869,
            applicable_at="2026-09-10T10:00:00+08:00",
            fallback_profile=predefined_weather_profile(
                category="hot",
                thermal_multiplier=2.0,
            ),
        )

        if (
            predefined_profile.source_type != "predefined"
            or predefined_profile.thermal_multiplier != 2.0
            or predefined_profile.fallback_reason is None
        ):
            raise RuntimeError("Predefined weather fallback was not applied.")

        cached_adapter = WeatherServiceAdapter(
            cache_file=cache_file,
            fetcher=unavailable,
        )
        cached_profile = cached_adapter.get_profile(
            latitude=3.1390,
            longitude=101.6869,
            applicable_at="2026-09-10T10:00:00+08:00",
            fallback_profile=predefined_weather_profile(),
        )

        if (
            cached_profile.source_type != "cache"
            or cached_profile.thermal_multiplier != 1.8
            or cached_profile.fallback_reason is None
        ):
            raise RuntimeError("Cached weather fallback was not applied.")

        data = load_planning_data(
            ROOT / "poi_data_calculated_final.csv",
            ROOT / "walking_time_matrix_final.csv",
        )
        schedule = decode_route(
            data,
            (1, 2, 12, 4, 33, 34, 36, 13),
            required_mandatory_ids=data.mandatory_ids,
        )

        if not schedule.feasible:
            raise RuntimeError("The fixture itinerary must be feasible.")

        repository = ItineraryHistoryRepository(
            temporary_path / "history.sqlite3"
        )
        record = repository.save(
            record=ItineraryRunRecord(
                planner_id="tripmates_lite",
                seed=1001,
                scenario_id="S1_nominal",
                catalogue_id="poi_data_calculated_final",
                matrix_id="walking_time_matrix_final",
                utility_rule_version="equal_weight_v1",
                configuration_id="benchmark_v1",
                trigger_state={"trigger_clock": "11:00"},
                itinerary_name="Rainy KL Day",
                fallback_status="thermal_admissible",
            ),
            schedule=schedule,
            metrics={
                "achieved_utility": 66.96204,
                "total_travel_min": schedule.total_travel_min,
                "wes": 384.719112,
            },
        )

        loaded = repository.load(record.run_id)
        if (
            loaded is None
            or loaded.record.itinerary_name != "Rainy KL Day"
            or len(loaded.stops) != len(schedule.entries)
            or loaded.schedule != schedule
            or loaded.metrics["wes"] != 384.719112
            or len(repository.list_runs()) != 1
        ):
            raise RuntimeError("Saved itinerary history was not restored intact.")

        if not repository.delete(record.run_id) or repository.load(record.run_id):
            raise RuntimeError("Itinerary deletion did not remove the saved record.")

    print("PASS: weather fallback and itinerary history services work.")


if __name__ == "__main__":
    main()
