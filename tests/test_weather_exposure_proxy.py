from pathlib import Path
import json

import pytest

from exposure_core import (
    calculate_exposed_fraction,
    calculate_weighted_interval,
)
from weather_service import (
    WeatherServiceAdapter,
    classify_weather,
    predefined_weather_profile,
)


def test_exposed_fraction_indoor_sheltered_outdoor():
    assert calculate_exposed_fraction(1.0, 0.0) == 0.0
    assert calculate_exposed_fraction(0.0, 1.0) == 0.0
    assert calculate_exposed_fraction(0.0, 0.0) == 1.0
    assert calculate_exposed_fraction(0.2, 0.5) == pytest.approx(0.4)


def test_normal_heavy_rain_and_heatwave_profiles():
    assert classify_weather(1, 25.0) == ("normal", 1.0)
    assert classify_weather(95, 25.0) == ("rain", 1.8)
    assert classify_weather(1, 36.0) == ("hot", 2.0)
    assert classify_weather(1, 32.0, 75.0) == ("hot_humid", 2.0)


def test_missing_weather_uses_predefined_or_explicit_unavailable(tmp_path):
    def failing(_url, _timeout):
        raise OSError("network unavailable")

    adapter = WeatherServiceAdapter(cache_file=tmp_path / "none.json", fetcher=failing)
    fallback = predefined_weather_profile(category="rain", thermal_multiplier=1.8)
    profile = adapter.get_profile(latitude=3.1, longitude=101.6, applicable_at="2026-11-04T09:00:00+08:00", fallback_profile=fallback)
    assert profile.source_type == "predefined"
    assert profile.thermal_multiplier == 1.8
    assert "network unavailable" in profile.fallback_reason

    unavailable = adapter.get_profile(latitude=3.1, longitude=101.6, applicable_at="2026-11-04T09:00:00+08:00")
    assert unavailable.source_type == "unavailable"
    assert unavailable.thermal_multiplier == 1.0
    assert unavailable.fallback_reason


def test_cached_weather_is_explicit(tmp_path):
    cache = tmp_path / "weather.json"
    cached = predefined_weather_profile(category="hot", thermal_multiplier=2.0)
    cache.write_text(json.dumps(cached.to_dict()), encoding="utf-8")
    adapter = WeatherServiceAdapter(cache_file=cache, fetcher=lambda *_: (_ for _ in ()).throw(OSError("offline")))
    profile = adapter.get_profile(latitude=3.1, longitude=101.6, applicable_at="2026-11-04T09:00:00+08:00", fallback_profile=predefined_weather_profile())
    assert profile.source_type in {"cache", "cached"}
    assert profile.thermal_multiplier == 2.0


def test_invalid_weather_data_falls_back_deterministically(tmp_path):
    def invalid(_url, _timeout):
        return {"current": {"weather_code": 95, "temperature_2m": "not-a-number"}}

    adapter = WeatherServiceAdapter(cache_file=tmp_path / "none.json", fetcher=invalid)
    profile = adapter.get_profile(latitude=3.1, longitude=101.6, applicable_at="2026-11-04T09:00:00+08:00", fallback_profile=predefined_weather_profile(category="normal", thermal_multiplier=1.0))
    assert profile.source_type == "predefined"
    assert "could not convert" in profile.fallback_reason


def test_pre_and_post_trigger_weighting_and_time_of_day():
    assert calculate_weighted_interval(9 * 60, 10 * 60, 9 * 60, 17 * 60, 9.5 * 60, 1.8) == pytest.approx(30 + 1.8 * 30)
    assert calculate_weighted_interval(13 * 60, 14 * 60, 9 * 60, 17 * 60, 12 * 60, 2.0) == pytest.approx(120)


def test_waiting_is_explicitly_zero_by_default():
    # The primary proxy has no waiting-location label; callers must opt in.
    from exposure_core import ExposureMetrics
    metrics = ExposureMetrics(0, 0, 0, 0, 0, 0, 0, 0, 1.0)
    assert metrics.waiting_exposure_min == 0.0
