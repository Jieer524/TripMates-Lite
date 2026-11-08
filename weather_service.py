"""Weather-profile retrieval with a reproducible local fallback path.

The planner consumes a small, stable WeatherProfile rather than an
Open-Meteo response. This keeps external API availability outside the
planning engine and records whether a live, cached, or predefined
condition was used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Literal
import math
from urllib.parse import urlencode
from urllib.request import urlopen


WeatherSource = Literal["live", "cache", "cached", "predefined", "unavailable"]


@dataclass(frozen=True)
class WeatherProfile:
    source_type: WeatherSource
    category: str
    thermal_multiplier: float
    latitude: float
    longitude: float
    applicable_at: str
    retrieved_at: str
    weather_code: int | None = None
    temperature_c: float | None = None
    relative_humidity_pct: float | None = None
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if self.thermal_multiplier < 1.0:
            raise ValueError(
                "Thermal multiplier cannot be below 1.0."
            )

        if not -90.0 <= self.latitude <= 90.0:
            raise ValueError("Latitude must be between -90 and 90.")

        if not -180.0 <= self.longitude <= 180.0:
            raise ValueError("Longitude must be between -180 and 180.")
        if not math.isfinite(self.thermal_multiplier):
            raise ValueError("Thermal multiplier must be finite.")
        if self.temperature_c is not None and not math.isfinite(self.temperature_c):
            raise ValueError("Temperature must be finite when supplied.")
        if self.relative_humidity_pct is not None and not 0.0 <= self.relative_humidity_pct <= 100.0:
            raise ValueError("Relative humidity must be between 0 and 100.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WeatherProfile":
        return cls(
            source_type=str(payload["source_type"]),
            category=str(payload["category"]),
            thermal_multiplier=float(payload["thermal_multiplier"]),
            latitude=float(payload["latitude"]),
            longitude=float(payload["longitude"]),
            applicable_at=str(payload["applicable_at"]),
            retrieved_at=str(payload["retrieved_at"]),
            weather_code=(
                int(payload["weather_code"])
                if payload.get("weather_code") is not None
                else None
            ),
            temperature_c=(
                float(payload["temperature_c"])
                if payload.get("temperature_c") is not None
                else None
            ),
            relative_humidity_pct=(
                float(payload["relative_humidity_pct"])
                if payload.get("relative_humidity_pct") is not None
                else None
            ),
            fallback_reason=(
                str(payload["fallback_reason"])
                if payload.get("fallback_reason") is not None
                else None
            ),
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def classify_weather(
    weather_code: int,
    temperature_c: float | None,
    relative_humidity_pct: float | None = None,
) -> tuple[str, float]:
    """Map observations to the planning-proxy multiplier scale."""

    rainy_codes = {
        51,
        53,
        55,
        56,
        57,
        61,
        63,
        65,
        66,
        67,
        80,
        81,
        82,
        95,
        96,
        99,
    }

    if weather_code in rainy_codes:
        return "rain", 1.8

    if temperature_c is not None and temperature_c >= 35.0:
        return "hot", 2.0

    if (
        temperature_c is not None
        and relative_humidity_pct is not None
        and temperature_c >= 32.0
        and relative_humidity_pct >= 70.0
    ):
        return "hot_humid", 2.0

    return "normal", 1.0


def predefined_weather_profile(
    *,
    category: str = "normal",
    thermal_multiplier: float = 1.0,
    latitude: float = 3.1390,
    longitude: float = 101.6869,
    applicable_at: str | None = None,
) -> WeatherProfile:
    timestamp = utc_now_iso()

    return WeatherProfile(
        source_type="predefined",
        category=category,
        thermal_multiplier=thermal_multiplier,
        latitude=latitude,
        longitude=longitude,
        applicable_at=applicable_at or timestamp,
        retrieved_at=timestamp,
    )


JsonFetcher = Callable[[str, float], dict[str, Any]]


def fetch_open_meteo_json(url: str, timeout_seconds: float) -> dict[str, Any]:
    with urlopen(url, timeout=timeout_seconds) as response:  # nosec B310
        payload = json.loads(response.read().decode("utf-8"))

    if not isinstance(payload, dict):
        raise ValueError("Open-Meteo returned an invalid JSON object.")

    return payload


class WeatherServiceAdapter:
    """Retrieve and cache the compact weather input used by the planner."""

    endpoint = "https://api.open-meteo.com/v1/forecast"

    def __init__(
        self,
        *,
        cache_file: str | Path | None = None,
        timeout_seconds: float = 5.0,
        fetcher: JsonFetcher = fetch_open_meteo_json,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")

        self.cache_file = (
            Path(cache_file) if cache_file is not None else None
        )
        self.timeout_seconds = float(timeout_seconds)
        self.fetcher = fetcher

    def get_profile(
        self,
        *,
        latitude: float,
        longitude: float,
        applicable_at: str,
        fallback_profile: WeatherProfile | None = None,
    ) -> WeatherProfile:
        """Return live weather, then cached weather, then the fallback."""

        try:
            profile = self.fetch_live_profile(
                latitude=latitude,
                longitude=longitude,
                applicable_at=applicable_at,
            )
            self.write_cache(profile)
            return profile
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            reason = f"live weather unavailable: {error}"

        cached_profile = self.read_cache()
        if cached_profile is not None:
            return replace(
                cached_profile,
                # ``cache`` is retained as the serialized legacy spelling;
                # documentation calls this state "cached".
                source_type="cache",
                applicable_at=applicable_at,
                fallback_reason=reason,
            )

        if fallback_profile is not None:
            return replace(
                fallback_profile,
                source_type="predefined",
                latitude=latitude,
                longitude=longitude,
                applicable_at=applicable_at,
                fallback_reason=reason,
            )

        # Explicit unavailable state: neutral benchmark multiplier, never
        # zero exposure. Callers can choose to disable thermal optimisation
        # while retaining the auditable source state.
        return WeatherProfile(
            source_type="unavailable",
            category="unavailable",
            thermal_multiplier=1.0,
            latitude=latitude,
            longitude=longitude,
            applicable_at=applicable_at,
            retrieved_at=utc_now_iso(),
            fallback_reason=reason,
        )

    def fetch_live_profile(
        self,
        *,
        latitude: float,
        longitude: float,
        applicable_at: str,
    ) -> WeatherProfile:
        query = urlencode(
            {
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,weather_code",
                "timezone": "Asia/Kuala_Lumpur",
            }
        )
        payload = self.fetcher(
            f"{self.endpoint}?{query}",
            self.timeout_seconds,
        )
        current = payload["current"]

        if not isinstance(current, dict):
            raise ValueError("Open-Meteo response is missing current weather.")

        weather_code = int(current["weather_code"])
        temperature_raw = current.get("temperature_2m")
        temperature_c = None if temperature_raw is None else float(temperature_raw)
        humidity_raw = current.get("relative_humidity_2m")
        relative_humidity_pct = None if humidity_raw is None else float(humidity_raw)
        if temperature_c is None and relative_humidity_pct is None:
            raise ValueError("Weather response has no temperature or humidity.")
        category, multiplier = classify_weather(
            weather_code,
            temperature_c,
            relative_humidity_pct,
        )

        return WeatherProfile(
            source_type="live",
            category=category,
            thermal_multiplier=multiplier,
            latitude=latitude,
            longitude=longitude,
            applicable_at=applicable_at,
            retrieved_at=utc_now_iso(),
            weather_code=weather_code,
            temperature_c=temperature_c,
            relative_humidity_pct=relative_humidity_pct,
        )

    def read_cache(self) -> WeatherProfile | None:
        if self.cache_file is None or not self.cache_file.exists():
            return None

        try:
            payload = json.loads(
                self.cache_file.read_text(encoding="utf-8")
            )
            if not isinstance(payload, dict):
                return None
            return WeatherProfile.from_dict(payload)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None

    def write_cache(self, profile: WeatherProfile) -> None:
        if self.cache_file is None:
            return

        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_text(
            json.dumps(profile.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
