"""
fetch_weather.py
----------------
Pulls real historical daily weather from the Open-Meteo archive API
(no API key required) for each region used in the project, then writes
a single weather.csv to DATA_OUTPUT_DIR.
"""

import logging
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_OUTPUT_DIR = Path(os.environ["DATA_OUTPUT_DIR"])

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
REQUEST_TIMEOUT_SECONDS = 30

# Representative coordinates per region (matches regions in generate_fake_data.py)
REGION_COORDINATES: dict[str, tuple[float, float]] = {
    "North": (53.0, -1.5),
    "South": (51.5, -0.5),
    "East":  (52.5,  1.0),
    "West":  (52.0, -3.0),
}


# ── Core fetch function ───────────────────────────────────────────────────────

def fetch_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch daily weather from the Open-Meteo archive API for a given
    coordinate range and return a cleaned DataFrame with columns:
        date, temperature_max, precipitation_sum, windspeed_max

    Args:
        latitude:   Decimal latitude of the location.
        longitude:  Decimal longitude of the location.
        start_date: ISO 8601 start date string (e.g. "2024-01-01").
        end_date:   ISO 8601 end date string   (e.g. "2024-06-29").

    Raises:
        RuntimeError: If the HTTP request fails, returns a non-200 status,
                      or the response contains no usable data.
    """
    params = {
        "latitude":        latitude,
        "longitude":       longitude,
        "start_date":      start_date,
        "end_date":        end_date,
        "daily":           "temperature_2m_max,precipitation_sum,windspeed_10m_max",
        "timezone":        "UTC",
    }

    try:
        response = requests.get(
            OPEN_METEO_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Open-Meteo request timed out after {REQUEST_TIMEOUT_SECONDS}s "
            f"for ({latitude}, {longitude})."
        )
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"Open-Meteo request failed for ({latitude}, {longitude}): {exc}"
        ) from exc

    if response.status_code != 200:
        raise RuntimeError(
            f"Open-Meteo returned HTTP {response.status_code} for "
            f"({latitude}, {longitude}): {response.text[:200]}"
        )

    payload = response.json()
    daily = payload.get("daily", {})

    if not daily or "time" not in daily:
        raise RuntimeError(
            f"Open-Meteo response contains no daily data for "
            f"({latitude}, {longitude}) between {start_date} and {end_date}."
        )

    weather_df = pd.DataFrame(
        {
            "date":              daily["time"],
            "temperature_max":   daily.get("temperature_2m_max"),
            "precipitation_sum": daily.get("precipitation_sum"),
            "windspeed_max":     daily.get("windspeed_10m_max"),
        }
    )

    # Ensure ISO 8601 date strings
    weather_df["date"] = pd.to_datetime(weather_df["date"]).dt.date.apply(
        lambda d: d.isoformat()
    )

    return weather_df


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    start_date = "2024-01-01"
    end_date = "2024-06-29"  # 180 days, matching generate_fake_data

    DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for region, (lat, lon) in REGION_COORDINATES.items():
        logger.info(
            "Fetching weather for region '%s' (%.4f, %.4f) …", region, lat, lon
        )
        region_df = fetch_weather(lat, lon, start_date, end_date)
        region_df["region"] = region
        frames.append(region_df)
        logger.info("  → %d rows fetched", len(region_df))

    weather_df = pd.concat(frames, ignore_index=True)
    output_path = DATA_OUTPUT_DIR / "weather.csv"
    weather_df.to_csv(output_path, index=False)
    logger.info("Saved weather data → %s (%d total rows)", output_path, len(weather_df))


if __name__ == "__main__":
    main()
