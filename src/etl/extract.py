"""
extract.py
----------
Reads the four CSV files produced by data_generation/ into pandas DataFrames
and validates that each file contains the expected columns before returning.
"""

import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_OUTPUT_DIR = Path(os.environ["DATA_OUTPUT_DIR"])

# Expected columns per CSV file
EXPECTED_COLUMNS: dict[str, list[str]] = {
    "assets.csv": [
        "asset_id", "asset_type", "region", "lat", "lon",
        "install_year", "customers_served", "criticality_tier",
    ],
    "sensor_readings.csv": [
        "asset_id", "date", "temperature", "vibration",
        "oil_quality", "partial_discharge", "failed",
    ],
    "weather.csv": [
        "region", "date", "temperature_max", "precipitation_sum", "windspeed_max",
    ],
    "incidents.csv": [
        "incident_id", "asset_id", "date", "cause",
        "duration_hours", "customers_affected",
    ],
}


def _read_and_validate(filename: str) -> pd.DataFrame:
    """
    Load a CSV from DATA_OUTPUT_DIR, validate that all expected columns are
    present, and return the DataFrame.

    Raises:
        FileNotFoundError: if the CSV does not exist.
        ValueError:        if required columns are missing.
    """
    path = DATA_OUTPUT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Expected data file not found: {path}. "
            "Run data_generation scripts first."
        )

    df = pd.read_csv(path)
    required = EXPECTED_COLUMNS[filename]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{filename} is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    logger.info("Loaded %s — %d rows", filename, len(df))
    return df


def extract_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read and validate all four source CSVs.

    Returns:
        (assets_df, sensor_df, weather_df, incidents_df)
    """
    assets_df    = _read_and_validate("assets.csv")
    sensor_df    = _read_and_validate("sensor_readings.csv")
    weather_df   = _read_and_validate("weather.csv")
    incidents_df = _read_and_validate("incidents.csv")
    return assets_df, sensor_df, weather_df, incidents_df
