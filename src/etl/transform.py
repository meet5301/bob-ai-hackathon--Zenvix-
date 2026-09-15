"""
transform.py
------------
Cleans and joins the four source DataFrames (assets, sensor_readings,
weather, incidents) into a single enriched DataFrame ready for loading
into Neon.
"""

import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _standardize_dates(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Parse a date column to ISO 8601 strings (YYYY-MM-DD)."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
    return df


def _drop_duplicates(df: pd.DataFrame, label: str) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d duplicate rows from %s.", dropped, label)
    return df


def clean_and_join(
    assets: pd.DataFrame,
    sensor_readings: pd.DataFrame,
    weather: pd.DataFrame,
    incidents: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Clean and join the four source DataFrames.

    Missing-value strategy (documented per column type):
    - sensor readings (temperature, vibration, oil_quality, partial_discharge):
        forward-filled within each asset_id group for short gaps (≤ 3 consecutive
        NaN rows); any remaining NaNs are back-filled; rows still missing after
        both passes are dropped — they cannot contribute useful signal.
    - weather columns: forward-filled within each region for short gaps.
    - asset_id FK: rows with a null or unrecognised asset_id are dropped because
        they cannot be joined to the asset registry.
    - incident columns (cause, duration_hours, customers_affected): kept as-is;
        nulls are meaningful (unknown cause / unreported duration).

    Returns:
        (enriched_sensor_df, weather_df, incidents_df) — three cleaned DataFrames
        in the correct order for loading (assets table is unchanged; callers load
        the original assets DataFrame directly).
    """
    # ── Standardize dates ──────────────────────────────────────────────────
    assets         = _standardize_dates(assets.copy(),         "install_year" if False else "install_year")
    sensor_readings = _standardize_dates(sensor_readings.copy(), "date")
    weather        = _standardize_dates(weather.copy(),         "date")
    incidents      = _standardize_dates(incidents.copy(),       "date")

    # install_year is a year integer, not a date — leave it alone
    assets = assets.copy()

    # ── Deduplicate ───────────────────────────────────────────────────────
    assets          = _drop_duplicates(assets,          "assets")
    sensor_readings = _drop_duplicates(sensor_readings, "sensor_readings")
    weather         = _drop_duplicates(weather,         "weather")
    incidents       = _drop_duplicates(incidents,       "incidents")

    # ── Drop rows with missing join keys ──────────────────────────────────
    known_asset_ids = set(assets["asset_id"])

    for df_name, df in (("sensor_readings", sensor_readings), ("incidents", incidents)):
        invalid_mask = ~df["asset_id"].isin(known_asset_ids) | df["asset_id"].isna()
        n_invalid = invalid_mask.sum()
        if n_invalid:
            logger.warning(
                "Dropping %d rows from %s with unknown/null asset_id.", n_invalid, df_name
            )
    sensor_readings = sensor_readings[sensor_readings["asset_id"].isin(known_asset_ids)]
    incidents       = incidents[incidents["asset_id"].isin(known_asset_ids)]

    # ── Handle missing sensor values ──────────────────────────────────────
    sensor_cols = ["temperature", "vibration", "oil_quality", "partial_discharge"]
    sensor_readings = (
        sensor_readings
        .sort_values(["asset_id", "date"])
        .groupby("asset_id", group_keys=False)
        .apply(lambda g: g.fillna(method="ffill", limit=3).fillna(method="bfill", limit=3))
    )
    rows_before = len(sensor_readings)
    sensor_readings = sensor_readings.dropna(subset=sensor_cols)
    rows_dropped = rows_before - len(sensor_readings)
    if rows_dropped:
        logger.warning(
            "Dropped %d sensor rows with unfillable NaNs.", rows_dropped
        )

    # ── Handle missing weather values ─────────────────────────────────────
    weather_val_cols = ["temperature_max", "precipitation_sum", "windspeed_max"]
    weather = (
        weather
        .sort_values(["region", "date"])
        .groupby("region", group_keys=False)
        .apply(lambda g: g.fillna(method="ffill", limit=3).fillna(method="bfill", limit=3))
    )

    # ── Join sensor_readings ↔ assets (to get region) ─────────────────────
    sensor_enriched = sensor_readings.merge(
        assets[["asset_id", "region"]],
        on="asset_id",
        how="inner",
    )

    # ── Join sensor_enriched ↔ weather (on region + date) ─────────────────
    sensor_enriched = sensor_enriched.merge(
        weather,
        on=["region", "date"],
        how="left",
    )

    unmatched_weather = sensor_enriched[weather_val_cols].isna().any(axis=1).sum()
    if unmatched_weather:
        logger.warning(
            "%d sensor rows could not be matched to a weather record.", unmatched_weather
        )

    logger.info(
        "Transformation complete: %d sensor rows, %d weather rows, %d incident rows.",
        len(sensor_enriched), len(weather), len(incidents),
    )

    return sensor_enriched, weather, incidents
