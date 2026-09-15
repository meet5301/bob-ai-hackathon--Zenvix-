"""
build_features.py
-----------------
Reads cleaned sensor and asset data from Neon, computes per-asset rolling
statistics, and builds the final training table used by ml/train_model.py.
"""

import logging
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SENSOR_COLS = ["temperature", "vibration", "oil_quality", "partial_discharge"]
ROLLING_WINDOWS = [7, 30]


def get_engine():
    """Create a SQLAlchemy engine from NEON_DATABASE_URL."""
    db_url = os.environ.get("NEON_DATABASE_URL")
    if not db_url:
        raise EnvironmentError(
            "NEON_DATABASE_URL is not set. Fill in .env before running."
        )
    return create_engine(db_url)


def _load_table(engine, query: str, label: str) -> pd.DataFrame:
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        logger.info("Loaded %s — %d rows", label, len(df))
        return df
    except SQLAlchemyError as exc:
        raise RuntimeError(f"Failed to query '{label}' from Neon: {exc}") from exc


def _validate_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame '{label}' is missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """
    Compute a rolling linear-regression slope over `window` observations.
    A positive slope on temperature/vibration/partial_discharge, or a negative
    slope on oil_quality, is the key degradation signal.
    Returns NaN for positions where fewer than `window` points are available.
    """
    def _slope(values: np.ndarray) -> float:
        if len(values) < window or np.isnan(values).any():
            return np.nan
        x = np.arange(len(values), dtype=float)
        coeffs = np.polyfit(x, values, 1)
        return float(coeffs[0])

    return series.rolling(window).apply(_slope, raw=True)


def compute_rolling_features(sensor_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each asset, compute 7-day and 30-day rolling mean, std, and slope for
    all four sensor columns.  Sorting by asset_id + date guarantees the rolling
    windows are computed in time order.

    Requires columns: asset_id, date, temperature, vibration,
                      oil_quality, partial_discharge, failed.
    """
    required = ["asset_id", "date"] + SENSOR_COLS + ["failed"]
    _validate_columns(sensor_df, required, "sensor_df")

    sensor_df = sensor_df.sort_values(["asset_id", "date"]).copy()
    sensor_df["date"] = pd.to_datetime(sensor_df["date"])

    feature_frames = []

    for asset_id, group in sensor_df.groupby("asset_id", sort=False):
        frame = group[["asset_id", "date", "failed"]].copy()

        for col in SENSOR_COLS:
            series = group[col]
            for window in ROLLING_WINDOWS:
                frame[f"{col}_roll{window}_mean"]  = series.rolling(window).mean().values
                frame[f"{col}_roll{window}_std"]   = series.rolling(window).std().values
                frame[f"{col}_roll{window}_slope"] = _rolling_slope(series, window).values

        feature_frames.append(frame)

    features_df = pd.concat(feature_frames, ignore_index=True)
    logger.info(
        "Rolling features computed: %d rows × %d columns",
        len(features_df), len(features_df.columns),
    )
    return features_df


def build_training_table(
    features_df: pd.DataFrame,
    assets_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join rolling features with asset metadata (criticality_tier,
    customers_served) and keep only rows where all feature columns are
    non-null (i.e. once the rolling window has enough history).

    Returns the final table ready for ml/train_model.py.
    """
    _validate_columns(
        assets_df, ["asset_id", "criticality_tier", "customers_served"], "assets_df"
    )

    training_df = features_df.merge(
        assets_df[["asset_id", "criticality_tier", "customers_served"]],
        on="asset_id",
        how="inner",
    )

    # Drop rows that still have NaN features (early window warm-up rows)
    feature_cols = [c for c in training_df.columns
                    if c not in ("asset_id", "date", "failed")]
    rows_before = len(training_df)
    training_df = training_df.dropna(subset=feature_cols)
    logger.info(
        "Dropped %d warm-up rows; training table has %d rows.",
        rows_before - len(training_df), len(training_df),
    )

    return training_df


def main() -> None:
    engine = get_engine()

    sensor_df = _load_table(
        engine,
        "SELECT asset_id, date, temperature, vibration, oil_quality, "
        "partial_discharge, failed FROM sensor_readings",
        "sensor_readings",
    )
    assets_df = _load_table(
        engine,
        "SELECT asset_id, criticality_tier, customers_served FROM assets",
        "assets",
    )

    features_df = compute_rolling_features(sensor_df)
    training_df = build_training_table(features_df, assets_df)

    output_path = os.environ.get("DATA_OUTPUT_DIR", "data/raw")
    import pathlib
    out = pathlib.Path(output_path) / "training_table.csv"
    training_df.to_csv(out, index=False)
    logger.info("Training table saved → %s", out)


if __name__ == "__main__":
    main()
