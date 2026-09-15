"""
generate_fake_data.py
---------------------
Generates synthetic grid assets, daily sensor readings with realistic
degradation patterns, and outage incidents.  Writes three CSV files to
DATA_OUTPUT_DIR (from environment config).
"""

import logging
import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_OUTPUT_DIR = Path(os.environ["DATA_OUTPUT_DIR"])

ASSET_TYPES = ["transformer", "substation", "breaker"]

# Approximate bounding boxes (lat_min, lat_max, lon_min, lon_max) per region
REGION_BOUNDS: dict[str, tuple[float, float, float, float]] = {
    "North": (52.0, 54.0, -2.0, 0.5),
    "South": (50.5, 52.0, -1.5, 1.0),
    "East":  (51.5, 53.0,  0.5, 2.0),
    "West":  (51.0, 53.5, -4.0, -2.0),
}

INCIDENT_CAUSES = [
    "lightning_strike",
    "overload",
    "equipment_aging",
    "wildlife_contact",
    "vegetation_contact",
    "human_error",
]

# Sensor baseline statistics (mean, std) for healthy assets
SENSOR_BASELINES: dict[str, tuple[float, float]] = {
    "temperature":       (60.0, 3.0),
    "vibration":         (1.0,  0.15),
    "oil_quality":       (85.0, 2.0),
    "partial_discharge": (5.0,  1.0),
}

# How far each sensor degrades in the 30-day run-up to failure
DEGRADATION_DELTAS: dict[str, float] = {
    "temperature":       30.0,   # rises toward 90 °C
    "vibration":         2.5,    # increases sharply
    "oil_quality":       -35.0,  # drops toward 50
    "partial_discharge": 20.0,   # spikes up
}


# ── Asset generation ──────────────────────────────────────────────────────────

def generate_assets(n_assets: int) -> pd.DataFrame:
    """Return a DataFrame of synthetic grid assets."""
    rng = np.random.default_rng(42)
    regions = list(REGION_BOUNDS.keys())

    rows = []
    for i in range(n_assets):
        region = regions[i % len(regions)]
        lat_min, lat_max, lon_min, lon_max = REGION_BOUNDS[region]
        rows.append(
            {
                "asset_id":        f"ASSET-{i+1:04d}",
                "asset_type":      rng.choice(ASSET_TYPES),
                "region":          region,
                "lat":             round(float(rng.uniform(lat_min, lat_max)), 6),
                "lon":             round(float(rng.uniform(lon_min, lon_max)), 6),
                "install_year":    int(rng.integers(1985, 2022)),
                "customers_served": int(rng.integers(100, 5001)),
                "criticality_tier": int(rng.integers(1, 4)),
            }
        )
    return pd.DataFrame(rows)


# ── Sensor series generation ──────────────────────────────────────────────────

def _degradation_ramp(day_index: int, total_days: int, ramp_days: int = 30) -> float:
    """
    Returns a scalar in [0.0, 1.0] representing how far through the degradation
    ramp day_index falls.  Only the final `ramp_days` of the series ramp up;
    earlier days return 0.
    """
    ramp_start = total_days - ramp_days
    if day_index < ramp_start:
        return 0.0
    return (day_index - ramp_start) / ramp_days


def generate_sensor_series(
    asset_id: str, will_fail: bool, days: int, rng: np.random.Generator
) -> pd.DataFrame:
    """
    Generate a daily sensor time-series for a single asset.

    Healthy assets remain noisy but stable around SENSOR_BASELINES.
    Failing assets exhibit a smooth degradation ramp in the final 30 days,
    with the `failed` flag set to 1 on the last day only.
    """
    records = []
    start_date = pd.Timestamp("2024-01-01")

    for day in range(days):
        ramp = _degradation_ramp(day, days) if will_fail else 0.0

        readings = {}
        for sensor, (base_mean, base_std) in SENSOR_BASELINES.items():
            delta = DEGRADATION_DELTAS[sensor] * ramp
            value = float(rng.normal(base_mean + delta, base_std))
            readings[sensor] = round(value, 3)

        failed_flag = 1 if (will_fail and day == days - 1) else 0

        records.append(
            {
                "asset_id":          asset_id,
                "date":              (start_date + pd.Timedelta(days=day)).date().isoformat(),
                **readings,
                "failed":            failed_flag,
            }
        )
    return pd.DataFrame(records)


# ── Incident generation ───────────────────────────────────────────────────────

def generate_incidents(
    assets_df: pd.DataFrame, n_incidents: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Return a DataFrame of synthetic outage incidents linked to assets."""
    asset_ids = assets_df["asset_id"].tolist()
    start_date = pd.Timestamp("2024-01-01")

    rows = []
    for _ in range(n_incidents):
        rows.append(
            {
                "incident_id":       str(uuid.uuid4()),
                "asset_id":          rng.choice(asset_ids),
                "date":              (
                    start_date + pd.Timedelta(days=int(rng.integers(0, 180)))
                ).date().isoformat(),
                "cause":             rng.choice(INCIDENT_CAUSES),
                "duration_hours":    round(float(rng.uniform(0.5, 72.0)), 2),
                "customers_affected": int(rng.integers(10, 3001)),
            }
        )
    return pd.DataFrame(rows)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    np.random.seed(42)  # legacy seed for reproducibility
    rng = np.random.default_rng(42)

    n_assets = 100
    days = 180
    n_incidents = 25
    fail_fraction = 0.15

    DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Generating %d assets …", n_assets)
    assets_df = generate_assets(n_assets)
    assets_path = DATA_OUTPUT_DIR / "assets.csv"
    assets_df.to_csv(assets_path, index=False)
    logger.info("Saved assets → %s", assets_path)

    # Decide which assets will fail
    n_failing = max(1, int(n_assets * fail_fraction))
    failing_ids = set(assets_df["asset_id"].sample(n=n_failing, random_state=42).tolist())
    logger.info("%d assets marked as failing", n_failing)

    logger.info("Generating sensor series (%d days × %d assets) …", days, n_assets)
    sensor_frames = []
    for asset_id in assets_df["asset_id"]:
        will_fail = asset_id in failing_ids
        sensor_frames.append(generate_sensor_series(asset_id, will_fail, days, rng))
    sensor_df = pd.concat(sensor_frames, ignore_index=True)
    sensor_path = DATA_OUTPUT_DIR / "sensor_readings.csv"
    sensor_df.to_csv(sensor_path, index=False)
    logger.info("Saved sensor readings → %s", sensor_path)

    logger.info("Generating %d incidents …", n_incidents)
    incidents_df = generate_incidents(assets_df, n_incidents, rng)
    incidents_path = DATA_OUTPUT_DIR / "incidents.csv"
    incidents_df.to_csv(incidents_path, index=False)
    logger.info("Saved incidents → %s", incidents_path)


if __name__ == "__main__":
    main()
