"""
load_to_neon.py
---------------
Connects to Neon (Postgres), ensures the schema exists, then runs the
full extract → transform → load pipeline.
"""

import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from etl.extract import extract_all
from etl.transform import clean_and_join

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def get_engine():
    """Create and return a SQLAlchemy engine from NEON_DATABASE_URL."""
    db_url = os.environ.get("NEON_DATABASE_URL")
    if not db_url:
        raise EnvironmentError(
            "NEON_DATABASE_URL is not set. "
            "Copy .env.example → .env and fill in your Neon connection string."
        )
    return create_engine(db_url)


def apply_schema(engine) -> None:
    """
    Execute schema.sql against the database.
    Uses CREATE TABLE IF NOT EXISTS, so it is safe to run repeatedly.
    """
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    try:
        with engine.begin() as conn:
            conn.execute(text(schema_sql))
        logger.info("Schema applied (or already exists).")
    except SQLAlchemyError as exc:
        raise RuntimeError(f"Failed to apply schema: {exc}") from exc


def load_dataframe(df: pd.DataFrame, table_name: str, engine) -> None:
    """
    Load a DataFrame into a Postgres table using truncate-and-reload.

    Strategy: TRUNCATE the target table (cascading to dependents if needed)
    then bulk-insert all rows in a single transaction.  This keeps the load
    idempotent — re-running the pipeline produces an identical, consistent DB
    state without accumulating duplicate rows.  A full upsert would require
    knowledge of unique keys for every table; truncate-and-reload is simpler
    and safe here because the entire data set is regenerated each run.

    Note: Tables with FK dependents (sensor_readings, incidents) are truncated
    after assets to avoid FK violations — callers must load assets first.
    """
    try:
        with engine.begin() as conn:
            conn.execute(text(f"TRUNCATE TABLE {table_name} RESTART IDENTITY CASCADE"))
            df.to_sql(
                table_name,
                con=conn,
                if_exists="append",
                index=False,
                method="multi",
            )
        logger.info("Loaded %d rows into '%s'.", len(df), table_name)
    except SQLAlchemyError as exc:
        raise RuntimeError(
            f"Failed to load data into table '{table_name}': {exc}"
        ) from exc


def main() -> None:
    engine = get_engine()

    apply_schema(engine)

    logger.info("Extracting source CSVs …")
    assets_df, sensor_df, weather_df, incidents_df = extract_all()

    logger.info("Transforming data …")
    sensor_enriched, weather_clean, incidents_clean = clean_and_join(
        assets_df, sensor_df, weather_df, incidents_df
    )

    # Load in FK-safe order: assets first, then dependent tables.
    logger.info("Loading assets …")
    load_dataframe(assets_df, "assets", engine)

    logger.info("Loading weather …")
    load_dataframe(weather_clean, "weather", engine)

    logger.info("Loading sensor_readings …")
    # sensor_enriched carries extra joined columns (region, weather cols);
    # keep only the columns that map to the sensor_readings table schema.
    sensor_table_cols = [
        "asset_id", "date", "temperature", "vibration",
        "oil_quality", "partial_discharge", "failed",
    ]
    load_dataframe(sensor_enriched[sensor_table_cols], "sensor_readings", engine)

    logger.info("Loading incidents …")
    load_dataframe(incidents_clean, "incidents", engine)

    logger.info("ETL pipeline complete.")


if __name__ == "__main__":
    main()
