"""
predict.py
----------
Loads the trained XGBoost model, runs failure-probability inference, and
ranks assets by a composite severity score for the API / dashboard.
"""

import logging
import os
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from xgboost import XGBClassifier

from feature_engineering.build_features import (
    build_training_table,
    compute_rolling_features,
    get_engine,
    _load_table,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Criticality-tier multipliers for severity scoring
CRITICALITY_WEIGHTS: dict[int, float] = {1: 3.0, 2: 2.0, 3: 1.0}


def load_model(path: str | Path) -> tuple[XGBClassifier, list[str]]:
    """
    Load the saved XGBoost model artifact.

    Returns:
        (xgb_model, feature_cols) tuple as stored by train_model.py.

    Raises:
        FileNotFoundError: if the model file does not exist.
        RuntimeError:      if the file cannot be deserialized.
    """
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            "Run ml/train_model.py first to generate it."
        )
    try:
        artifact = joblib.load(model_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model from '{model_path}': {exc}") from exc

    return artifact["model"], artifact["feature_cols"]


def predict_failure_probability(
    model: XGBClassifier,
    feature_cols: list[str],
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run failure-probability inference.

    Args:
        model:        Trained XGBClassifier.
        feature_cols: Column names the model was trained on (from the saved artifact).
        features_df:  DataFrame that must contain at minimum `asset_id`, `date`,
                      and all columns in `feature_cols`.

    Returns:
        DataFrame with columns: asset_id, date, failure_probability.

    Raises:
        ValueError: if any expected feature column is absent.
    """
    missing = [c for c in feature_cols if c not in features_df.columns]
    if missing:
        raise ValueError(
            f"features_df is missing columns required by the model: {missing}"
        )

    X = features_df[feature_cols]
    probabilities = model.predict_proba(X)[:, 1]

    result = features_df[["asset_id", "date"]].copy()
    result["failure_probability"] = probabilities
    return result


def rank_by_severity(
    predictions_df: pd.DataFrame,
    assets_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute a severity score for every asset and return a DataFrame sorted
    descending by severity.

    severity_score = failure_probability × customers_served × criticality_weight

    Where criticality_weight maps criticality_tier → {1: 3, 2: 2, 3: 1}.
    Higher tiers (lower tier number) receive a larger multiplier because they
    affect more critical infrastructure.

    Returns columns:
        asset_id, region, asset_type, criticality_tier, customers_served,
        failure_probability, criticality_weight, severity_score
    """
    # Use the most-recent prediction per asset
    latest = (
        predictions_df
        .sort_values("date", ascending=False)
        .drop_duplicates(subset="asset_id")
    )

    merged = latest.merge(
        assets_df[["asset_id", "region", "asset_type", "criticality_tier", "customers_served"]],
        on="asset_id",
        how="inner",
    )

    merged["criticality_weight"] = (
        merged["criticality_tier"]
        .map(CRITICALITY_WEIGHTS)
        .fillna(1.0)
    )
    merged["severity_score"] = (
        merged["failure_probability"]
        * merged["customers_served"]
        * merged["criticality_weight"]
    )

    ranked = merged.sort_values("severity_score", ascending=False).reset_index(drop=True)
    logger.info(
        "Ranked %d assets; top severity_score = %.2f",
        len(ranked), ranked["severity_score"].iloc[0] if len(ranked) else 0.0,
    )
    return ranked


def main() -> pd.DataFrame:
    model_path = os.environ.get("MODEL_PATH")
    if not model_path:
        raise EnvironmentError("MODEL_PATH is not set in environment / .env.")

    model, feature_cols = load_model(model_path)
    engine = get_engine()

    sensor_df = _load_table(
        engine,
        "SELECT asset_id, date, temperature, vibration, oil_quality, "
        "partial_discharge, failed FROM sensor_readings",
        "sensor_readings",
    )
    assets_df = _load_table(
        engine,
        "SELECT asset_id, region, asset_type, criticality_tier, customers_served FROM assets",
        "assets",
    )

    features_df  = compute_rolling_features(sensor_df)
    training_df  = build_training_table(features_df, assets_df)
    predictions  = predict_failure_probability(model, feature_cols, training_df)
    ranked       = rank_by_severity(predictions, assets_df)

    output_dir = Path(os.environ.get("DATA_OUTPUT_DIR", "data/raw"))
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "ranked_assets.csv"
    ranked.to_csv(out_path, index=False)
    logger.info("Ranked asset table saved → %s", out_path)

    return ranked


if __name__ == "__main__":
    main()
