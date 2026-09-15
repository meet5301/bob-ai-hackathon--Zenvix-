"""
train_model.py
--------------
Loads the training table from feature_engineering, trains an XGBoost
failure-prediction classifier and a logistic-regression baseline, logs
comparison metrics, and saves the XGBoost model to disk.
"""

import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

LABEL_COL = "failed"
DROP_COLS  = {"asset_id", "date", LABEL_COL}


def _get_model_path() -> Path:
    raw = os.environ.get("MODEL_PATH")
    if not raw:
        raise EnvironmentError(
            "MODEL_PATH is not set. Add it to your .env file "
            "(e.g. MODEL_PATH=models/xgb_failure_model.joblib)."
        )
    return Path(raw)


def _split_features_label(
    training_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    feature_cols = [c for c in training_df.columns if c not in DROP_COLS]
    X = training_df[feature_cols]
    y = training_df[LABEL_COL].astype(int)
    return X, y, feature_cols


def _log_metrics(name: str, y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> None:
    report = classification_report(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    logger.info("── %s ──\n%s\nAUC-ROC: %.4f", name, report, auc)


def train(training_df: pd.DataFrame) -> XGBClassifier:
    """
    Train an XGBoost classifier and a logistic-regression baseline on the
    training table.  Logs precision, recall, F1, and AUC for both models.
    Saves the XGBoost model to MODEL_PATH and returns it.
    """
    X, y, feature_cols = _split_features_label(training_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    logger.info(
        "Train/test split: %d train rows, %d test rows (label balance %.2f%%)",
        len(X_train), len(X_test), 100 * y_train.mean(),
    )

    # ── XGBoost ────────────────────────────────────────────────────────────
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False,
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    xgb_pred = xgb_model.predict(X_test)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    _log_metrics("XGBoost", y_test, xgb_pred, xgb_prob)

    # ── Logistic Regression baseline ────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    lr_model = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    )
    lr_model.fit(X_train_scaled, y_train)

    lr_pred = lr_model.predict(X_test_scaled)
    lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]
    _log_metrics("Logistic Regression (baseline)", y_test, lr_pred, lr_prob)

    # ── Feature importances ───────────────────────────────────────────────
    importances = pd.Series(
        xgb_model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    logger.info("Top 10 feature importances:\n%s", importances.head(10).to_string())

    # ── Save XGBoost model ────────────────────────────────────────────────
    model_path = _get_model_path()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": xgb_model, "feature_cols": feature_cols}, model_path)
    logger.info("XGBoost model saved → %s", model_path)

    return xgb_model


def main() -> None:
    np.random.seed(42)

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

    features_df  = compute_rolling_features(sensor_df)
    training_df  = build_training_table(features_df, assets_df)
    train(training_df)


if __name__ == "__main__":
    main()
