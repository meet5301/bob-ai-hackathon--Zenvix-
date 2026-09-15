"""
main.py
-------
FastAPI service exposing the Grid Guard prediction, ranking, and
watsonx.ai maintenance-plan endpoints.
"""

import logging
import os
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from feature_engineering.build_features import (
    build_training_table,
    compute_rolling_features,
)
from llm.generate_plan import generate_maintenance_plan
from ml.predict import load_model, predict_failure_probability, rank_by_severity

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Grid Guard API",
    description="Power Outage Prediction & Grid Equipment Failure Advisor — Team Maverick",
    version="1.0.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
_dashboard_origin = os.environ.get("API_BASE_URL", "http://localhost:8501")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[_dashboard_origin, "http://localhost:8501"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ── Pydantic response models ──────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str


class PredictionRow(BaseModel):
    asset_id: str
    date: str
    failure_probability: float


class PredictionsResponse(BaseModel):
    predictions: list[PredictionRow]


class RankedAssetRow(BaseModel):
    asset_id: str
    region: str
    asset_type: str
    criticality_tier: int
    customers_served: int
    failure_probability: float
    criticality_weight: float
    severity_score: float


class RankingsResponse(BaseModel):
    rankings: list[RankedAssetRow]


class PlanResponse(BaseModel):
    plan: str


# ── Shared pipeline helpers ───────────────────────────────────────────────────

def _load_model_cached():
    model_path = os.environ.get("MODEL_PATH")
    if not model_path:
        raise EnvironmentError("MODEL_PATH is not set.")
    return load_model(model_path)


def _run_pipeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the feature-engineering → predict pipeline and return
    (predictions_df, assets_df).  Loads data from local CSV files.
    """
    data_dir = os.environ.get("DATA_OUTPUT_DIR", "data/raw")
    sensor_df = pd.read_csv(os.path.join(data_dir, "sensor_readings.csv"))
    assets_df = pd.read_csv(os.path.join(data_dir, "assets.csv"))

    model, feature_cols = _load_model_cached()
    features_df    = compute_rolling_features(sensor_df)
    training_df    = build_training_table(features_df, assets_df)
    predictions_df = predict_failure_probability(model, feature_cols, training_df)
    return predictions_df, assets_df


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/predictions", response_model=PredictionsResponse)
def predictions_endpoint() -> PredictionsResponse:
    try:
        predictions_df, _ = _run_pipeline()
    except Exception as exc:
        logger.exception("Failed to compute predictions.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    rows = [
        PredictionRow(
            asset_id=row["asset_id"],
            date=str(row["date"]),
            failure_probability=float(row["failure_probability"]),
        )
        for _, row in predictions_df.iterrows()
    ]
    return PredictionsResponse(predictions=rows)


@app.get("/rankings", response_model=RankingsResponse)
def rankings_endpoint() -> RankingsResponse:
    try:
        predictions_df, assets_df = _run_pipeline()
        ranked_df = rank_by_severity(predictions_df, assets_df)
    except Exception as exc:
        logger.exception("Failed to compute rankings.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    rows = [
        RankedAssetRow(
            asset_id=row["asset_id"],
            region=row["region"],
            asset_type=row["asset_type"],
            criticality_tier=int(row["criticality_tier"]),
            customers_served=int(row["customers_served"]),
            failure_probability=float(row["failure_probability"]),
            criticality_weight=float(row["criticality_weight"]),
            severity_score=float(row["severity_score"]),
        )
        for _, row in ranked_df.iterrows()
    ]
    return RankingsResponse(rankings=rows)


@app.get("/plan", response_model=PlanResponse)
def plan_endpoint() -> PlanResponse:
    try:
        predictions_df, assets_df = _run_pipeline()
        ranked_df = rank_by_severity(predictions_df, assets_df)
        plan_text = generate_maintenance_plan(ranked_df, top_n=10)
    except EnvironmentError as exc:
        logger.error("Watsonx environment not configured: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to generate maintenance plan.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PlanResponse(plan=plan_text)
