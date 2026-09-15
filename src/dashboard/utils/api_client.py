"""
api_client.py
-------------
Thin wrapper around the Grid Guard FastAPI endpoints.
Each function returns the parsed payload on success, or raises a
RuntimeError with a descriptive message that the UI can display.
"""

import logging

import pandas as pd
import requests

from dashboard.config import API_BASE_URL

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 30


def get_predictions() -> pd.DataFrame:
    """
    Fetch failure-probability predictions from GET /predictions.

    Returns:
        DataFrame with columns: asset_id, date, failure_probability.

    Raises:
        RuntimeError: on network failure or non-200 response.
    """
    url = f"{API_BASE_URL}/predictions"
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Could not reach predictions endpoint ({url}): {exc}") from exc

    if response.status_code != 200:
        raise RuntimeError(
            f"Predictions endpoint returned HTTP {response.status_code}: "
            f"{response.text[:300]}"
        )

    rows = response.json().get("predictions", [])
    logger.info("Received %d prediction rows.", len(rows))
    return pd.DataFrame(rows)


def get_rankings() -> pd.DataFrame:
    """
    Fetch severity-ranked assets from GET /rankings.

    Returns:
        DataFrame with columns: asset_id, region, asset_type,
        criticality_tier, customers_served, failure_probability,
        criticality_weight, severity_score.

    Raises:
        RuntimeError: on network failure or non-200 response.
    """
    url = f"{API_BASE_URL}/rankings"
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Could not reach rankings endpoint ({url}): {exc}") from exc

    if response.status_code != 200:
        raise RuntimeError(
            f"Rankings endpoint returned HTTP {response.status_code}: "
            f"{response.text[:300]}"
        )

    rows = response.json().get("rankings", [])
    logger.info("Received %d ranked asset rows.", len(rows))
    return pd.DataFrame(rows)


def get_plan() -> str:
    """
    Fetch the watsonx-generated maintenance plan from GET /plan.

    Returns:
        The plan as a plain-text string.

    Raises:
        RuntimeError: on network failure, non-200 response, or if the API
                      signals a genuine generation failure.
    """
    url = f"{API_BASE_URL}/plan"
    try:
        response = requests.get(url, timeout=120)  # LLM generation can be slower
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Could not reach plan endpoint ({url}): {exc}") from exc

    if response.status_code != 200:
        error_detail = response.json().get("detail", response.text[:300])
        raise RuntimeError(
            f"Plan endpoint returned HTTP {response.status_code}: {error_detail}"
        )

    plan = response.json().get("plan", "")
    if not plan:
        raise RuntimeError("Plan endpoint returned an empty plan.")

    logger.info("Received maintenance plan (%d characters).", len(plan))
    return plan
