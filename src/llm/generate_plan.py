"""
generate_plan.py
----------------
Calls IBM watsonx.ai (Bob) to generate a prioritised maintenance and crew
pre-positioning plan from the top-N ranked assets.
"""

import logging
import os

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"
WATSONX_MODEL_ID = "ibm/granite-13b-instruct-v2"
REQUEST_TIMEOUT_SECONDS = 60


def _get_iam_token(api_key: str) -> str:
    """
    Exchange an IBM Cloud API key for a short-lived IAM bearer token.

    Raises:
        RuntimeError: if the token exchange fails.
    """
    try:
        response = requests.post(
            IAM_TOKEN_URL,
            data={
                "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                "apikey": api_key,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"IAM token request failed: {exc}") from exc

    if response.status_code != 200:
        raise RuntimeError(
            f"IAM token exchange returned HTTP {response.status_code}: "
            f"{response.text[:300]}"
        )

    token = response.json().get("access_token")
    if not token:
        raise RuntimeError("IAM response did not include an access_token.")
    return token


def _build_prompt(ranked_assets_df: pd.DataFrame, top_n: int) -> str:
    """
    Build the structured natural-language prompt for watsonx.ai from the
    top-N highest-severity assets.
    """
    top = ranked_assets_df.head(top_n)

    asset_lines = []
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        asset_lines.append(
            f"  {rank}. Asset {row['asset_id']} | Region: {row['region']} | "
            f"Type: {row.get('asset_type', 'unknown')} | "
            f"Criticality Tier: {int(row['criticality_tier'])} | "
            f"Failure Probability: {row['failure_probability']:.1%} | "
            f"Severity Score: {row['severity_score']:.1f}"
        )

    asset_block = "\n".join(asset_lines)

    prompt = (
        "You are an expert power-grid operations manager. "
        "Based on the asset risk rankings below, produce a concise, prioritised "
        "maintenance and crew pre-positioning plan. "
        "For each asset, specify: recommended action (inspect / repair / replace), "
        "urgency level (immediate / within 48 h / within 7 days), "
        "and suggested crew size. "
        "End with a brief regional crew staging recommendation.\n\n"
        f"Top {top_n} At-Risk Assets (ranked by severity score):\n"
        f"{asset_block}\n\n"
        "Maintenance Plan:"
    )
    return prompt


def _call_watsonx(prompt: str, api_key: str, project_id: str, watsonx_url: str) -> str:
    """
    Submit the prompt to the watsonx.ai text-generation endpoint and return
    the generated text.

    Raises:
        RuntimeError: on any HTTP or parsing failure.
    """
    token = _get_iam_token(api_key)

    endpoint = f"{watsonx_url.rstrip('/')}/ml/v1/text/generation?version=2023-05-29"

    payload = {
        "model_id": WATSONX_MODEL_ID,
        "input": prompt,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 800,
            "min_new_tokens": 50,
            "stop_sequences": [],
        },
        "project_id": project_id,
    }

    try:
        response = requests.post(
            endpoint,
            json=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"watsonx.ai API request failed: {exc}") from exc

    if response.status_code != 200:
        raise RuntimeError(
            f"watsonx.ai returned HTTP {response.status_code}: "
            f"{response.text[:400]}"
        )

    results = response.json().get("results", [])
    if not results or "generated_text" not in results[0]:
        raise RuntimeError(
            "watsonx.ai response did not contain generated_text. "
            f"Raw response: {response.text[:400]}"
        )

    return results[0]["generated_text"].strip()


def generate_maintenance_plan(
    ranked_assets_df: pd.DataFrame,
    top_n: int = 10,
) -> str:
    """
    Build a prioritised maintenance plan for the top-N at-risk assets by
    calling IBM watsonx.ai.

    Raises:
        EnvironmentError: if required env vars are missing.
        RuntimeError:     if the API call fails — no hardcoded fallback is
                          returned; the caller must handle the failure visibly.
    """
    required_cols = ["asset_id", "region", "criticality_tier",
                     "failure_probability", "severity_score"]
    missing = [c for c in required_cols if c not in ranked_assets_df.columns]
    if missing:
        raise ValueError(
            f"ranked_assets_df is missing columns: {missing}"
        )

    api_key    = os.environ.get("WATSONX_API_KEY")
    project_id = os.environ.get("WATSONX_PROJECT_ID")
    url        = os.environ.get("WATSONX_URL")

    if not all([api_key, project_id, url]):
        missing_vars = [
            k for k, v in {
                "WATSONX_API_KEY": api_key,
                "WATSONX_PROJECT_ID": project_id,
                "WATSONX_URL": url,
            }.items() if not v
        ]
        raise EnvironmentError(
            f"Missing required environment variables: {missing_vars}. "
            "Set them in your .env file."
        )

    prompt = _build_prompt(ranked_assets_df, top_n)
    logger.info("Sending prompt to watsonx.ai for %d assets …", min(top_n, len(ranked_assets_df)))

    plan = _call_watsonx(prompt, api_key, project_id, url)
    logger.info("Maintenance plan generated (%d characters).", len(plan))
    return plan
