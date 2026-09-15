"""
config.py
---------
Single source of truth for all dashboard-wide configuration.
All other dashboard modules must import from here — never read environment
variables or define constants elsewhere.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API ───────────────────────────────────────────────────────────────────────
# Base URL of the FastAPI service.  Must be set in .env; no fallback is silently
# applied so that a misconfigured environment surfaces immediately.
_raw_api_url = os.environ.get("API_BASE_URL")
if not _raw_api_url:
    raise EnvironmentError(
        "API_BASE_URL is not set. "
        "Copy .env.example → .env and set API_BASE_URL to your FastAPI base URL."
    )
API_BASE_URL: str = _raw_api_url.rstrip("/")

# ── Risk-level colour mapping ─────────────────────────────────────────────────
# Used by asset_map.py and risk_table.py to colour-code severity buckets.
RISK_COLORS: dict[str, str] = {
    "High":   "#d73027",   # red
    "Medium": "#fc8d59",   # orange
    "Low":    "#91bfdb",   # blue
}

# Severity-score thresholds that determine the risk bucket.
# Any score >= HIGH_THRESHOLD  → "High"
# Any score >= MEDIUM_THRESHOLD and < HIGH_THRESHOLD → "Medium"
# Anything below → "Low"
SEVERITY_HIGH_THRESHOLD:   float = 5000.0
SEVERITY_MEDIUM_THRESHOLD: float = 1500.0
