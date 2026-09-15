"""
risk_table.py
-------------
Renders a filterable, sortable table of assets ranked by severity_score.
"""

import pandas as pd
import streamlit as st

from dashboard.config import RISK_COLORS, SEVERITY_HIGH_THRESHOLD, SEVERITY_MEDIUM_THRESHOLD


def _risk_badge(severity_score: float) -> str:
    if severity_score >= SEVERITY_HIGH_THRESHOLD:
        color = RISK_COLORS["High"]
        label = "High"
    elif severity_score >= SEVERITY_MEDIUM_THRESHOLD:
        color = RISK_COLORS["Medium"]
        label = "Medium"
    else:
        color = RISK_COLORS["Low"]
        label = "Low"
    return f"<span style='color:{color};font-weight:600'>{label}</span>"


def render_risk_table(rankings_df: pd.DataFrame) -> None:
    """
    Display a sortable, filterable risk table.

    Expects columns: asset_id, region, failure_probability, severity_score,
                     criticality_tier, customers_served.
    """
    required = [
        "asset_id", "region", "failure_probability",
        "severity_score", "criticality_tier", "customers_served",
    ]
    missing = [c for c in required if c not in rankings_df.columns]
    if missing:
        st.error(f"Risk table cannot render — missing columns: {missing}")
        return

    st.subheader("📋 Asset Risk Rankings")

    # ── Filters ────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        all_regions = sorted(rankings_df["region"].unique().tolist())
        selected_regions = st.multiselect(
            "Filter by Region",
            options=all_regions,
            default=all_regions,
        )

    with col2:
        all_tiers = sorted(rankings_df["criticality_tier"].unique().tolist())
        selected_tiers = st.multiselect(
            "Filter by Criticality Tier",
            options=all_tiers,
            default=all_tiers,
            format_func=lambda t: f"Tier {t}",
        )

    filtered = rankings_df[
        rankings_df["region"].isin(selected_regions)
        & rankings_df["criticality_tier"].isin(selected_tiers)
    ].copy()

    if filtered.empty:
        st.info("No assets match the selected filters.")
        return

    # ── Display columns ───────────────────────────────────────────────────
    display_df = filtered[required].copy()
    display_df["failure_probability"] = (
        display_df["failure_probability"].map("{:.1%}".format)
    )
    display_df["severity_score"] = display_df["severity_score"].map("{:.1f}".format)
    display_df["criticality_tier"] = display_df["criticality_tier"].apply(
        lambda t: f"Tier {t}"
    )

    display_df = display_df.rename(
        columns={
            "asset_id":           "Asset ID",
            "region":             "Region",
            "failure_probability":"Failure Prob.",
            "severity_score":     "Severity Score",
            "criticality_tier":   "Criticality",
            "customers_served":   "Customers Served",
        }
    )

    st.write(
        f"Showing **{len(filtered)}** of **{len(rankings_df)}** assets"
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)
