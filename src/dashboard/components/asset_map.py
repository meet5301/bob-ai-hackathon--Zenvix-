"""
asset_map.py
------------
Renders a Plotly scatter-mapbox of all assets colour-coded by risk level.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.config import (
    RISK_COLORS,
    SEVERITY_HIGH_THRESHOLD,
    SEVERITY_MEDIUM_THRESHOLD,
)


def _assign_risk_level(severity_score: float) -> str:
    if severity_score >= SEVERITY_HIGH_THRESHOLD:
        return "High"
    if severity_score >= SEVERITY_MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


def render_asset_map(rankings_df: pd.DataFrame) -> None:
    """
    Display a scatter map of grid assets colour-coded by risk level.

    Expects columns: asset_id, lat, lon, severity_score, region,
                     asset_type, failure_probability.
    """
    required = ["asset_id", "lat", "lon", "severity_score",
                 "region", "asset_type", "failure_probability"]
    missing = [c for c in required if c not in rankings_df.columns]
    if missing:
        st.error(f"Asset map cannot render — missing columns: {missing}")
        return

    df = rankings_df.copy()
    df["risk_level"] = df["severity_score"].apply(_assign_risk_level)
    df["color"]      = df["risk_level"].map(RISK_COLORS)

    st.subheader("🗺️ Asset Risk Map")

    traces = []
    for level in ["High", "Medium", "Low"]:
        subset = df[df["risk_level"] == level]
        if subset.empty:
            continue
        traces.append(
            go.Scattermapbox(
                lat=subset["lat"],
                lon=subset["lon"],
                mode="markers",
                marker=go.scattermapbox.Marker(
                    size=10,
                    color=RISK_COLORS[level],
                    opacity=0.85,
                ),
                text=subset.apply(
                    lambda r: (
                        f"<b>{r['asset_id']}</b><br>"
                        f"Type: {r['asset_type']}<br>"
                        f"Region: {r['region']}<br>"
                        f"Failure prob: {r['failure_probability']:.1%}<br>"
                        f"Severity: {r['severity_score']:.0f}"
                    ),
                    axis=1,
                ),
                hoverinfo="text",
                name=f"{level} Risk",
            )
        )

    center_lat = float(df["lat"].mean())
    center_lon = float(df["lon"].mean())

    layout = go.Layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=6,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=480,
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#e5e7eb",
            borderwidth=1,
        ),
    )

    fig = go.Figure(data=traces, layout=layout)
    st.plotly_chart(fig, use_container_width=True)

    # Legend caption
    legend_md = " · ".join(
        f"<span style='color:{RISK_COLORS[lvl]};font-weight:600'>{lvl}</span>"
        for lvl in ["High", "Medium", "Low"]
    )
    st.caption(
        f"Colour coding — {legend_md}  "
        f"(High ≥ {SEVERITY_HIGH_THRESHOLD:,.0f}, "
        f"Medium ≥ {SEVERITY_MEDIUM_THRESHOLD:,.0f}, "
        "Low < threshold)",
        unsafe_allow_html=True,
    )
