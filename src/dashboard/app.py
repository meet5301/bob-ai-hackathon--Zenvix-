"""
app.py
------
Main entry point for the Grid Guard Streamlit dashboard.
"""

import streamlit as st

from dashboard.components.asset_map import render_asset_map
from dashboard.components.maintenance_plan import render_maintenance_plan, render_plan_error
from dashboard.components.risk_table import render_risk_table
from dashboard.utils.api_client import get_plan, get_rankings

st.set_page_config(
    page_title="Grid Guard — Power Outage Prediction & Grid Equipment Failure Advisor",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Grid Guard")
st.caption("Power Outage Prediction & Grid Equipment Failure Advisor — Team Maverick")
st.divider()


# ── Load rankings ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner="Loading risk rankings …")
def _fetch_rankings():
    return get_rankings()


rankings_error: str | None = None
rankings_df = None

try:
    rankings_df = _fetch_rankings()
except RuntimeError as exc:
    rankings_error = str(exc)


# ── Asset map ─────────────────────────────────────────────────────────────────
if rankings_error:
    st.error(
        f"**Failed to load asset rankings from the API.**\n\n{rankings_error}\n\n"
        "Make sure the FastAPI service is running and `API_BASE_URL` is correct."
    )
else:
    if rankings_df is None or rankings_df.empty:
        st.warning("No ranked assets returned from the API.")
    else:
        render_asset_map(rankings_df)
        st.divider()
        render_risk_table(rankings_df)
        st.divider()

        # ── Maintenance plan ───────────────────────────────────────────────
        st.subheader("🛠️ Maintenance & Crew Pre-Positioning Plan")

        if st.button("Generate Maintenance Plan", type="primary"):
            with st.spinner("Calling IBM watsonx.ai (Bob) …"):
                try:
                    plan_text = get_plan()
                    render_maintenance_plan(plan_text)
                except RuntimeError as plan_exc:
                    render_plan_error(str(plan_exc))
        else:
            st.info(
                "Click **Generate Maintenance Plan** to call IBM watsonx.ai (Bob) "
                "and produce a prioritised maintenance and crew pre-positioning plan "
                "for the top 10 at-risk assets."
            )
