"""
maintenance_plan.py
-------------------
Renders the watsonx.ai-generated maintenance plan with appropriate
loading and error states.
"""

import streamlit as st


def render_maintenance_plan(plan_text: str) -> None:
    """
    Display the maintenance plan in a readable, formatted block.

    Args:
        plan_text: The plain-English plan string returned by generate_plan.py,
                   or an empty string if it has not yet been fetched.
    """
    st.subheader("🛠️ Maintenance & Crew Pre-Positioning Plan")

    if not plan_text:
        # This branch is only reached while the plan is still loading;
        # the caller should pass plan_text="" and control the spinner externally.
        st.info("Plan not yet generated. Click **Generate Maintenance Plan** above.")
        return

    st.markdown(plan_text)


def render_plan_error(error_message: str) -> None:
    """
    Display a clear error state when plan generation fails.
    Never shows a fake or placeholder plan — the user must see the real error.
    """
    st.error(
        f"**Maintenance plan generation failed.**\n\n"
        f"{error_message}\n\n"
        "Check that `WATSONX_API_KEY`, `WATSONX_PROJECT_ID`, and `WATSONX_URL` "
        "are correctly set in your `.env` file and that the watsonx.ai service "
        "is reachable."
    )
