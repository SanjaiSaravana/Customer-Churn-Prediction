# app/app.py
# ─────────────────────────────────────────────────────────────────────────────
# Main Streamlit entry point — multi-page navigation.
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

st.set_page_config(
    page_title = "Churn Prediction Platform",
    page_icon  = "📊",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.image(
    "https://img.icons8.com/color/96/combo-chart.png", width=80
)
st.sidebar.title("📊 Churn Platform")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to",
    [
        "🏠 Executive Dashboard",
        "👥 Customer Analytics",
        "🔮 Churn Prediction",
        "⚠️  Risk Analysis",
        "📈 Model Performance",
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Telco Customer Churn · v1.0")

# ── Route to page ─────────────────────────────────────────────────────────────
if page == "🏠 Executive Dashboard":
    from pages.executive_dashboard import show
    show()
elif page == "👥 Customer Analytics":
    from pages.customer_analytics import show
    show()
elif page == "🔮 Churn Prediction":
    from pages.churn_prediction import show
    show()
elif page == "⚠️  Risk Analysis":
    from pages.risk_analysis import show
    show()
elif page == "📈 Model Performance":
    from pages.model_performance import show
    show()