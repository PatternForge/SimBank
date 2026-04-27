import streamlit as st
from Dashboard.data.queries import get_system_metrics

st.header("System Health")

metrics = get_system_metrics()

col1, col2, col3 = st.columns(3)

col1.metric("Health Fields (%)", metrics["healthy_pct"])
col2.metric("Broken Fields", metrics["broken_fields"])
col3.metric("Open Drift Events", metrics["open_drifts"])
