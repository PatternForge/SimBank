import streamlit as st
from Dashboard.data.queries import get_system_metrics, get_blast_radius_summary

st.header("System Health")

metrics = get_system_metrics()
blast_summary = get_blast_radius_summary()

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Healthy Fields (%)", metrics["healthy_pct"])
col2.metric("Broken Fields", metrics["broken_fields"])
col3.metric("Open Drift Events", metrics["open_drifts"])
col4.metric("Avg Blast Radius", blast_summary.get("avg_score", 0))
col5.metric("Max Blast Radius", blast_summary.get("max_score", 0))
