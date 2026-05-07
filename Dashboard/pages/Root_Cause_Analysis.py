import streamlit as st
from Dashboard.data.queries import (
    get_broken_fields,
    get_failure_context,
    get_backward_lineage,
    get_blast_radius,
)
from Dashboard.ui.graphs import render_lineage_graph
from Dashboard.ui.layout import section

st.header("Root Cause Analysis")

field = st.selectbox(
    "Select a broken field",
    options=get_broken_fields()
)

if field:
    context = get_failure_context(field)
    lineage = get_backward_lineage(field)
    blast = get_blast_radius(field)

    col1, col2 = st.columns([3, 1])

    with col1:
        section("Failure Summary")
        for k, v in context.items():
            st.write(f"**{k}:** {v}")

        section("Upstream Lineage")
        render_lineage_graph(field, lineage, direction="backward")

    with col2:
        section("Blast Radius")
        if blast:
            st.metric("Blast Radius Score", blast.get("blast_radius_score", 0))
            st.write(f"Downstream Fields: {blast.get('downstream_fields', 0)}")
            st.write(f"Downstream CTEs: {blast.get('downstream_ctes', 0)}")
            st.write(f"Downstream Models: {blast.get('downstream_models', 0)}")
            st.write(f"Downstream Domains: {blast.get('downstream_domains', 0)}")
        else:
            st.write("No blast radius data available for this field.")
