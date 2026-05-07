import streamlit as st
from Dashboard.data.queries import (
    get_all_fields,
    get_forward_lineage,
    get_field_summary,
    get_blast_radius,
)
from Dashboard.ui.graphs import render_lineage_graph
from Dashboard.ui.layout import section

st.header("Impact Analysis")

field = st.selectbox(
    "Select a field",
    options=get_all_fields()
)

if field:
    lineage = get_forward_lineage(field)
    summary = get_field_summary(field)
    blast = get_blast_radius(field)

    col1, col2 = st.columns([3, 1])

    with col1:
        section("Downstream Impact")
        render_lineage_graph(field, lineage)

    with col2:
        section("Summary")
        for k, v in summary.items():
            st.write(f"**{k}:** {v}")

        section("Blast Radius")
        if blast:
            st.metric("Blast Radius Score", blast.get("blast_radius_score", 0))
            st.write(f"Downstream Fields: {blast.get('downstream_fields', 0)}")
            st.write(f"Downstream CTEs: {blast.get('downstream_ctes', 0)}")
            st.write(f"Downstream Models: {blast.get('downstream_models', 0)}")
            st.write(f"Downstream Domains: {blast.get('downstream_domains', 0)}")
        else:
            st.write("No blast radius data available for this field.")
