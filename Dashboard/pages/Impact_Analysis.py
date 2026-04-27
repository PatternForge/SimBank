import streamlit as st
from Dashboard.data.queries import (
    get_all_fields,
    get_forward_lineage,
    get_field_summary,
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

    col1, col2 = st.columns([3, 1])

    with col1:
        section("Downstream Impact")
        render_lineage_graph(field, lineage)

    with col2:
        section("Summary")
        for k, v in summary.items():
            st.write(f"**{k}:** {v}")

