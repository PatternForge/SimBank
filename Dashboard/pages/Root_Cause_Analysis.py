import streamlit as st
from Dashboard.data.queries import (
    get_broken_fields,
    get_failure_context,
    get_backward_lineage,
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

    col1, col2 = st.columns([3, 1])

    section("Failure Summary")
    for k, v in context.items():
        st.write(f"**{k}:** {v}")

    section("Upstream Lineage")
    render_lineage_graph(field, lineage, direction="backward")
