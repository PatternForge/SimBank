import streamlit as st
from graphviz import Digraph


def render_lineage_graph(root, edges, direction='forward'):
    dot = Digraph()

    dot.node(root, root)

    for e in edges:
        src = e["from_field"]
        tgt = e["to_field"]
        dot.node(src, src)
        dot.node(tgt, tgt)
        dot.edge(src, tgt)

    st.graphviz_chart(dot, use_container_width=True)
