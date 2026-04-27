import os
from pathlib import Path
from Orchestrator.governance.graph_engine import (
    build_graph,
    downstream_fields,
    upstream_fields,
)
from Orchestrator.governance.graph_render import render_graph


OUTPUT_DIR = "artifacts/governance"


def run(state):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    raw = state.metadata.get("lineage", [])

    lineage = []
    for r in raw:
        src_cte = r.get("SOURCE_CTE")
        src_col = r.get("SOURCE_FIELD")
        tgt_cte = r.get("CTE_NAME")
        tgt_col = r.get("FIELD_NAME")

        if not src_cte or not src_col:
            continue

        lineage.append({
            "from_field": f"{src_cte}.{src_col}",
            "to_field": f"{tgt_cte}.{tgt_col}",
        })

    g = build_graph(lineage)
    core_fields = state.metadata.get("core_fields", [])

    for field in core_fields:
        downstream = downstream_fields(g, field)
        edges = [(field, d) for d in downstream]

        dot = render_graph(field, edges)
        dot.render(f"{OUTPUT_DIR}/{field}_impact", cleanup=True)

    broken_fields = state.metadata.get("broken_fields", [])

    for broken in broken_fields:
        upstream = upstream_fields(g, broken)
        edges = [(u, broken) for u in upstream]

        dot = render_graph(broken, edges, direction="upstream")
        dot.render(f"{OUTPUT_DIR}/{broken}_root_cause", cleanup=True)

    state.mark("governance_visuals", "success")
