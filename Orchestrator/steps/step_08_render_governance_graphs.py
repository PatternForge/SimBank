import os
from pathlib import Path

from Orchestrator.governance.graph_engine import (
    build_graph,
    downstream_fields,
    upstream_fields,
)

from Orchestrator.governance.graph_render import (
    render_graph,
    save_svg,
)

OUTPUT_DIR = "artifacts/governance"


def sanitize_filename(value):
    """
    Make lineage field names safe for filesystem usage.
    """
    return (
        value.replace(".", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def run(state):

    print("\n" + "=" * 60)
    print("[STEP 08] Governance Graph Rendering")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    raw = state.metadata.get("lineage", [])

    print(f"[STEP 08] Raw lineage rows found: {len(raw)}")

    lineage = []

    for r in raw:

        src_cte = r.get("SOURCE_CTE")
        src_col = r.get("SOURCE_FIELD")

        tgt_cte = r.get("CTE_NAME")
        tgt_col = r.get("FIELD_NAME")

        # Skip incomplete lineage rows
        if not src_cte or not src_col or not tgt_cte or not tgt_col:
            continue

        lineage.append({
            "from_field": f"{src_cte}.{src_col}",
            "to_field": f"{tgt_cte}.{tgt_col}",
        })

    print(f"[STEP 08] Valid lineage edges: {len(lineage)}")

    # Prevent graph engine crash on empty lineage
    if not lineage:
        print("[STEP 08] No lineage data available")
        state.mark("governance_visuals", "skipped")
        return

    print("[STEP 08] Building graph")

    g = build_graph(lineage)

    print(f"[STEP 08] Graph node count: {len(g.nodes())}")

    # Calculate downstream impact score
    impact_scores = {
        node: len(downstream_fields(g, node))
        for node in g.nodes()
    }

    core_fields = sorted(
        impact_scores,
        key=lambda x: impact_scores[x],
        reverse=True
    )[:10]

    print(f"[STEP 08] Core fields identified: {len(core_fields)}")

    state.metadata["core_fields"] = core_fields

    broken_fields = state.metadata.get("broken_fields", []) or []

    print(f"[STEP 08] Broken fields identified: {len(broken_fields)}")

    # ------------------------------------------------------------------
    # Render downstream impact graphs
    # ------------------------------------------------------------------

    for field in core_fields:

        print(f"[STEP 08] Rendering impact graph: {field}")

        downstream = downstream_fields(g, field)

        edges = [(field, d) for d in downstream]

        dot = render_graph(field, edges)

        safe_field = sanitize_filename(field)

        output_path = f"{OUTPUT_DIR}/{safe_field}_impact.svg"

        save_svg(dot, output_path)

        print(f"[STEP 08] Saved: {output_path}")

    # ------------------------------------------------------------------
    # Render upstream root-cause graphs
    # ------------------------------------------------------------------

    for broken in broken_fields:

        print(f"[STEP 08] Rendering root cause graph: {broken}")

        upstream = upstream_fields(g, broken)

        edges = [(u, broken) for u in upstream]

        dot = render_graph(
            broken,
            edges,
            direction="upstream"
        )

        safe_broken = sanitize_filename(broken)

        output_path = f"{OUTPUT_DIR}/{safe_broken}_root_cause.svg"

        save_svg(dot, output_path)

        print(f"[STEP 08] Saved: {output_path}")

    print("\n[STEP 08] Governance graph rendering complete")

    state.mark("governance_visuals", "success")
    return