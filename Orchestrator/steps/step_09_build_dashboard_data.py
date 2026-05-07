import os
import sys
import snowflake.connector

from Orchestrator.governance.graph_engine import build_graph, downstream_fields

print("[STEP 09] Loaded Python:", sys.executable)


def run(state):

    print("\n" + "=" * 60)
    print("[STEP 09] Build Dashboard Data")
    print("=" * 60)

    raw = state.metadata.get("lineage", [])
    print(f"[STEP 09] Raw lineage rows: {len(raw)}")

    if not raw:
        state.mark("governance_dashboard_data", "skipped", "No lineage metadata")
        return

    lineage = []

    for r in raw:
        src_cte = r.get("SOURCE_CTE")
        src_col = r.get("SOURCE_FIELD")
        tgt_cte = r.get("CTE_NAME")
        tgt_col = r.get("FIELD_NAME")

        from_field = r.get("from_field")
        to_field = r.get("to_field")

        if from_field and to_field:
            lineage.append({"from_field": from_field, "to_field": to_field})
            continue

        if src_cte and src_col and tgt_cte and tgt_col:
            lineage.append({
                "from_field": f"{src_cte}.{src_col}",
                "to_field": f"{tgt_cte}.{tgt_col}",
            })

    print(f"[STEP 09] Normalised edges: {len(lineage)}")

    if not lineage:
        state.mark("governance_dashboard_data", "skipped", "No usable lineage")
        return

    g = build_graph(lineage)

    print(f"[STEP 09] Graph nodes: {len(g.nodes())}")
    print(f"[STEP 09] Graph edges: {len(g.edges())}")

    if len(g.nodes()) == 0:
        state.mark("governance_dashboard_data", "skipped", "Empty graph")
        return

    records = []

    for node in g.nodes():
        downstream = downstream_fields(g, node)

        records.append({
            "field_fqn": node,
            "downstream_fields": len(downstream),
            "downstream_ctes": 0,
            "downstream_models": 0,
            "downstream_domains": 0,
            "blast_radius_score": float(len(downstream)),
        })

    print(f"[STEP 09] Records to insert: {len(records)}")

    if not records:
        state.mark("governance_dashboard_data", "skipped", "No graph nodes")
        return

    db = os.getenv("SNOWFLAKE_DATABASE")
    gov = os.getenv("SNOWFLAKE_GOVT")

    print(f"[STEP 09] Target table: {db}.{gov}.FIELD_BLAST_RADIUS")

    conn = snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        role=os.getenv("SNOWFLAKE_ROLE"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        login_timeout=20,
    )

    cur = conn.cursor()

    run_id = state.run_id

    delete_sql = f"""
        DELETE FROM {db}.{gov}.FIELD_BLAST_RADIUS
        WHERE run_id = %s
    """

    print(f"[STEP 09] Deleting existing rows for run_id={run_id}")
    cur.execute(delete_sql, (run_id,))
    print(f"[STEP 09] Deleted rows: {cur.rowcount}")

    insert_sql = f"""
        INSERT INTO {db}.{gov}.FIELD_BLAST_RADIUS (
            run_id,
            field_fqn,
            downstream_fields,
            downstream_ctes,
            downstream_models,
            downstream_domains,
            blast_radius_score
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    rows = [
        (
            run_id,
            r["field_fqn"],
            r["downstream_fields"],
            r["downstream_ctes"],
            r["downstream_models"],
            r["downstream_domains"],
            r["blast_radius_score"],
        )
        for r in records
    ]

    print("[STEP 09] Bulk inserting records...")

    cur.executemany(insert_sql, rows)

    conn.commit()

    print(f"[STEP 09] Inserted rows: {len(rows)}")

    cur.close()
    conn.close()

    state.mark("governance_dashboard_data", "success")

    print("[STEP 09] DONE")