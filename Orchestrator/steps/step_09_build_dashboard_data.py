import os
import json
from pathlib import Path
from dotenv import load_dotenv
import snowflake.connector


def run(state):
    REPO_ROOT = Path(__file__).resolve().parents[2]
    load_dotenv(REPO_ROOT / ".env")

    db = os.getenv("SNOWFLAKE_DATABASE")
    gov = os.getenv("SNOWFLAKE_GOVT")

    conn = snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        role=os.getenv("SNOWFLAKE_ROLE"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
    )
    cur = conn.cursor()

    # Ensure deterministic session
    cur.execute(f"USE DATABASE {db}")
    cur.execute(f"USE SCHEMA {db}.{gov}")

    cur.execute(f"""
        create table if not exists {db}.{gov}.FIELD_CATALOG (
            field_fqn string,
            model_name string,
            cte_name string,
            field_name string,
            source_cte string,
            source_field string
        )
    """)

    cur.execute(f"""
        create table if not exists {db}.{gov}.FIELD_HEALTH (
            field_fqn string,
            status string,
            reason string,
            is_healthy boolean
        )
    """)

    cur.execute(f"""
        create table if not exists {db}.{gov}.FIELD_CHANGE_DIFF (
            field_fqn string,
            change_type string,
            detail string
        )
    """)

    lineage_rows = state.metadata.get("lineage", [])

    catalog_rows = []
    for r in lineage_rows:
        fqn = f"{r['CTE_NAME']}.{r['FIELD_NAME']}"
        catalog_rows.append((
            fqn,
            r["MODEL_NAME"],
            r["CTE_NAME"],
            r["FIELD_NAME"],
            r["SOURCE_CTE"],
            r["SOURCE_FIELD"],
        ))

    cur.execute(f"truncate table {db}.{gov}.FIELD_CATALOG")
    cur.executemany(f"""
        insert into {db}.{gov}.FIELD_CATALOG (
            field_fqn, model_name, cte_name, field_name, source_cte, source_field
        ) values (%s, %s, %s, %s, %s, %s)
    """, catalog_rows)

    broken = set(state.metadata.get("broken_fields", []))

    health_rows = []
    for r in catalog_rows:
        fqn = r[0]
        if fqn in broken:
            health_rows.append((fqn, "BROKEN", "Detected in drift", False))
        else:
            health_rows.append((fqn, "OK", None, True))

    cur.execute(f"truncate table {db}.{gov}.FIELD_HEALTH")
    cur.executemany(f"""
        insert into {db}.{gov}.FIELD_HEALTH (
            field_fqn, status, reason, is_healthy
        ) values (%s, %s, %s, %s)
    """, health_rows)

    drift_dir = REPO_ROOT / "SimBank" / "Output"
    drift_files = list(drift_dir.glob("drift_results_*.json"))

    diff_rows = []
    if drift_files:
        latest = max(drift_files, key=lambda p: p.stat().st_mtime)
        with open(latest, "r") as f:
            drift = json.load(f)

        for table, info in drift.get("column_changes", {}).items():
            for col, detail in info.items():
                fqn = f"{table}.{col}"
                diff_rows.append((fqn, detail.get("type"), json.dumps(detail)))

    cur.execute(f"truncate table {db}.{gov}.FIELD_CHANGE_DIFF")
    if diff_rows:
        cur.executemany(f"""
            insert into {db}.{gov}.FIELD_CHANGE_DIFF (
                field_fqn, change_type, detail
            ) values (%s, %s, %s)
        """, diff_rows)

    cur.close()
    conn.close()

    state.mark("dashboard", "success")
