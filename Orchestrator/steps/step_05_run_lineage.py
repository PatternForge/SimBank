import subprocess
import sys
import snowflake.connector
from pathlib import Path
import os


def run(state):
    REPO_ROOT = Path(__file__).resolve().parents[2]
    LINEAGE_SCRIPT = REPO_ROOT / "simbank_dbt" / "lineage" / "run_lineage.py"

    if not LINEAGE_SCRIPT.exists():
        raise FileNotFoundError(f"Lineage script not found: {LINEAGE_SCRIPT}")

    subprocess.run(
        [sys.executable, str(LINEAGE_SCRIPT), "--snowflake"],
        check=True,
    )

    # 2. Connect to Snowflake
    conn = snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "SIMBANK_WH"),
        database="SIMBANK",
        schema="RAW",
        role=os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN"),
    )
    cur = conn.cursor()

    cur.execute("""
        SELECT 
            MODEL_NAME,
            CTE_NAME,
            FIELD_NAME,
            SOURCE_CTE,
            SOURCE_FIELD
        FROM SIMBANK.RAW.FIELD_LINEAGE
    """)

    rows = cur.fetchall()
    cols = [c[0] for c in cur.description]
    lineage_rows = [dict(zip(cols, row)) for row in rows]

    cur.close()
    conn.close()

    state.metadata["lineage"] = lineage_rows
    state.mark("lineage", "success")
