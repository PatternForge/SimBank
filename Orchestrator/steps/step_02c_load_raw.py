import os
import snowflake.connector
import pandas as pd
from pathlib import Path


def ensure_table(conn, table_name, csv_path):
    conn.cursor().execute(f"DROP TABLE IF EXISTS RAW.{table_name}")
    df = pd.read_csv(csv_path, nrows=1)
    cols = ", ".join([f"{c} VARCHAR" for c in df.columns])
    conn.cursor().execute(f"CREATE TABLE RAW.{table_name} ({cols})")


def load_table(conn, table_name, csv_path):
    conn.cursor().execute(f"DELETE FROM RAW.{table_name}")
    conn.cursor().execute(f"PUT file://{csv_path} @%{table_name}")
    conn.cursor().execute(f"""
        COPY INTO RAW.{table_name}
        FROM @%{table_name}
        FILE_FORMAT = (
            TYPE = CSV
            FIELD_OPTIONALLY_ENCLOSED_BY='"'
            SKIP_HEADER=1
        )
    """)


def run(state):
    REPO_ROOT = Path(__file__).resolve().parents[2]
    sources_dir = REPO_ROOT / "SimBank" / "sources"

    conn = snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema="RAW",
        role=os.getenv("SNOWFLAKE_ROLE"),
    )

    tables = [
        "RETAIL_LOANS",
        "RETAIL_DEPOSITS",
        "BUSINESS_LOANS",
        "BUSINESS_DEPOSITS",
        "SIMULATED_PARAMETERS",
        "FEES_COSTS",
        "FTP_INPUTS",
        "STRESS_INPUTS",
    ]

    for t in tables:
        csv_path = sources_dir / f"{t.lower()}.csv"
        ensure_table(conn, t, csv_path)
        load_table(conn, t, csv_path)

    conn.close()
    state.mark("load_raw", "success")
