import os
from pathlib import Path
from dotenv import load_dotenv
import snowflake.connector


def run(state):
    REPO_ROOT = Path(__file__).resolve().parents[2]
    load_dotenv(REPO_ROOT / ".env")

    conn = snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        role=os.getenv("SNOWFLAKE_ROLE"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
    )

    db   = os.getenv("SNOWFLAKE_DATABASE")
    raw  = os.getenv("SNOWFLAKE_RAW")
    stg  = os.getenv("SNOWFLAKE_STG")
    mart = os.getenv("SNOWFLAKE_MART")
    gov  = os.getenv("SNOWFLAKE_GOVT")

    cur = conn.cursor()

    cur.execute(f"USE ROLE {os.getenv('SNOWFLAKE_ROLE')}")
    cur.execute(f"USE WAREHOUSE {os.getenv('SNOWFLAKE_WAREHOUSE')}")
    cur.execute(f"USE DATABASE {db}")

    ddls = [
        f"create database if not exists {db}",
        f"create schema if not exists {db}.{raw}",
        f"create schema if not exists {db}.{stg}",
        f"create schema if not exists {db}.{mart}",
        f"create schema if not exists {db}.{gov}",
    ]

    for ddl in ddls:
        cur.execute(ddl)

    cur.close()
    conn.close()

    state.mark("snowflake_env", "success")
