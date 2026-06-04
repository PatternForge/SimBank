import os
import snowflake.connector
from dotenv import load_dotenv
from pathlib import Path


def run(state):
    REPO_ROOT = Path(__file__).resolve().parents[2]
    load_dotenv(REPO_ROOT / ".env")

    db = os.getenv("SNOWFLAKE_DATABASE")
    raw = os.getenv("SNOWFLAKE_RAW")
    stg = os.getenv("SNOWFLAKE_STG")
    mart = os.getenv("SNOWFLAKE_MART")
    gov = os.getenv("SNOWFLAKE_GOVT")
    warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
    role = os.getenv("SNOWFLAKE_ROLE")

    conn = snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        role=role,
    )

    cur = conn.cursor()

    cur.execute(f"""
        CREATE WAREHOUSE IF NOT EXISTS {warehouse}
            WAREHOUSE_SIZE = 'XSMALL'
            AUTO_SUSPEND = 60
            AUTO_RESUME = TRUE
            INITIALLY_SUSPENDED = TRUE
    """)

    cur.execute(f"CREATE DATABASE IF NOT EXISTS {db}")
    cur.execute(f"CREATE SCHEMA IF NOT EXISTS {db}.{raw}")
    cur.execute(f"CREATE SCHEMA IF NOT EXISTS {db}.{stg}")
    cur.execute(f"CREATE SCHEMA IF NOT EXISTS {db}.{mart}")
    cur.execute(f"CREATE SCHEMA IF NOT EXISTS {db}.{gov}")

    cur.execute(f"USE ROLE {role}")
    cur.execute(f"USE WAREHOUSE {warehouse}")
    cur.execute(f"USE DATABASE {db}")
    cur.execute(f"USE SCHEMA {db}.{gov}")

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {db}.{gov}.FIELD_CATALOG (
            field_fqn STRING,
            model_name STRING,
            cte_name STRING,
            field_name STRING,
            source_cte STRING,
            source_field STRING
        )
    """)

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {db}.{gov}.FIELD_LINEAGE (
            model_name STRING,
            cte_name STRING,
            field_name STRING,
            source_cte STRING,
            source_field STRING
        )
    """)

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {db}.{gov}.FIELD_HEALTH (
            field_fqn STRING,
            status STRING,
            reason STRING,
            is_healthy BOOLEAN
        )
    """)

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {db}.{gov}.FIELD_CHANGE_DIFF (
            field_fqn STRING,
            change_type STRING,
            detail STRING
        )
    """)

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {db}.{gov}.FIELD_BLAST_RADIUS (
            run_id STRING,
            field_fqn STRING,
            downstream_fields INT,
            downstream_ctes INT,
            downstream_models INT,
            downstream_domains INT,
            blast_radius_score FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
        )
    """)

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {db}.{gov}.DATA_ATTESTATIONS (
            RUN_ID STRING,
            MANIFEST_HASH STRING,       
            STATUS STRING,
            APPROVED_BY STRING,
            APPROVED_AT TIMESTAMP
        )
    """)

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {db}.{gov}.CODE_BASELINES (
            RUN_ID STRING,
            FILE_PATH STRING,
            FILE_HASH STRING,
            FILE_CONTENT STRING,        
            STATUS STRING,
            APPROVED_BY STRING,
            APPROVED_AT TIMESTAMP
        )
    """)

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {db}.{gov}.DOCS_BASELINES (
            RUN_ID STRING,
            DOCS_HASH STRING,
            ARTIFACT_ZIP BINARY,
            STATUS STRING,
            APPROVED_BY STRING,
            APPROVED_AT TIMESTAMP
        )
    """)

    cur.close()
    conn.close()

    if state is not None:
        state.mark("governance_bootstrap", "success")
