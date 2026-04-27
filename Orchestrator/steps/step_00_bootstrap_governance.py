import os
import snowflake.connector


def run(state):
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

    cur.execute(f"USE ROLE {os.getenv('SNOWFLAKE_ROLE')}")
    cur.execute(f"USE WAREHOUSE {os.getenv('SNOWFLAKE_WAREHOUSE')}")
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
        create table if not exists {db}.{gov}.FIELD_LINEAGE (
            MODEL_NAME string,
            CTE_NAME string,
            FIELD_NAME string,
            SOURCE_CTE string,
            SOURCE_FIELD string
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

    cur.close()
    conn.close()

    state.mark("governance_bootstrap", "success")
