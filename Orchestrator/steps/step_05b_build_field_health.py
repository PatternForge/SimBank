import os
import snowflake.connector

def run(state):
    print("\n============================================================")
    print("Step 05b: Build FIELD_HEALTH")
    print("============================================================")

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

    print("Clearing existing FIELD_HEALTH...")
    cur.execute(f"TRUNCATE TABLE {db}.{gov}.FIELD_HEALTH")

    print("Populating FIELD_HEALTH from FIELD_CATALOG...")
    cur.execute(f"""
        INSERT INTO {db}.{gov}.FIELD_HEALTH (
        field_fqn,
        status,
        reason,
        is_healthy
    )
        SELECT
        field_fqn,
        CASE WHEN source_cte IS NULL AND source_field IS NULL AND cte_name NOT LIKE 'RAW_%' THEN 'broken' ELSE 'healthy'
        END AS status,
        CASE WHEN source_cte IS NULL AND source_field IS NULL AND cte_name NOT LIKE 'RAW_%' 
        THEN 'No valid upstream lineage' ELSE NULL END AS reason,
        CASE WHEN source_cte IS NULL AND source_field IS NULL AND cte_name NOT LIKE 'RAW_%' THEN FALSE ELSE TRUE
        END AS is_healthy
    FROM {db}.{gov}.FIELD_CATALOG
    """)

    print(f"Inserted rows: {cur.rowcount}")

    conn.commit()
    cur.close()
    conn.close()

    state.mark("field_health", "success")
