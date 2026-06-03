import os
import snowflake.connector

def run(state):
    print("\n============================================================")
    print("Step 05a: Build FIELD_CATALOG")
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

    print("Clearing existing FIELD_CATALOG...")
    cur.execute(f"TRUNCATE TABLE {db}.{gov}.FIELD_CATALOG")

    print("Populating FIELD_CATALOG from FIELD_LINEAGE...")
    cur.execute(f"""
        INSERT INTO {db}.{gov}.FIELD_CATALOG (
            field_fqn,
            model_name,
            cte_name,
            field_name,
            source_cte,
            source_field
        )
        SELECT
            CONCAT(cte_name, '.', field_name) AS field_fqn,
            model_name,
            cte_name,
            field_name,
            source_cte,
            source_field
        FROM {db}.{gov}.FIELD_LINEAGE
    """)

    print(f"Inserted rows: {cur.rowcount}")

    conn.commit()
    cur.close()
    conn.close()

    state.mark("field_catalog", "success")
