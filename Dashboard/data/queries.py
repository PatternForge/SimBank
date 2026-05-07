import pandas as pd
from Dashboard.data.connection import get_connection
from Dashboard.config import SNOWFLAKE_SCHEMA, SNOWFLAKE_DATABASE

GOV = f"{SNOWFLAKE_DATABASE}.{SNOWFLAKE_SCHEMA}"


def fetch_df(sql):
    conn = get_connection()
    df = pd.read_sql(sql, conn)
    conn.close()
    df.columns = [c.lower() for c in df.columns]
    return df


def get_all_fields():
    df = fetch_df(f"select field_fqn from {GOV}.FIELD_CATALOG order by field_fqn")
    return df["field_fqn"].tolist()


def get_forward_lineage(field):
    sql = f"""
        select
            source_field as from_field,
            field_name as to_field,
            cte_name as layer
        from {GOV}.FIELD_LINEAGE
        where field_name = '{field}'
    """
    return fetch_df(sql).to_dict("records")


def get_backward_lineage(field):
    sql = f"""
        select
            field_name as from_field,
            source_field as to_field,
            cte_name as layer
        from {GOV}.FIELD_LINEAGE
        where source_field = '{field}'
    """
    return fetch_df(sql).to_dict("records")


def get_field_summary(field):
    sql = f"""
        select *
        from {GOV}.FIELD_CATALOG
        where field_fqn = '{field}'
    """
    df = fetch_df(sql)
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def get_broken_fields():
    df = fetch_df(f"""
        select field_fqn
        from {GOV}.FIELD_HEALTH
        where status = 'BROKEN'
        order by field_fqn
    """)
    return df["field_fqn"].tolist()


def get_failure_context(field):
    sql = f"""
        select *
        from {GOV}.FIELD_HEALTH
        where field_fqn = '{field}'
    """
    df = fetch_df(sql)
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def get_system_metrics():
    return {
        "healthy_pct": fetch_df(
            f"select avg(is_healthy::int) * 100 as pct from {GOV}.FIELD_HEALTH"
        ).iloc[0, 0],
        "broken_fields": fetch_df(
            f"select count(*) as cnt from {GOV}.FIELD_HEALTH where status='BROKEN'"
        ).iloc[0, 0],
        "open_drifts": fetch_df(
            f"select count(*) as cnt from {GOV}.FIELD_CHANGE_DIFF"
        ).iloc[0, 0],
    }


def get_blast_radius(field):
    sql = f"""
        select
            downstream_fields,
            downstream_ctes,
            downstream_models,
            downstream_domains,
            blast_radius_score
        from {GOV}.FIELD_BLAST_RADIUS
        where field_fqn = '{field}'
        order by created_at desc
        limit 1
    """
    df = fetch_df(sql)
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def get_blast_radius_summary():
    sql = f"""
        select
            avg(blast_radius_score) as avg_score,
            max(blast_radius_score) as max_score
        from {GOV}.FIELD_BLAST_RADIUS
    """
    df = fetch_df(sql)
    if df.empty:
        return {"avg_score": 0, "max_score": 0}
    return df.iloc[0].to_dict()
