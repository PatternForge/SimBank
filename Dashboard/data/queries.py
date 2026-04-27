import pandas as pd
from connection import get_connection


def fetch_df(sql):
    conn = get_connection()
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


def get_all_fields():
    df = fetch_df("select field_fqn from FIELD_CATALOG order by field_fqn")
    return df["field_fqn"].tolist()


def get_forward_lineage(field):
    sql = f"""
        select from_field, to_field, layer
        from FIELD_LINEAGE
        where from_field = '{field}'
    """
    return fetch_df(sql).to_dict("records")


def get_backward_lineage(field):
    sql = f"""
        select from_field, to_field, layer
        from FIELD_LINEAGE
        where to_field = '{field}'
    """
    return fetch_df(sql).to_dict("records")


def get_field_summary(field):
    sql = f"""
        select *
        from FIELD_CATALOG
    where field_fqn = '{field}'
    """
    df = fetch_df(sql)
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def get_broken_fields():
    df = fetch_df("""
        select field_fqn
        from FIELD_HEALTH
        where status = 'BROKEN'
        order by field_fqn
    """)
    return df["field_fqn"].tolist()


def get_failure_context(field):
    df = fetch_df("""
        select *
        from FIELD_HEALTH
        where field_fqn = '{field}'
    """)
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def get_system_metrics():
    return {
        "healthy_pct": fetch_df("select avg(is_healthy) from FIELD_HEALTH").iloc[0, 0],
        "broken_fields": fetch_df("select count(*) from FIELD_HEALTH where status='BROKEN'").iloc[0, 0],
        "open_drifts": fetch_df("select count(*) from FIELD_CHANGE_DIFF").iloc[0, 0],
    }

