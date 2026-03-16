import sqlglot
from sqlglot import exp


def extract_lineage(sql, dialect="snowflake"):
    parsed = sqlglot.parse_one(sql, dialect=dialect)
    lineage = []

    for cte in parsed.find_all(exp.CTE):
        cte_name = cte.alias.upper()
        alias_map = {}
        referenced = set()
        derived = set()

        select = cte.find(exp.Select)
        if select:
            for table in select.find_all(exp.Table):
                if table.alias:
                    alias_map[table.alias.upper()] = table.name.upper()
                else:
                    alias_map[table.name.upper()] = table.name.upper()

            for col in select.find_all(exp.Column):
                table_ref = col.table.upper() if col.table else ""
                resolved = alias_map.get(table_ref, table_ref)
                referenced.add((col.name.upper(), resolved))

            for alias in select.find_all(exp.Alias):
                derived.add(alias.alias.upper())

        lineage.append({
            "cte": cte_name,
            "alias_map": alias_map,
            "referenced": referenced,
            "derived": derived
        })

    return lineage


def resolve_wildcards(lineage):
    cte_fields = {}
    resolved_lineage = []

    for entry in lineage:
        cte_name = entry["cte"]
        alias_map = entry["alias_map"]
        resolved = set()

        for item in entry["referenced"]:
            if not isinstance(item, tuple):
                continue
            field, source = item
            if field == "*":
                upstream_cte = alias_map.get(source, source)
                inherited = cte_fields.get(upstream_cte, set())
                resolved.update(inherited)
            else:
                resolved.add((field, source))

        snapshot = frozenset(resolved | {(f, cte_name) for f in entry["derived"]})
        cte_fields[cte_name] = snapshot

        resolved_lineage.append({
            "cte": cte_name,
            "referenced": resolved,
            "derived": entry["derived"]
        })

    return resolved_lineage


def build_lineage_table(resolved_lineage, model_name="stg_retail_deposits"):
    rows = []
    seen = set()

    for entry in resolved_lineage:
        cte_name = entry["cte"]

        for field, source in entry["referenced"]:
            key = (model_name, cte_name, field, "referenced")
            if key not in seen:
                seen.add(key)
                rows.append({
                    "model": model_name,
                    "cte": cte_name,
                    "field_name": field,
                    "field_type": "referenced",
                    "source_cte": source
                })

        for field in entry["derived"]:
            key = (model_name, cte_name, field, "derived")
            if key not in seen:
                seen.add(key)
                rows.append({
                    "model": model_name,
                    "cte": cte_name,
                    "field_name": field,
                    "field_type": "derived",
                    "source_cte": cte_name
                })

    return rows


def run_lineage_extractor(sql, model_name="stg_retail_deposits", dialect="snowflake"):
    lineage = extract_lineage(sql, dialect=dialect)
    resolved = resolve_wildcards(lineage)
    rows = build_lineage_table(resolved, model_name=model_name)
    return rows
