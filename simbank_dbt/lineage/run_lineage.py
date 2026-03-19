import argparse
import html
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import sqlglot
from sqlglot import exp
from dotenv import load_dotenv
load_dotenv()


MODELS_DIR = Path("../models/staging")
OUTPUT_DIR = Path("../../lineage_output")
OUTPUT_DIR.mkdir(exist_ok=True)


def get_snowflake_columns(schema, table):
    try:
        import snowflake.connector
        conn = snowflake.connector.connect(
            user=os.environ.get("SNOWFLAKE_USER"),
            password=os.environ.get("SNOWFLAKE_PASSWORD"),
            account=os.environ.get("SNOWFLAKE_ACCOUNT"),
            warehouse="SIMBANK_WH",
            database="SIMBANK",
            schema=schema.upper()
        )
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = '{schema.upper()}'
            AND TABLE_NAME = '{table.upper()}'
            ORDER BY ORDINAL_POSITION
        """)
        cols = {row[0].upper() for row in cursor.fetchall()}
        cursor.close()
        conn.close()
        return cols
    except Exception as e:
        print(f" ERROR: {e}")
        # return set()


def resolve_source_fields(alias_map):
    for src_name in alias_map.values():
        if "." in src_name:
            parts = src_name.split(".")
            if len(parts) == 2:
                schema, table = parts
                return get_snowflake_columns(schema, table)
    return set()


def load_sql(path):
    with open(path, "r", encoding="utf-8") as f:
        return html.unescape(f.read())


def normalize_dbt(sql):
    sql = re.sub(
        r"\{\{\s*source\(['\"](\w+)['\"],\s*['\"](\w+)['\"]\)\s*\}\}",
        lambda m: f"{m.group(1)}.{m.group(2)}",
        sql
    )
    sql = re.sub(
        r"\{\{\s*ref\(['\"](\w+)['\"]\)\s*\}\}",
        lambda m: m.group(1),
        sql
    )
    return sql


def get_cte_name(cte):
    a = cte.alias
    return a.name.upper() if hasattr(a, "name") else str(a).upper()


def get_alias_name(alias):
    return alias.name.upper() if hasattr(alias, "name") else str(alias).upper()


def extract_column_dependencies(expr, default_source, alias_map):
    if not isinstance(expr, exp.Expression):
        return set()
    deps = set()
    if isinstance(expr, exp.Column):
        if expr.table:
            t = alias_map.get(expr.table.upper(), expr.table.upper())
        else:
            t = default_source
        if t:
            deps.add((t, expr.name.upper()))
    for child in expr.iter_expressions():
        deps |= extract_column_dependencies(child, default_source, alias_map)
    return deps


def extract_lineage(sql, dialect="snowflake"):
    parsed = sqlglot.parse_one(sql, dialect=dialect)
    cte_list = list(parsed.find_all(exp.CTE))
    cte_order = [get_cte_name(c) for c in cte_list]
    cte_set = {get_cte_name(c): c for c in cte_list}
    output_fields = {}
    lineage = []

    for cte_name in cte_order:
        cte = cte_set[cte_name]
        select = cte.find(exp.Select)
        tables = list(select.find_all(exp.Table))

        alias_map = {}
        for tbl in tables:
            real = tbl.name.upper()
            alias = get_alias_name(tbl.alias) if tbl.alias else real
            alias_map[alias] = real

        default_source = tables[0].name.upper() if len(tables) == 1 else None
        field_map = []
        tracked_cols = set()

        cte_names_set = set(cte_order)

        for src_name in alias_map.values():
            if src_name not in cte_names_set and src_name not in output_fields:
                cols = get_snowflake_columns("RAW", src_name)
                if cols:
                    output_fields[src_name] = cols

        for proj in select.expressions:

            if isinstance(proj, exp.Alias):
                # Explicitly named field: P.ACCOUNTBALANCE * 1.1 AS ADJUSTEDBALANCE
                out_col = get_alias_name(proj.alias)
                deps = extract_column_dependencies(proj.this, default_source, alias_map)
                field_map.append((out_col, deps))
                tracked_cols.add(out_col)

            elif isinstance(proj, exp.Column) and proj.name == "*":
                # P.* — expand all fields from the upstream CTE that P resolves to
                src_alias = proj.table.upper() if proj.table else None
                src = alias_map.get(src_alias, src_alias) if src_alias else default_source
                for col in output_fields.get(src, set()):
                    if col not in tracked_cols:
                        field_map.append((col, {(src, col)}))
                        tracked_cols.add(col)

            elif isinstance(proj, exp.Star):
                # Bare * without table qualifier — expand from default source
                for tbl in tables:
                    src = tbl.name.upper()
                    for col in output_fields.get(src, set()):
                        if col not in tracked_cols:
                            field_map.append((col, {(src, col)}))
                            tracked_cols.add(col)

            elif isinstance(proj, exp.Column):
                # Simple column reference: P.ACCOUNTBALANCE or ACCOUNTBALANCE
                out_col = proj.name.upper()
                if proj.table:
                    src = alias_map.get(proj.table.upper(), proj.table.upper())
                else:
                    src = default_source
                deps = {(src, out_col)} if src else set()
                field_map.append((out_col, deps))
                tracked_cols.add(out_col)

        output_fields[cte_name] = {col for col, _ in field_map}
        lineage.append({"cte": cte_name, "field_map": field_map})

    return lineage


def build_lineage_table(lineage, model_name):
    rows = []
    for entry in lineage:
        cte = entry["cte"]
        for out_col, deps in entry["field_map"]:
            if not deps:
                rows.append({
                    "model": model_name,
                    "cte": cte,
                    "field_name": out_col,
                    "source_cte": None,
                    "source_field": None
                })
            for src_cte, src_col in deps:
                rows.append({
                    "model": model_name,
                    "cte": cte,
                    "field_name": out_col,
                    "source_cte": src_cte,
                    "source_field": src_col
                })
    return rows


def build_cte_dag(sql, dialect="snowflake"):
    parsed = sqlglot.parse_one(sql, dialect=dialect)
    G = nx.DiGraph()
    ctes = {get_cte_name(c): c for c in parsed.find_all(exp.CTE)}
    for name, cte in ctes.items():
        select = cte.find(exp.Select)
        if not select:
            continue
        for table in select.find_all(exp.Table):
            upstream = table.name.upper()
            if upstream in ctes:
                G.add_edge(upstream, name)
    return G


def build_lineage_graph(df):
    G = nx.DiGraph()
    for r in df.itertuples():
        if r.source_cte and r.source_field:
            s = f"{r.source_cte}.{r.source_field}"
            t = f"{r.cte}.{r.field_name}"
            if s != t:
                G.add_edge(s, t)
    return G


def plot_split_lineage(G, cte_dag, cte, field, model_name):
    target = f"{cte}.{field}"
    if target not in G.nodes:
        print(f"\n  No lineage found for {cte}.{field} — check spelling or extraction coverage.")
        return

    topo = list(nx.topological_sort(cte_dag))
    upstream = [x for x in topo if x in nx.ancestors(cte_dag, cte)]
    downstream = [x for x in topo if x in nx.descendants(cte_dag, cte)]

    up_fields = {n for n in nx.ancestors(G, target) if n.split(".")[0] in upstream}
    down_fields = {n for n in nx.descendants(G, target) if n.split(".")[0] in downstream}

    def group(fields, ctes):
        return {c: sorted(f for f in fields if f.split(".")[0] == c) for c in ctes}

    up_map = group(up_fields, upstream)
    down_map = group(down_fields, downstream)

    pos = {}
    y_step = 0.20
    block_gap = 0.35
    start_y = 2.0
    x_left, x_mid, x_right = 0.20, 1.0, 1.80

    y = start_y
    pos["UP_HDR"] = (x_left, y)
    y -= 0.5
    for c in upstream:
        flds = up_map.get(c, [])
        if not flds:
            continue
        pos[f"{c}_LBL"] = (x_left, y)
        y -= y_step
        for f in flds:
            pos[f] = (x_left, y)
            y -= y_step
        y -= block_gap

    pos["TGT_HDR"] = (x_mid, start_y - 0.5)
    pos[target] = (x_mid, start_y - 0.9)

    y = start_y
    pos["DN_HDR"] = (x_right, y)
    y -= 0.5
    for c in downstream:
        flds = down_map.get(c, [])
        if not flds:
            continue
        pos[f"{c}_LBL"] = (x_right, y)
        y -= y_step
        for f in flds:
            pos[f] = (x_right, y)
            y -= y_step
        y -= block_gap

    H = nx.DiGraph()
    for s, t in G.edges():
        if s in pos and t in pos:
            H.add_edge(s, t)

    y_vals = [y for _, y in pos.values()]
    y_min, y_max = min(y_vals) - 0.5, max(y_vals) + 0.5

    plt.figure(figsize=(32, 20))
    plt.suptitle(f"{model_name}  ·  {cte}.{field}", fontsize=11, color="#555555", y=0.98)
    nx.draw(H, pos, with_labels=False, node_size=0, edge_color="gray", arrowsize=7)
    ax = plt.gca()

    for k, (x, y) in pos.items():
        if k == "UP_HDR":
            ax.text(x, y, "UPSTREAM", fontsize=13, weight="bold", ha="center")
        elif k == "DN_HDR":
            ax.text(x, y, "DOWNSTREAM", fontsize=13, weight="bold", ha="center")
        elif k == "TGT_HDR":
            ax.text(x, y, "TARGET", fontsize=13, weight="bold", ha="center")
        elif k.endswith("_LBL"):
            ax.text(x, y, k.replace("_LBL", ""), fontsize=8, weight="bold", ha="center")
        elif "." in k:
            color = "#c0392b" if k == target else "black"
            weight = "bold" if k == target else "normal"
            ax.text(x, y, k.split(".")[1], fontsize=7, ha="center", color=color, weight=weight)

    plt.xlim(0, 2)
    plt.ylim(y_min, y_max)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def discover_models():
    return sorted(p.stem for p in MODELS_DIR.glob("*.sql"))


def load_model(model_name):
    path = MODELS_DIR / f"{model_name}.sql"
    sql_raw = load_sql(path)
    sql = normalize_dbt(sql_raw)
    lineage = extract_lineage(sql)
    rows = build_lineage_table(lineage, model_name)
    df = pd.DataFrame(rows)
    cte_dag = build_cte_dag(sql)
    G = build_lineage_graph(df)
    return df, cte_dag, G, lineage


def dump_model(model_name):
    print(f"\n  Extracting {model_name}...")
    df, _, _, _ = load_model(model_name)
    out = OUTPUT_DIR / f"{model_name}_lineage.csv"
    df.to_csv(out, index=False)
    print(f"  Written to {out}  ({len(df)} rows)")


def dump_all(models):
    frames = []
    for m in models:
        print(f"  Extracting {m}...")
        df, _, _, _ = load_model(m)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    out = OUTPUT_DIR / "field_lineage_all.csv"
    combined.to_csv(out, index=False)
    print(f"\n  Full dump written to {out}  ({len(combined)} rows across {len(models)} models)")


def prompt_choice(label, options):
    print(f"\n  {label}:")
    for i, o in enumerate(options, 1):
        print(f"    {i:>2}.  {o}")
    upper_options = [o.upper() for o in options]
    while True:
        raw = input("\n  Select number or type name: ").strip().upper()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        if raw in upper_options:
            return options[upper_options.index(raw)]
        print("  Invalid selection — try again.")


def interactive_mode():
    print("\n╔══════════════════════════════╗")
    print("║   SimBank Lineage Explorer   ║")
    print("╚══════════════════════════════╝")

    models = discover_models()
    if not models:
        print(f"\n  No .sql files found in {MODELS_DIR}")
        sys.exit(1)

    print("\n  What would you like to do?")
    print("    1.  Trace a specific field")
    print("    2.  Dump a model to CSV")
    print("    3.  Dump all models to CSV")

    while True:
        mode = input("\n  Select number: ").strip()
        if mode in ("1", "2", "3"):
            break
        print("  Invalid selection — try again.")

    if mode == "3":
        print()
        dump_all(models)
        return

    model_name = prompt_choice("Available models", models)

    if mode == "2":
        print()
        dump_model(model_name)
        return

    print(f"\n  Loading {model_name}...")
    df, cte_dag, G, lineage = load_model(model_name)

    cte_names = [entry["cte"] for entry in lineage]
    cte_name = prompt_choice("CTEs", cte_names)

    fields = sorted({col for entry in lineage if entry["cte"] == cte_name for col, _ in entry["field_map"]})
    field = prompt_choice(f"Fields in {cte_name}", fields)

    print(f"\n  Plotting {cte_name}.{field}\n")
    plot_split_lineage(G, cte_dag, cte_name, field, model_name)


def main():
    parser = argparse.ArgumentParser(
        prog="run_lineage",
        description="SimBank field-level lineage extractor",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_lineage.py\n"
            "  python run_lineage.py --model stg_retail_deposits\n"
            "  python run_lineage.py --model stg_retail_deposits --cte BALANCE_CALCS --field ACCOUNTBALANCE\n"
            "  python run_lineage.py --dump\n"
        )
    )
    parser.add_argument("--model", help="Model name (without .sql)")
    parser.add_argument("--cte",   help="CTE name (requires --model)")
    parser.add_argument("--field", help="Field name (requires --model and --cte)")
    parser.add_argument("--dump",  action="store_true", help="Dump all models to a single CSV")

    args = parser.parse_args()

    if not any(vars(args).values()):
        interactive_mode()
        return

    if args.dump:
        models = discover_models()
        print(f"\n  Found {len(models)} model(s).\n")
        dump_all(models)
        return

    if args.model and not args.cte and not args.field:
        dump_model(args.model)
        return

    if args.model and args.cte and args.field:
        print(f"\n  Loading {args.model}...")
        df, cte_dag, G, lineage = load_model(args.model)
        plot_split_lineage(G, cte_dag, args.cte.upper(), args.field.upper(), args.model)
        return

    print("\n  Invalid argument combination. Run with --help for usage.")
    sys.exit(1)


if __name__ == "__main__":
    main()
