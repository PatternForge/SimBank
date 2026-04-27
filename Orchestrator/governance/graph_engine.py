import networkx as nx


def build_graph(lineage_rows):
    g = nx.DiGraph()
    for r in lineage_rows:
        g.add_edge(r["from_field"], r["to_field"])
    return g


def downstream_fields(g, field):
    return sorted(nx.descendants(g, field))


def upstream_fields(g, field):
    return sorted(nx.ancestors(g, field))


def blast_radius(g, field):
    return len(nx.descendants(g, field))

