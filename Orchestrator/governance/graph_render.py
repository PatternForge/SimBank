from graphviz import Digraph


def render_graph(root, edges, direction="downstream"):
    dot = Digraph(format="svg")
    dot.attr(rankdir="LR")

    dot.node(root, root, shape="box")

    for src, tgt in edges:
        dot.node(src, src)
        dot.node(tgt, tgt)
        dot.edge(src, tgt)

    return dot

