from graphviz import Digraph

def render_graph(root, edges, direction="downstream"):
    dot = Digraph(engine="dot")

    dot.node(root, root)

    for src, tgt in edges:
        dot.node(src, src)
        dot.node(tgt, tgt)
        dot.edge(src, tgt)

    return dot

def save_svg(dot, path):
    svg = dot.source  # pure Python, no external dot call
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)
