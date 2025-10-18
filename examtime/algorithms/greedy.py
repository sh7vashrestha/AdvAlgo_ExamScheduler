from typing import Dict
import random
import networkx as nx

def welsh_powell(G: nx.Graph, order: str = 'degree') -> Dict[str, int]:
    if order == 'degree':
        nodes = sorted(G.nodes(), key=lambda u: G.degree(u), reverse=True)
    elif order == 'random':
        nodes = list(G.nodes())
        random.shuffle(nodes)
    else:
        raise ValueError("order must be 'degree' or 'random'")
    coloring: Dict[str, int] = {}
    for u in nodes:
        neighbor_colors = {coloring[v] for v in G.neighbors(u) if v in coloring}
        c = 0
        while c in neighbor_colors:
            c += 1
        coloring[u] = c
    return coloring
