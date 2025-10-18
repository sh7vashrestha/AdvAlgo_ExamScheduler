from typing import Dict, Optional, Set
import networkx as nx

def dsatur_feasible(G: nx.Graph, feasible_colors: Optional[Dict[str, Set[int]]] = None) -> Dict[str, int]:
    """DSATUR with optional feasibility mask per node (period indices allowed)."""
    coloring: Dict[str, int] = {}
    saturation = {u: set() for u in G.nodes()}
    degrees = {u: G.degree(u) for u in G.nodes()}

    def first_feasible_color(u, neighbor_colors):
        """Pick the smallest allowed color not used by neighbors.

        If feasibility mask is provided and all feasible colors are already
        used by neighbors, fall back to the feasible color that minimizes
        conflicts (least neighbor count of that color). This prevents an
        infinite loop when the graph's chromatic number exceeds available
        feasible colors.
        """
        # No feasibility mask: standard DSATUR choice
        if feasible_colors is None or u not in feasible_colors:
            c = 0
            while c in neighbor_colors:
                c += 1
            return c
        # With feasibility: try the smallest feasible not used by neighbors
        for c in sorted(feasible_colors[u]):
            if c not in neighbor_colors:
                return c
        # Fallback: choose feasible color with minimal conflict increase
        # Count neighbor colors
        counts = {}
        for v in G.neighbors(u):
            if v in coloring:
                cv = coloring[v]
                counts[cv] = counts.get(cv, 0) + 1
        return min(sorted(feasible_colors[u]), key=lambda c: counts.get(c, 0))

    while len(coloring) < G.number_of_nodes():
        candidates = [u for u in G.nodes() if u not in coloring]
        u = max(candidates, key=lambda x: (len(saturation[x]), degrees[x]))
        neighbor_colors = {coloring[v] for v in G.neighbors(u) if v in coloring}
        c = first_feasible_color(u, neighbor_colors)
        coloring[u] = c
        for v in G.neighbors(u):
            if v not in coloring:
                saturation[v].add(c)
    return coloring
