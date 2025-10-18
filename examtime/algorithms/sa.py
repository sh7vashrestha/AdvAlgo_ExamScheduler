import math
import random
import time
from typing import Dict, Optional
import networkx as nx

class SAParams:
    def __init__(self, T0=1.0, alpha=0.995, max_iters=200000, penalty_conflict=1000.0, penalty_slot=1.0):
        self.T0 = T0
        self.alpha = alpha
        self.max_iters = max_iters
        self.penalty_conflict = penalty_conflict
        self.penalty_slot = penalty_slot

def timeslot_count(coloring: Dict[str, int]) -> int:
    return 0 if not coloring else max(coloring.values()) + 1

def sa_cost(G: nx.Graph, coloring: Dict[str, int], penalty_conflict: float, penalty_slot: float) -> float:
    conflicts = sum(1 for u, v in G.edges() if coloring.get(u) == coloring.get(v))
    k = timeslot_count(coloring)
    return penalty_conflict * conflicts + penalty_slot * k

def compact_colors(coloring: Dict[str, int]) -> Dict[str, int]:
    if not coloring:
        return coloring
    used = sorted(set(coloring.values()))
    relabel = {c: i for i, c in enumerate(used)}
    return {n: relabel[c] for n, c in coloring.items()}

def simulated_annealing(G: nx.Graph, init: Optional[Dict[str, int]] = None, params: Optional[SAParams] = None,
                        time_limit: Optional[float] = None, seed: Optional[int] = None) -> Dict[str, int]:
    rng = random.Random(seed)
    if params is None:
        params = SAParams()
    coloring = init.copy() if init else {}
    if not coloring:
        nodes = sorted(G.nodes(), key=lambda u: G.degree(u), reverse=True)
        for u in nodes:
            neighbor_colors = {coloring[v] for v in G.neighbors(u) if v in coloring}
            c = 0
            while c in neighbor_colors:
                c += 1
            coloring[u] = c
    best = coloring.copy()
    best_cost = sa_cost(G, best, params.penalty_conflict, params.penalty_slot)
    nodes = list(G.nodes())
    # Use a monotonic clock so system time changes don't affect the limit
    start = time.perf_counter()
    T = params.T0
    for it in range(params.max_iters):
        if time_limit and (time.perf_counter() - start) >= time_limit:
            break
        u = rng.choice(nodes)
        cur = coloring[u]
        k = timeslot_count(coloring)
        prop = rng.randrange(0, k + 1)
        if prop == cur and rng.random() < 0.5:
            prop = k
        new_col = coloring.copy()
        new_col[u] = prop
        old_cost = sa_cost(G, coloring, params.penalty_conflict, params.penalty_slot)
        new_cost = sa_cost(G, new_col, params.penalty_conflict, params.penalty_slot)
        dE = new_cost - old_cost
        if dE <= 0 or rng.random() < math.exp(-dE / max(T, 1e-9)):
            coloring = new_col
            if new_cost < best_cost:
                best_cost = new_cost
                best = new_col.copy()
        if it % 1000 == 0:
            best = compact_colors(best)
        T *= params.alpha
    return compact_colors(best)
