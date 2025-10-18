import networkx as nx
from ..models import Exam, Room, Period, Schedule
from .validation import conflicts_ok, durations_ok, capacity_ok


def _greedy_clique_lb(G: nx.Graph) -> int:
    """Fast lower bound on chromatic number via a greedy maximal clique.

    Picks the highest-degree node, then greedily grows a clique by repeatedly
    adding a node that is adjacent to all current clique members. This is a
    heuristic lower bound (size of a found clique), O(m) per step.
    """
    if G.number_of_nodes() == 0:
        return 0
    # Start from a high-degree node
    seed = max(G.nodes(), key=lambda u: G.degree(u))
    clique = {seed}
    candidates = set(G.neighbors(seed))
    while candidates:
        # Choose candidate with max degree restricted to current candidate set
        u = max(candidates, key=lambda v: G.degree(v))
        # Filter to nodes adjacent to all in current clique
        new_cands = {v for v in candidates if all(G.has_edge(v, w) for w in clique)}
        if u in new_cands:
            clique.add(u)
            candidates = new_cands.intersection(G.neighbors(u))
        else:
            candidates.remove(u)
    return len(clique)

def summary(G: nx.Graph, exams, rooms, periods, sched: Schedule) -> str:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    periods_used = len(set(sched.period_of.values()))
    ok_conf = conflicts_ok(G, sched)
    ok_dur = durations_ok(exams, periods, sched)
    ok_cap = True if not rooms else capacity_ok(exams, rooms, sched)
    total_periods = len(periods)
    lb = _greedy_clique_lb(G)
    warning = ""
    if total_periods < lb:
        warning = (
            f"Warning: periods={total_periods} < clique LB={lb}; zero-conflict coloring is impossible.\n"
        )
    return (
        f"Nodes: {n}  Edges: {m}\n"
        f"Periods available: {total_periods}  Used: {periods_used}\n"
        f"Clique lower bound: {lb}\n"
        f"Valid (conflicts): {ok_conf}  Valid (durations): {ok_dur}  Valid (capacity): {ok_cap}\n"
        f"{warning}"
    )
