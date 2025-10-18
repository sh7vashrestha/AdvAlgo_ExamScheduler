from typing import Dict, Set, List, Tuple
import networkx as nx

def build_conflict_graph_from_students(students: Dict[str, Set[str]]) -> nx.Graph:
    G = nx.Graph()
    all_exams: Set[str] = set()
    for exams in students.values():
        all_exams.update(exams)
    for ex in all_exams:
        G.add_node(ex)
    for exams in students.values():
        exams = list(exams)
        for i in range(len(exams)):
            for j in range(i + 1, len(exams)):
                u, v = exams[i], exams[j]
                if u != v:
                    G.add_edge(u, v)
    return G
