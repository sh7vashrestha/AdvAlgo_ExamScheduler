from typing import Dict, List, Tuple, Set
import networkx as nx

from ..models import Exam, Room, Period, Schedule

def conflicts_ok(G: nx.Graph, sched: Schedule) -> bool:
    for u, v in G.edges():
        if sched.period_of.get(u) == sched.period_of.get(v):
            return False
    return True

def durations_ok(exams: Dict[str, Exam], periods: Dict[str, Period], sched: Schedule) -> bool:
    for ex_id, pid in sched.period_of.items():
        if ex_id not in exams or pid not in periods:
            return False
        if exams[ex_id].duration_min > periods[pid].duration_min:
            return False
    return True

def capacity_ok(exams: Dict[str, Exam], rooms: Dict[str, Room], sched: Schedule) -> bool:
    for pid, pairs in sched.rooming.items():
        used = set()
        placed = set()
        for ex_id, room_id in pairs:
            key = (ex_id, room_id)
            if key in used:
                return False
            used.add(key)
            placed.add(ex_id)
            if room_id not in rooms:
                return False
            if rooms[room_id].capacity < exams[ex_id].enrollment:
                return False
        to_place = {e for e, p in sched.period_of.items() if p == pid}
        if placed != to_place:
            return False
    return True
