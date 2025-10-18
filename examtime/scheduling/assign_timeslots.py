from typing import Dict, Set, Optional
import networkx as nx

from ..models import Exam, Period, Schedule
from ..algorithms.greedy import welsh_powell
from ..algorithms.dsatur import dsatur_feasible

def compute_feasible_colors(exams: Dict[str, Exam], periods: Dict[str, Period]):
    pid_by_index = {i: pid for i, pid in enumerate(sorted(periods.keys()))}
    index_by_pid = {pid: i for i, pid in pid_by_index.items()}
    feas = {}
    for ex_id, ex in exams.items():
        feas[ex_id] = set()
        for pid, p in periods.items():
            if ex.duration_min <= p.duration_min:
                feas[ex_id].add(index_by_pid[pid])
    return feas

def assign_timeslots(G: nx.Graph, exams: Dict[str, Exam], periods: Dict[str, Period], algo: str = 'dsatur') -> Schedule:
    sorted_pids = sorted(periods.keys())
    if algo == 'greedy':
        color_map = welsh_powell(G, order='degree')
    elif algo == 'dsatur':
        feas = compute_feasible_colors(exams, periods)
        color_map = dsatur_feasible(G, feasible_colors=feas)
    else:
        raise ValueError("algo must be 'greedy' or 'dsatur'")
    schedule = Schedule()
    for ex_id, c in color_map.items():
        pid = sorted_pids[c % len(sorted_pids)]
        if exams[ex_id].duration_min > periods[pid].duration_min:
            for alt in sorted_pids:
                if exams[ex_id].duration_min <= periods[alt].duration_min:
                    pid = alt
                    break
        schedule.period_of[ex_id] = pid
    return schedule
