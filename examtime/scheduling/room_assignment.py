from typing import Dict, List, Tuple

from ..models import Exam, Room, Schedule

def assign_rooms_greedy(exams: Dict[str, Exam], rooms: Dict[str, Room], schedule: Schedule) -> Schedule:
    by_period: Dict[str, List[str]] = {}
    for ex_id, pid in schedule.period_of.items():
        by_period.setdefault(pid, []).append(ex_id)
    schedule.rooming = {}
    for pid, ex_list in by_period.items():
        ex_list.sort(key=lambda e: exams[e].enrollment, reverse=True)
        room_ids = sorted(rooms.keys(), key=lambda r: rooms[r].capacity)
        used: List[Tuple[str, str]] = []
        for eid in ex_list:
            need = exams[eid].enrollment
            placed = False
            for rid in room_ids:
                if rooms[rid].capacity >= need and (eid, rid) not in used:
                    used.append((eid, rid))
                    placed = True
                    break
            if not placed:
                rid = max(room_ids, key=lambda r: rooms[r].capacity)
                used.append((eid, rid))
        schedule.rooming[pid] = used
    return schedule
