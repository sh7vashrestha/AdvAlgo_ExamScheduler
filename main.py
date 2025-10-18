import argparse
import networkx as nx
from typing import Dict

from examtime.io_utils import (
    load_rooms, load_periods, load_exam_durations, load_toronto_stu,
    load_edge_list_csv, save_schedule_csv, save_rooming_csv
)
from examtime.models import Exam, Period, Schedule
from examtime.graph_build import build_conflict_graph_from_students
from examtime.scheduling.assign_timeslots import assign_timeslots
from examtime.scheduling.room_assignment import assign_rooms_greedy
from examtime.scheduling.evaluation import summary

# SA imports
from examtime.algorithms.sa import simulated_annealing, SAParams


def map_colors_to_periods(coloring: Dict[str, int], periods: Dict[str, Period], exams: Dict[str, Exam]) -> Dict[str, str]:
    """
    Map SA color indices -> actual period IDs.
    - If there are fewer periods than colors, we wrap (may cause conflicts; avoided in practice by penalty).
    - Then ensure duration feasibility (pick first period that fits if needed).
    """
    sorted_pids = sorted(periods.keys())
    period_count = len(sorted_pids)
    period_of: Dict[str, str] = {}
    for ex_id, c in coloring.items():
        pid = sorted_pids[c % period_count]
        if exams[ex_id].duration_min > periods[pid].duration_min:
            # pick first feasible period
            for alt in sorted_pids:
                if exams[ex_id].duration_min <= periods[alt].duration_min:
                    pid = alt
                    break
        period_of[ex_id] = pid
    return period_of


def main():
    p = argparse.ArgumentParser(description="ExamTime – Room & Duration Aware Timetabling")
    # Input modes
    p.add_argument('--stu', type=str, help='Toronto .stu file (student -> list of exams)')
    p.add_argument('--edges', type=str, help='Edge list CSV u,v')
    p.add_argument('--generate', type=int, default=None, help='Generate synthetic graph with N courses (no rooms)')
    p.add_argument('--density', type=float, default=0.15)

    # Resources
    p.add_argument('--rooms', type=str, required=False, help='rooms.csv with id,capacity')
    p.add_argument('--periods', type=str, required=False, help='periods.csv with id,start,end,duration_minutes')
    p.add_argument('--durations', type=str, required=False, help='Optional CSV exam_id,duration_minutes')

    # Algo
    p.add_argument('--algo', type=str, default='dsatur', help='greedy | dsatur | sa')

    # SA params
    p.add_argument('--sa_T0', type=float, default=1.0)
    p.add_argument('--sa_alpha', type=float, default=0.995)
    p.add_argument('--sa_iters', type=int, default=200000)
    p.add_argument('--sa_pen_conflict', type=float, default=1000.0)
    p.add_argument('--sa_pen_slot', type=float, default=1.0)
    p.add_argument('--time_limit', type=float, default=60.0, help='SA time cap (seconds)')

    # Output
    p.add_argument('--out_schedule', type=str, default='schedule.csv')
    p.add_argument('--out_rooming', type=str, default='rooming.csv')
    args = p.parse_args()

    # Load students/exams or edges
    if args.stu:
        students, enroll_per_exam = load_toronto_stu(args.stu)
        G = build_conflict_graph_from_students(students)
    elif args.edges:
        edges = load_edge_list_csv(args.edges)
        G = nx.Graph()
        G.add_edges_from(edges)
        import random
        enroll_per_exam = {n: random.randint(20, 200) for n in G.nodes()}
    elif args.generate is not None:
        import random
        G = nx.gnp_random_graph(args.generate, args.density)
        G = nx.relabel_nodes(G, lambda x: f"E{x}")
        enroll_per_exam = {n: random.randint(20, 200) for n in G.nodes()}
    else:
        raise SystemExit("Provide --stu, --edges, or --generate N")

    # Exams dict (default duration 120)
    exams: Dict[str, Exam] = {ex_id: Exam(id=ex_id, duration_min=120, enrollment=cnt)
                              for ex_id, cnt in enroll_per_exam.items()}
    if args.durations:
        dur_map = load_exam_durations(args.durations)
        for ex_id, d in dur_map.items():
            if ex_id in exams:
                exams[ex_id].duration_min = d

    # Rooms & periods
    rooms = load_rooms(args.rooms) if args.rooms else {}
    if args.periods:
        periods = load_periods(args.periods)
    else:
        # Default periods if none provided (bumped from 10 -> 30)
        periods = {f"P{i}": Period(id=f"P{i}", duration_min=120) for i in range(30)}

    # === Period (timeslot) assignment ===
    if args.algo in ('greedy', 'dsatur'):
        sched = assign_timeslots(G, exams, periods, algo=args.algo)
    elif args.algo == 'sa':
        # Treat --time_limit as the total SA-stage budget, including initializer
        import time as _time
        _start = _time.perf_counter()
        # Good initializer: DSATUR-based schedule -> initial color map by period index
        init_sched = assign_timeslots(G, exams, periods, algo='dsatur')
        elapsed_init = _time.perf_counter() - _start
        remaining = max(0.01, float(args.time_limit) - elapsed_init)
        sorted_pids = sorted(periods.keys())
        index_by_pid = {pid: i for i, pid in enumerate(sorted_pids)}
        init_colors = {ex_id: index_by_pid[pid] for ex_id, pid in init_sched.period_of.items()}
        # Run SA within the remaining budget
        sa_params = SAParams(
            T0=args.sa_T0,
            alpha=args.sa_alpha,
            max_iters=args.sa_iters,
            penalty_conflict=args.sa_pen_conflict,
            penalty_slot=args.sa_pen_slot
        )
        color_map = simulated_annealing(G, init=init_colors, params=sa_params, time_limit=remaining)
        # Map colors back to real periods and build Schedule
        sched = Schedule()
        sched.period_of = map_colors_to_periods(color_map, periods, exams)
    else:
        raise SystemExit("Unknown --algo. Use greedy | dsatur | sa")

    # === Room assignment ===
    if rooms:
        sched = assign_rooms_greedy(exams, rooms, sched)

    # Validate & print summary
    print(summary(G, exams, rooms, periods, sched))

    # Save CSVs
    save_schedule_csv(args.out_schedule, sched.period_of)
    if rooms:
        save_rooming_csv(args.out_rooming, sched.rooming)
        print(f"Saved: {args.out_schedule}, {args.out_rooming}")
    else:
        print(f"Saved: {args.out_schedule}")


if __name__ == '__main__':
    main()
