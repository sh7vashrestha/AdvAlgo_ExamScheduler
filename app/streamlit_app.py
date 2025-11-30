import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import io
import json
import hashlib
import time
from typing import Dict
import csv
import random

import networkx as nx
import pandas as pd
import streamlit as st

from examtime.io_utils import (
    load_rooms, load_periods, load_exam_durations, load_toronto_stu, load_edge_list_csv
)
from examtime.models import Exam, Period, Schedule
from examtime.graph_build import build_conflict_graph_from_students
from examtime.scheduling.assign_timeslots import assign_timeslots
from examtime.scheduling.room_assignment import assign_rooms_greedy
from examtime.scheduling.evaluation import summary
from examtime.algorithms.sa import simulated_annealing, SAParams

# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(page_title="ExamTime – Scheduler", layout="wide")
st.title("ExamTime – Room & Duration Aware Scheduler")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def map_colors_to_periods(coloring: Dict[str, int], periods: Dict[str, Period], exams: Dict[str, Exam]) -> Dict[str, str]:
    """Map SA color indices to real period IDs, repairing durations if needed."""
    sorted_pids = sorted(periods.keys())
    period_count = len(sorted_pids)
    period_of: Dict[str, str] = {}
    for ex_id, c in coloring.items():
        pid = sorted_pids[c % period_count]
        if exams[ex_id].duration_min > periods[pid].duration_min:
            for alt in sorted_pids:
                if exams[ex_id].duration_min <= periods[alt].duration_min:
                    pid = alt
                    break
        period_of[ex_id] = pid
    return period_of

def _bytes_of(upload):
    if upload is None:
        return None
    return upload.getvalue()

# ---------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------
@st.cache_data
def load_rooms_cached(rooms_bytes: bytes):
    if rooms_bytes is None:
        return {}
    return load_rooms(io.BytesIO(rooms_bytes))

@st.cache_data
def load_periods_cached(periods_bytes: bytes):
    if periods_bytes is None:
        return {}
    return load_periods(io.BytesIO(periods_bytes))

@st.cache_data
def load_durations_cached(durations_bytes: bytes):
    if durations_bytes is None:
        return {}
    return load_exam_durations(io.BytesIO(durations_bytes))

@st.cache_data
def build_graph_from_stu_cached(stu_bytes: bytes):
    students, enroll = load_toronto_stu(io.StringIO(stu_bytes.decode("utf-8")))
    G = build_conflict_graph_from_students(students)
    return G, enroll

@st.cache_data
def build_graph_from_edges_cached(edges_bytes: bytes):
    edges = load_edge_list_csv(io.BytesIO(edges_bytes))
    G = nx.Graph()
    G.add_edges_from(edges)
    return G

@st.cache_data
def build_synthetic_cached(n: int, p: float, seed: int = 42):
    G = nx.gnp_random_graph(n, p, seed=seed)
    G = nx.relabel_nodes(G, lambda x: f"E{x}")
    return G

@st.cache_data
def run_timeslots_cached(algo: str, edges_list, exams_dict, periods_dict):
    G = nx.Graph()
    G.add_nodes_from(exams_dict.keys())
    G.add_edges_from(edges_list)
    exams = {k: Exam(id=k, duration_min=v["duration"], enrollment=v["enroll"]) for k, v in exams_dict.items()}
    periods = {k: Period(id=k, duration_min=v["duration"]) for k, v in periods_dict.items()}
    sched = assign_timeslots(G, exams, periods, algo=algo)
    return sched.period_of

# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
st.subheader("Inputs")
mode = st.radio("Input mode", ["Toronto .stu", "Edge list CSV", "Synthetic"], horizontal=True)

st.subheader("Algorithm")
algo = st.selectbox("Timeslot algorithm", ["dsatur", "greedy", "sa"], index=0)

# Handle SA parameters
sa_state_keys = [
    "sa_T0", "sa_alpha", "sa_iters", "sa_pen_conflict", "sa_pen_slot", "sa_time_limit"
]
if "_prev_algo" not in st.session_state:
    st.session_state._prev_algo = algo
elif st.session_state._prev_algo != algo:
    if algo != "sa":
        for k in sa_state_keys:
            if k in st.session_state:
                del st.session_state[k]
    st.session_state._prev_algo = algo

# ---------------------------------------------------------------------
# Form Inputs
# ---------------------------------------------------------------------
with st.form("controls"):
    c1, c2, c3 = st.columns(3)
    rooms_file = c1.file_uploader("Rooms CSV (id,capacity)", type=["csv"])
    periods_file = c2.file_uploader("Periods CSV (id,start,end,duration_minutes)", type=["csv"])
    durations_file = c3.file_uploader("(Optional) Durations CSV (exam_id,duration_minutes)", type=["csv"])

    if mode == "Toronto .stu":
        stu_file = st.file_uploader("Toronto .stu file", type=["stu", "txt"])
        edges_file = None
        n = p = None
    elif mode == "Edge list CSV":
        edges_file = st.file_uploader("Edge list CSV (u,v)", type=["csv"])
        stu_file = None
        n = p = None
    else:
        n = st.number_input("Synthetic exams (N)", 10, 5000, 200, step=10)
        p = st.slider("Edge probability (density)", 0.01, 0.9, 0.15)
        stu_file = edges_file = None

    if algo == "sa":
        with st.expander("Simulated Annealing settings", expanded=True):
            colA, colB, colC = st.columns(3)
            sa_T0 = colA.number_input("Initial temperature (T0)", 0.01, 100.0, 1.0, 0.01, key="sa_T0")
            sa_alpha = colB.number_input("Cooling factor (alpha)", 0.90, 0.9999, 0.995, 0.0001, format="%.4f", key="sa_alpha")
            sa_iters = colC.number_input("Max iterations", 1000, 2_000_000, 200_000, 1000, key="sa_iters")
            colD, colE, colF = st.columns(3)
            sa_pen_conflict = colD.number_input("Penalty: conflicts", 0.0, 100000.0, 1000.0, 1.0, key="sa_pen_conflict")
            sa_pen_slot = colE.number_input("Penalty: #colors", 0.0, 1000.0, 1.0, 0.1, key="sa_pen_slot")
            sa_time_limit = colF.number_input("Time limit (sec)", 1.0, 600.0, 60.0, 1.0, key="sa_time_limit")

    submitted = st.form_submit_button("Run Scheduler")

# ---------------------------------------------------------------------
# Run on Submit
# ---------------------------------------------------------------------
if submitted:
    t_total0 = time.perf_counter()

    rooms_bytes = _bytes_of(rooms_file)
    periods_bytes = _bytes_of(periods_file)
    durations_bytes = _bytes_of(durations_file)

    rooms = load_rooms_cached(rooms_bytes)
    periods = load_periods_cached(periods_bytes)
    if not periods:
        periods = {f"P{i}": Period(id=f"P{i}", duration_min=120) for i in range(30)}

    if mode == "Toronto .stu":
        if stu_file is None:
            st.error("Please upload a .stu file.")
            st.stop()
        G, enroll_per_exam = build_graph_from_stu_cached(_bytes_of(stu_file))
    elif mode == "Edge list CSV":
        if edges_file is None:
            st.error("Please upload an edge list CSV.")
            st.stop()
        G = build_graph_from_edges_cached(_bytes_of(edges_file))
        enroll_per_exam = {n: random.randint(20, 200) for n in G.nodes()}
    else:
        G = build_synthetic_cached(int(n), float(p))
        enroll_per_exam = {nid: random.randint(20, 200) for nid in G.nodes()}

    exams = {ex_id: Exam(id=ex_id, duration_min=120, enrollment=cnt)
             for ex_id, cnt in enroll_per_exam.items()}

    if durations_bytes is not None:
        dur_map = load_durations_cached(durations_bytes)
        for ex_id, d in dur_map.items():
            if ex_id in exams:
                exams[ex_id].duration_min = int(d)

    t_algo0 = time.perf_counter()
    sched = Schedule()

    if algo in ("dsatur", "greedy"):
        G_edges = list(G.edges())
        exams_dict = {k: {"duration": exams[k].duration_min, "enroll": exams[k].enrollment} for k in exams.keys()}
        periods_dict = {pid: {"duration": p.duration_min} for pid, p in periods.items()}
        try:
            period_of = run_timeslots_cached(algo, G_edges, exams_dict, periods_dict)
            sched.period_of = period_of
        except Exception:
            sched = assign_timeslots(G, exams, periods, algo=algo)
    elif algo == "sa":
        sa_budget = float(sa_time_limit)
        sa_begin = time.perf_counter()
        init_sched = assign_timeslots(G, exams, periods, algo="dsatur")
        elapsed_init = time.perf_counter() - sa_begin
        remaining = max(0.01, sa_budget - elapsed_init)
        sorted_pids = sorted(periods.keys())
        index_by_pid = {pid: i for i, pid in enumerate(sorted_pids)}
        init_colors = {ex_id: index_by_pid[pid] for ex_id, pid in init_sched.period_of.items()}
        params = SAParams(
            T0=sa_T0,
            alpha=sa_alpha,
            max_iters=int(sa_iters),
            penalty_conflict=sa_pen_conflict,
            penalty_slot=sa_pen_slot
        )
        color_map = simulated_annealing(G, init=init_colors, params=params, time_limit=remaining)
        sched.period_of = map_colors_to_periods(color_map, periods, exams)

    t_algo1 = time.perf_counter()

    if rooms:
        sched = assign_rooms_greedy(exams, rooms, sched)

    # -----------------------------------------------------------------
    # Save outputs to /outputs/<algo>_<timestamp>/
    # -----------------------------------------------------------------
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    algo_clean = algo.lower().replace(" ", "_")
    output_dir = os.path.join(os.getcwd(), "outputs", f"{algo_clean}_run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    schedule_path = os.path.join(output_dir, "schedule.csv")
    with open(schedule_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["exam_id", "period_id"])
        for ex_id, pid in sorted(sched.period_of.items()):
            w.writerow([ex_id, pid])

    if sched.rooming:
        rooming_path = os.path.join(output_dir, "rooming.csv")
        with open(rooming_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["period_id", "exam_id", "room_id"])
            for pid, pairs in sched.rooming.items():
                for ex_id, rid in pairs:
                    w.writerow([pid, ex_id, rid])

    summary_text = summary(G, exams, rooms, periods, sched)
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)
        f.write(f"\nAlgorithm: {algo}\n")
        f.write(f"Runtime: {time.perf_counter() - t_total0:.3f}s\n")

    # -----------------------------------------------------------------
    # UI Output
    # -----------------------------------------------------------------
    st.subheader("Summary")
    st.text(summary_text)
    t_total1 = time.perf_counter()
    st.caption(f"Timeslot algorithm time: {t_algo1 - t_algo0:.3f}s · Total time: {t_total1 - t_total0:.3f}s")

    sch_buf = io.StringIO()
    w = csv.writer(sch_buf)
    w.writerow(["exam_id", "period_id"])
    for ex_id, pid in sorted(sched.period_of.items()):
        w.writerow([ex_id, pid])
    st.download_button("Download schedule.csv", sch_buf.getvalue(), file_name="schedule.csv", mime="text/csv")

    if sched.rooming:
        room_buf = io.StringIO()
        w = csv.writer(room_buf)
        w.writerow(["period_id", "exam_id", "room_id"])
        for pid, pairs in sched.rooming.items():
            for ex_id, rid in pairs:
                w.writerow([pid, ex_id, rid])
        st.download_button("Download rooming.csv", room_buf.getvalue(), file_name="rooming.csv", mime="text/csv")

    st.info(f"Results saved locally to: {output_dir}")
    st.success("Scheduling complete.")
