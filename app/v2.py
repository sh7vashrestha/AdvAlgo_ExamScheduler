import os
import io
import time
import math
import random
import string
import shutil
from datetime import datetime, timedelta
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import streamlit as st
import networkx as nx
from faker import Faker

# PDF generation
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib import colors

# ------------------------------
# Streamlit App Config
# ------------------------------
st.set_page_config(
    page_title="Exam Timetabling Simulator (Graph Coloring)",
    layout="wide",
)

# ------------------------------
# Utilities
# ------------------------------
SEED_DEFAULT = 42
rng = np.random.default_rng(SEED_DEFAULT)

@st.cache_data(show_spinner=False)
def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    return True


def make_output_dir():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("outputs", f"gen_run_{ts}")
    pdf_dir = os.path.join(out_dir, "pdfs")
    stu_pdf_dir = os.path.join(pdf_dir, "student_schedules")
    os.makedirs(stu_pdf_dir, exist_ok=True)
    return out_dir, pdf_dir, stu_pdf_dir


def save_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


# ------------------------------
# Synthetic Data Generation
# ------------------------------
@st.cache_data(show_spinner=True)
def generate_dataset(
    n_students: int,
    n_courses: int,
    n_rooms: int,
    n_periods: int,
    min_courses_per_student: int,
    max_courses_per_student: int,
    seed: int = SEED_DEFAULT,
):
    _seed_everything(seed)
    fake = Faker()
    Faker.seed(seed)

    # Students
    years = ["Sophomore", "Junior", "Senior"]
    programs = [
        "Computer Science", "Information Technology", "Data Science", "Mathematics",
        "Physics", "Chemistry", "Biology", "Economics", "Finance", "Marketing",
        "Electrical Engineering", "Mechanical Engineering", "Civil Engineering", "Psychology",
    ]
    students = []
    for sid in range(1, n_students + 1):
        students.append({
            "student_id": sid,
            "name": fake.name(),
            "program": random.choice(programs),
            "year": random.choice(years),
        })
    students_df = pd.DataFrame(students)

    # Courses
    departments = [
        "CS", "IT", "MATH", "PHYS", "CHEM", "BIO", "ECON", "FIN", "MKT",
        "EE", "ME", "CE", "PSY",
    ]
    courses = []
    for cid in range(1, n_courses + 1):
        dept = random.choice(departments)
        course_id = f"{dept}{100 + random.randrange(1, 400)}"
        course_name = f"{dept} Course {cid}"
        credit = random.choice([2, 3, 4])
        courses.append({
            "course_id": course_id,
            "course_name": course_name,
            "department": dept,
            "credit_hours": credit,
        })
    courses_df = pd.DataFrame(courses).drop_duplicates(subset=["course_id"]).reset_index(drop=True)
    # If duplicates reduced length, pad with synthetic unique suffixes
    while len(courses_df) < n_courses:
        idx = len(courses_df) + 1
        dept = random.choice(departments)
        course_id = f"{dept}{500 + idx}"
        courses_df.loc[len(courses_df)] = [course_id, f"{dept} Course {idx}", dept, random.choice([2,3,4])]

    # Rooms
    buildings = ["Science Hall", "Tech Center", "Main Building", "North Wing", "Library Annex", "Engineering Complex"]
    rooms = []
    for rid in range(1, n_rooms + 1):
        building = random.choice(buildings)
        room_number = random.randint(100, 699)
        capacity = random.randint(20, 200)
        rooms.append({
            "classroom_id": f"R{rid:03d}",
            "building_name": building,
            "room_number": room_number,
            "capacity": capacity,
        })
    rooms_df = pd.DataFrame(rooms)

    # Periods / Timeslots (~n_periods)
    # Build 2-hour slots starting 09:00 to 19:00 (5 slots/day) across enough days to reach n_periods
    slots_per_day = 8
    start_hour = 8
    slot_len_hours = 1.5
    days_needed = math.ceil(n_periods / slots_per_day)
    base_date = datetime(2025, 12, 1)
    periods = []
    pid = 1
    for d in range(days_needed):
        day_label = (base_date + timedelta(days=d)).strftime("%Y-%m-%d")
        for s in range(slots_per_day):
            if pid > n_periods:
                break
            stime = (datetime(2025, 1, 1, start_hour, 0) + timedelta(minutes=int(s * slot_len_hours * 60)))
            etime = stime + timedelta(minutes=int(slot_len_hours * 60))
            periods.append({
                "period_id": pid,
                "day": day_label,
                "start_time": stime.strftime("%H:%M"),
                "end_time": etime.strftime("%H:%M"),
            })
            pid += 1
    periods_df = pd.DataFrame(periods)

    # Enrollment distribution: make some courses very popular, others small
    # Create popularity weights via Zipf-like distribution
    popularity = np.random.zipf(a=1.4, size=n_courses)
    popularity = popularity / popularity.sum()

    # Enrollment table: each student takes 3-6 courses sampled by popularity
    enrollments = []
    course_ids = courses_df["course_id"].tolist()
    for sid in students_df["student_id"]:
        k = random.randint(min_courses_per_student, max_courses_per_student)
        chosen = np.random.choice(course_ids, size=k, replace=False, p=popularity)
        for c in chosen:
            enrollments.append({"student_id": sid, "course_id": c})
    enroll_df = pd.DataFrame(enrollments)

    # Compute course sizes for later rooming
    course_sizes = enroll_df.groupby("course_id").size().rename("exam_size").reset_index()
    courses_df = courses_df.merge(course_sizes, on="course_id", how="left").fillna({"exam_size": 0})
    courses_df["exam_size"] = courses_df["exam_size"].astype(int)

    # Toronto .stu format (line per student with course_ids)
    stu_lines = enroll_df.groupby("student_id")["course_id"].apply(list)
    # ensure every student_id appears even if missing in enroll_df
    stu_lines = stu_lines.reindex(students_df["student_id"]).apply(lambda x: x if isinstance(x, list) else [])
    stu_text = "\n".join(" ".join(map(str, lst)) for lst in stu_lines)

    return students_df, courses_df, rooms_df, periods_df, enroll_df, stu_text


# ------------------------------
# Conflict Graph Construction
# ------------------------------
@st.cache_data(show_spinner=True)
def build_conflict_graph(enroll_df: pd.DataFrame):
    # Map course -> set(student_ids)
    course_to_students = defaultdict(set)
    for row in enroll_df.itertuples(index=False):
        course_to_students[row.course_id].add(row.student_id)

    # Build graph
    G = nx.Graph()
    courses = list(course_to_students.keys())
    G.add_nodes_from(courses)

    # Efficient pairwise intersection via sorted list sweep
    course_list = list(course_to_students.items())
    n = len(course_list)
    for i in range(n):
        ci, si = course_list[i]
        for j in range(i + 1, n):
            cj, sj = course_list[j]
            # quick check using smaller set
            if len(si) < len(sj):
                small, large = si, sj
            else:
                small, large = sj, si
            hit = any(s in large for s in small)
            if hit:
                G.add_edge(ci, cj)

    # Stats
    density = nx.density(G)
    num_edges = G.number_of_edges()
    return G, density, num_edges


# ------------------------------
# Coloring Algorithms
# ------------------------------
@st.cache_data(show_spinner=True)
def color_with_dsatur(G: nx.Graph):
    coloring = nx.coloring.greedy_color(G, strategy="DSATUR")
    ncolors = 1 + max(coloring.values()) if coloring else 0
    return coloring, ncolors


@st.cache_data(show_spinner=True)
def color_with_greedy(G: nx.Graph):
    coloring = nx.coloring.greedy_color(G, strategy="largest_first")
    ncolors = 1 + max(coloring.values()) if coloring else 0
    return coloring, ncolors


def objective_conflicts(G: nx.Graph, coloring: dict) -> int:
    # Count conflicting edges that share same color
    bad = 0
    for u, v in G.edges():
        if coloring.get(u) == coloring.get(v):
            bad += 1
    return bad


def compress_colors(coloring: dict):
    # Re-map colors to 0..k-1 (stable order)
    used = sorted(set(coloring.values()))
    mapping = {c: i for i, c in enumerate(used)}
    return {k: mapping[v] for k, v in coloring.items()}, len(used)


@st.cache_data(show_spinner=True)
def anneal_refine(
    G: nx.Graph,
    init_coloring: dict,
    init_colors: int,
    target_colors: int,
    T0: float = 1.0,
    alpha: float = 0.995,
    max_iters: int = 20000,
    conflict_penalty: float = 10.0,
    time_limit_s: float = 5.0,
):
    # Try to recolor into <= target_colors while minimizing conflicts
    # If target_colors >= init_colors, we simply return the init result
    if target_colors >= init_colors:
        best, k = compress_colors(init_coloring)
        return best, k, 0, 0

    start = time.perf_counter()
    nodes = list(G.nodes())
    coloring, _ = compress_colors(init_coloring)
    best = coloring.copy()
    best_score = objective_conflicts(G, best) + conflict_penalty * max(0, (1 + max(best.values()) - target_colors))

    T = T0
    iters = 0
    rng = np.random.default_rng(SEED_DEFAULT)

    while iters < max_iters and (time.perf_counter() - start) < time_limit_s:
        iters += 1
        # choose random node and a candidate color in [0, target_colors-1]
        node = rng.choice(nodes)
        new_color = int(rng.integers(0, target_colors))
        old_color = coloring[node]
        if new_color == old_color:
            continue
        coloring[node] = new_color
        # Compute score
        # quick local delta evaluation
        delta = 0
        for nbr in G.neighbors(node):
            if coloring[nbr] == new_color:
                delta += 1
            if coloring[nbr] == old_color:
                delta -= 1
        # global penalty for colors exceeding target
        cur_max = 1 + max(coloring.values())
        penalty = conflict_penalty * max(0, cur_max - target_colors)
        score_new = objective_conflicts(G, coloring) + penalty

        # accept / reject
        # if improved or probabilistically by temperature
        if score_new <= best_score or rng.random() < math.exp(-(score_new - best_score) / max(T, 1e-9)):
            if score_new < best_score:
                best_score = score_new
                best = coloring.copy()
        else:
            coloring[node] = old_color

        T *= alpha

    best, k = compress_colors(best)
    elapsed = time.perf_counter() - start
    return best, k, best_score, elapsed


# ------------------------------
# Period Mapping & Room Assignment
# ------------------------------
@st.cache_data(show_spinner=True)
def map_colors_to_periods(coloring: dict, periods_df: pd.DataFrame):
    used_colors = sorted(set(coloring.values()))
    need = len(used_colors)
    have = len(periods_df)
    period_map = {}
    # If not enough periods, auto-extend periods_df logically (no UI change)
    if need > have:
        # extend by cloning with later days
        extra = need - have
        last_day = pd.to_datetime(periods_df["day"].iloc[-1])
        start_time = periods_df["start_time"].iloc[0]
        end_time = periods_df["end_time"].iloc[0]
        for i in range(extra):
            periods_df.loc[len(periods_df)] = {
                "period_id": len(periods_df) + 1,
                "day": (last_day + timedelta(days=i + 1)).strftime("%Y-%m-%d"),
                "start_time": start_time,
                "end_time": end_time,
            }
    # Recompute have
    have = len(periods_df)

    for i, c in enumerate(used_colors):
        period_map[c] = periods_df.iloc[i]["period_id"]

    course_period = {course: period_map[col] for course, col in coloring.items()}

    # Build schedule table
    sched_records = []
    for course, pid in course_period.items():
        row = periods_df.loc[periods_df["period_id"] == pid].iloc[0]
        sched_records.append({
            "exam_id": course,
            "period_id": pid,
            "day": row["day"],
            "start": row["start_time"],
            "end": row["end_time"],
        })
    schedule_df = pd.DataFrame(sched_records)
    return schedule_df, periods_df


@st.cache_data(show_spinner=True)
def room_assignment(schedule_df: pd.DataFrame, rooms_df: pd.DataFrame, course_sizes: pd.DataFrame):
    # For each period, assign rooms to exams greedily by exam size -> largest rooms first
    size_map = dict(zip(course_sizes["course_id"], course_sizes["exam_size"]))

    # Pre-sort rooms by capacity descending
    rooms_sorted = rooms_df.sort_values("capacity", ascending=False).reset_index(drop=True)

    assignments = []
    unassigned = []

    for pid, group in schedule_df.groupby("period_id"):
        exams = group.copy()
        exams["exam_size"] = exams["exam_id"].map(size_map).fillna(0).astype(int)
        exams = exams.sort_values("exam_size", ascending=False).reset_index(drop=True)

        # Track remaining capacity per room for this period (single seating per exam; allow multi-room split if needed)
        available = rooms_sorted.copy()
        available["used"] = False

        for ex in exams.itertuples(index=False):
            need = int(ex.exam_size)
            if need <= 0:
                continue
            # try to place in single best-fit room first
            candidate = available[~available["used"] & (available["capacity"] >= need)]
            if len(candidate) > 0:
                r = candidate.iloc[0]
                assignments.append({
                    "period_id": pid,
                    "exam_id": ex.exam_id,
                    "room_id": r.classroom_id,
                    "building": r.building_name,
                    "room_number": r.room_number,
                    "capacity": r.capacity,
                    "allocated": need,
                })
                available.loc[available["classroom_id"] == r.classroom_id, "used"] = True
                continue
            # else split across multiple rooms (largest first)
            rem = need
            for r in available[~available["used"]].itertuples(index=False):
                if rem <= 0:
                    break
                take = min(rem, int(r.capacity))
                assignments.append({
                    "period_id": pid,
                    "exam_id": ex.exam_id,
                    "room_id": r.classroom_id,
                    "building": r.building_name,
                    "room_number": r.room_number,
                    "capacity": r.capacity,
                    "allocated": take,
                })
                available.loc[available["classroom_id"] == r.classroom_id, "used"] = True
                rem -= take
            if rem > 0:
                unassigned.append({
                    "period_id": pid,
                    "exam_id": ex.exam_id,
                    "unplaced_students": rem,
                })

    rooming_df = pd.DataFrame(assignments)
    unassigned_df = pd.DataFrame(unassigned)
    return rooming_df, unassigned_df


# ------------------------------
# PDF Reports
# ------------------------------

def _canvas_header(c: canvas.Canvas, title: str):
    c.setFont("Helvetica-Bold", 14)
    c.drawString(1 * inch, 10.5 * inch, title)
    c.setStrokeColor(colors.black)
    c.line(1 * inch, 10.4 * inch, 7.5 * inch, 10.4 * inch)


def export_master_schedule_pdf(path: str, schedule_df: pd.DataFrame, rooms_df: pd.DataFrame):
    c = canvas.Canvas(path, pagesize=LETTER)
    _canvas_header(c, "Master Exam Schedule")

    y = 10.1 * inch
    c.setFont("Helvetica", 9)

    # Merge schedule with rooming (many-to-many)
    # We'll pass in a pre-merged table from caller for clarity
    for _, row in schedule_df.iterrows():
        text = f"{row['day']} {row['start']}-{row['end']}  •  {row['exam_id']}  •  {row.get('room_id','')} ({row.get('building','')}-{row.get('room_number','')})  • cap {row.get('capacity','')}"
        if y < 1 * inch:
            c.showPage()
            _canvas_header(c, "Master Exam Schedule (cont.)")
            y = 10.1 * inch
            c.setFont("Helvetica", 9)
        c.drawString(0.7 * inch, y, text)
        y -= 0.2 * inch

    c.save()


def export_student_schedule_pdfs(dir_path: str, students_df: pd.DataFrame, enroll_df: pd.DataFrame, schedule_df: pd.DataFrame, rooming_df: pd.DataFrame):
    # Build lookup of exam -> (day,start,end)
    sched_map = schedule_df.set_index("exam_id")["period_id"].to_dict()
    pid_to_time = schedule_df.drop_duplicates(["period_id"]).set_index("period_id")["day"].to_dict()
    pid_to_start = schedule_df.drop_duplicates(["period_id"]).set_index("period_id")["start"].to_dict()
    pid_to_end = schedule_df.drop_duplicates(["period_id"]).set_index("period_id")["end"].to_dict()

    # For rooms: exam may span multiple rooms; show first room id for simplicity
    exam_first_room = rooming_df.groupby(["period_id", "exam_id"]).first().reset_index()
    exam_first_room = exam_first_room.set_index(["exam_id"])

    grouped = enroll_df.groupby("student_id")["course_id"].apply(list)

    for row in students_df.itertuples(index=False):
        sid = row.student_id
        courses = grouped.get(sid, [])
        c = canvas.Canvas(os.path.join(dir_path, f"{sid}.pdf"), pagesize=LETTER)
        _canvas_header(c, f"Exam Schedule — Student {sid}")
        c.setFont("Helvetica", 10)
        y = 10.0 * inch
        if not courses:
            c.drawString(1 * inch, y, "No exams assigned.")
            c.save()
            continue
        for exam in courses:
            pid = sched_map.get(exam)
            if pid is None:
                line = f"{exam}: (Unscheduled)"
            else:
                day = pid_to_time.get(pid, "?")
                stime = pid_to_start.get(pid, "?")
                etime = pid_to_end.get(pid, "?")
                try:
                    rrow = exam_first_room.loc[exam]
                    room = f"{rrow['room_id']} ({rrow['building']}-{rrow['room_number']})"
                except Exception:
                    room = "TBA"
                line = f"{exam}: {day} {stime}-{etime} — {room}"
            if y < 1 * inch:
                c.showPage()
                _canvas_header(c, f"Exam Schedule — Student {sid} (cont.)")
                c.setFont("Helvetica", 10)
                y = 10.0 * inch
            c.drawString(0.8 * inch, y, line)
            y -= 0.24 * inch
        c.save()


# ------------------------------
# Streamlit UI
# ------------------------------

def main():
    st.title("🧪 Exam Timetabling Simulator — Graph Coloring (DSATUR / Greedy / SA)")
    with st.sidebar:
        st.header("Configuration")
        n_students = st.number_input("Number of students", min_value=100, max_value=200000, value=10000, step=100)
        n_courses = st.number_input("Number of courses", min_value=10, max_value=2000, value=100, step=5)
        n_rooms = st.number_input("Number of rooms", min_value=10, max_value=2000, value=50, step=5)
        n_periods = st.number_input("Number of periods", min_value=10, max_value=500, value=80, step=5)
        min_cps, max_cps = st.slider("Courses per student (min, max)", 1, 10, (3, 6))
        seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=SEED_DEFAULT)

        st.divider()
        algo = st.selectbox("Scheduling algorithm", ["DSATUR", "Greedy", "SA (refine DSATUR)"])
        st.caption("SA starts from DSATUR and attempts to compress colors while avoiding conflicts.")
        with st.expander("SA Hyperparameters"):
            T0 = st.number_input("T₀", min_value=0.01, max_value=100.0, value=1.0, step=0.1)
            alpha = st.number_input("α (decay)", min_value=0.90, max_value=0.9999, value=0.995, step=0.001)
            max_iters = st.number_input("Max iters", min_value=1000, max_value=500000, value=40000, step=1000)
            penalty = st.number_input("Conflict/overflow penalty", min_value=1.0, max_value=1000.0, value=25.0, step=1.0)
            time_limit = st.number_input("Time limit (s)", min_value=1.0, max_value=120.0, value=8.0, step=0.5)
            target_colors = st.number_input("Target colors (<= periods)", min_value=1, max_value=2000, value=60)

        st.divider()
        gen_master_pdf = st.checkbox("Generate Master Schedule PDF", value=True)
        gen_student_pdfs = st.checkbox("Generate Per-Student PDFs (10k = heavy)", value=False)

        st.divider()
        run_btn = st.button("🚀 Generate Dataset & Run Scheduler", type="primary")

    # Placeholders
    status = st.empty()
    prog = st.empty()
    t0_all = time.perf_counter()

    if run_btn:
        out_dir, pdf_dir, stu_pdf_dir = make_output_dir()

        # 1) Generate dataset
        status.info("Generating synthetic dataset…")
        t0 = time.perf_counter()
        students_df, courses_df, rooms_df, periods_df, enroll_df, stu_text = generate_dataset(
            n_students, n_courses, n_rooms, n_periods, min_cps, max_cps, seed
        )
        t1 = time.perf_counter()
        status.success(f"Dataset ready in {t1 - t0:.2f}s — {len(students_df)} students, {len(courses_df)} courses, {len(rooms_df)} rooms, {len(periods_df)} periods.")

        # Save base CSVs
        save_csv(students_df, os.path.join(out_dir, "students.csv"))
        save_csv(courses_df, os.path.join(out_dir, "courses.csv"))
        save_csv(rooms_df, os.path.join(out_dir, "classrooms.csv"))
        save_csv(periods_df, os.path.join(out_dir, "timeslots.csv"))
        save_csv(enroll_df, os.path.join(out_dir, "enrollments.csv"))
        with open(os.path.join(out_dir, "enrollments.stu"), "w") as f:
            f.write(stu_text)

        # 2) Build conflict graph
        status.info("Constructing conflict graph…")
        t0 = time.perf_counter()
        G, density, n_edges = build_conflict_graph(enroll_df)
        t1 = time.perf_counter()
        status.success(f"Graph: |V|={G.number_of_nodes()} |E|={n_edges} density={density:.4f} in {t1 - t0:.2f}s")

        # 3) Coloring
        if algo == "DSATUR":
            status.info("Coloring with DSATUR…")
            t0 = time.perf_counter()
            coloring, ncolors = color_with_dsatur(G)
            t1 = time.perf_counter()
            status.success(f"DSATUR used {ncolors} periods in {t1 - t0:.2f}s")
        elif algo == "Greedy":
            status.info("Coloring with Greedy (largest_first)…")
            t0 = time.perf_counter()
            coloring, ncolors = color_with_greedy(G)
            t1 = time.perf_counter()
            status.success(f"Greedy used {ncolors} periods in {t1 - t0:.2f}s")
        else:
            status.info("Coloring with DSATUR, then Simulated Annealing refine…")
            t0 = time.perf_counter()
            base_coloring, base_k = color_with_dsatur(G)
            # Ensure target <= provided periods
            tgt = min(int(target_colors), int(n_periods))
            refined, k2, score, sa_time = anneal_refine(
                G,
                base_coloring,
                base_k,
                target_colors=tgt,
                T0=T0,
                alpha=alpha,
                max_iters=max_iters,
                conflict_penalty=penalty,
                time_limit_s=time_limit,
            )
            coloring, ncolors = refined, k2
            t1 = time.perf_counter()
            status.success(f"SA refined from {base_k} → {ncolors} colors in {t1 - t0:.2f}s (SA {sa_time:.2f}s)")

        # 4) Map to periods (auto-extend if needed)
        status.info("Mapping colors to time periods…")
        schedule_df, periods_df2 = map_colors_to_periods(coloring, periods_df.copy())
        if len(periods_df2) > len(periods_df):
            st.warning(f"Not enough periods provided; auto-extended to {len(periods_df2)} to fit coloring ({ncolors}).")
        # Save schedule CSV
        save_csv(schedule_df, os.path.join(out_dir, "schedule_generated.csv"))

        # 5) Room assignment (allow multi-room splits)
        status.info("Assigning rooms…")
        rooming_df, unassigned_df = room_assignment(schedule_df, rooms_df, courses_df[["course_id", "exam_size"]].rename(columns={"course_id":"course_id"}))
        save_csv(rooming_df, os.path.join(out_dir, "rooming_generated.csv"))
        if not unassigned_df.empty:
            save_csv(unassigned_df, os.path.join(out_dir, "unassigned.csv"))
            st.error(f"{len(unassigned_df)} exam placements ran out of rooms (some students unseated). See unassigned.csv")

        # Merge for display & master PDF
        sched_for_display = schedule_df.merge(
            rooming_df[["period_id","exam_id","room_id","building","room_number","capacity"]],
            on=["period_id","exam_id"], how="left"
        ).sort_values(["day","start","exam_id"]).reset_index(drop=True)

        # 6) PDFs
        if gen_master_pdf:
            status.info("Rendering Master Schedule PDF…")
            export_master_schedule_pdf(
                os.path.join(pdf_dir, "Master_Exam_Schedule.pdf"),
                sched_for_display,
                rooms_df,
            )
        if gen_student_pdfs:
            status.info("Rendering Per-Student PDFs (this can take a while)…")
            export_student_schedule_pdfs(
                os.path.join(pdf_dir, "student_schedules"),
                students_df,
                enroll_df,
                schedule_df,
                rooming_df,
            )

        # 7) Summary
        total_conflicts = objective_conflicts(G, coloring)
        periods_used = ncolors
        runtime_total = time.perf_counter() - t0_all
        summary_lines = [
            f"Algorithm: {algo}",
            f"Students: {len(students_df)}  Courses: {len(courses_df)}  Rooms: {len(rooms_df)}",
            f"Graph edges: {G.number_of_edges()}  Density: {nx.density(G):.6f}",
            f"Periods requested: {n_periods}  Used by coloring: {periods_used}  Actual periods table: {len(periods_df2)}",
            f"Rooming rows: {len(rooming_df)}  Unassigned rows: {0 if unassigned_df.empty else len(unassigned_df)}",
            f"Coloring conflicts on edges (should be 0 for DSATUR/Greedy): {total_conflicts}",
            f"Runtime (end-to-end): {runtime_total:.2f}s",
        ]
        with open(os.path.join(out_dir, "summary.txt"), "w") as f:
            f.write("\n".join(summary_lines))

        st.success("Scheduling complete. Outputs saved.")

        # Downloads
        colA, colB, colC, colD = st.columns(4)
        with colA:
            with open(os.path.join(out_dir, "schedule_generated.csv"), "rb") as f:
                st.download_button("Download schedule_generated.csv", f, file_name="schedule_generated.csv")
        with colB:
            with open(os.path.join(out_dir, "rooming_generated.csv"), "rb") as f:
                st.download_button("Download rooming_generated.csv", f, file_name="rooming_generated.csv")
        with colC:
            with open(os.path.join(out_dir, "enrollments.csv"), "rb") as f:
                st.download_button("Download enrollments.csv", f, file_name="enrollments.csv")
        with colD:
            with open(os.path.join(out_dir, "enrollments.stu"), "rb") as f:
                st.download_button("Download enrollments.stu", f, file_name="enrollments.stu")

        colE, colF = st.columns(2)
        if gen_master_pdf and os.path.exists(os.path.join(pdf_dir, "Master_Exam_Schedule.pdf")):
            with colE:
                with open(os.path.join(pdf_dir, "Master_Exam_Schedule.pdf"), "rb") as f:
                    st.download_button("Master_Exam_Schedule.pdf", f, file_name="Master_Exam_Schedule.pdf")
        if gen_student_pdfs:
            with colF:
                st.info(f"Student PDFs saved under {os.path.join(out_dir, 'pdfs', 'student_schedules')} (too many to bundle here).")

        st.subheader("Summary")
        st.code("\n".join(summary_lines))

        st.subheader("Scheduled Exams (sample)")
        st.dataframe(sched_for_display.head(200), use_container_width=True)


if __name__ == "__main__":
    main()
