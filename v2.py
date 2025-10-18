#!/usr/bin/env python3
"""
=====================================================
Generate ExamTime-Compatible Files from schedule.csv
=====================================================

INPUT:
  - classrooms.csv
  - timeslots.csv
  - courses.csv
  - students.csv
  - schedule.csv

OUTPUT:
  data/generated/
      ├── rooms.csv
      ├── periods.csv
      ├── durations.csv
      ├── enrollments.stu
      └── summary.txt
"""

import os
import pandas as pd
import random
from datetime import datetime

# -----------------------------
# CONFIGURATION
# -----------------------------
BASE_PATH = "."
OUT_PATH = os.path.join(BASE_PATH, "data", "generated")
os.makedirs(OUT_PATH, exist_ok=True)
random.seed(42)

# -----------------------------
# STEP 1: Load CSVs
# -----------------------------
def load_csv(name):
    path = os.path.join(BASE_PATH, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Missing required file: {name}")
    return pd.read_csv(path)

print("🔍 Loading input CSVs ...")
classrooms = load_csv("./data/classrooms.csv")
timeslots = load_csv("./data/timeslots.csv")
courses = load_csv("./data/courses.csv")
students = load_csv("./data/students.csv")
schedule = load_csv("./data/schedule.csv")
print("✅ All input CSVs loaded successfully.")

# -----------------------------
# STEP 2: Generate rooms.csv
# -----------------------------
rooms_df = classrooms.rename(columns={
    "classroom_id": "id",
    "capacity": "capacity"
})[["id", "capacity"]]

rooms_df.to_csv(os.path.join(OUT_PATH, "rooms.csv"), index=False)
print(f"🏫 rooms.csv saved ({len(rooms_df)} rooms)")

# -----------------------------
# STEP 3: Clean timeslots → periods.csv
# -----------------------------
def parse_time_safe(t):
    try:
        return datetime.strptime(str(t).strip(), "%H:%M")
    except Exception:
        return None

periods = []
for _, r in timeslots.iterrows():
    start = parse_time_safe(r["start_time"])
    end = parse_time_safe(r["end_time"])
    if not start or not end:
        continue
    if end <= start:  # fix reversed
        start, end = end, start
    dur = int((end - start).seconds / 60)
    if dur < 15 or dur > 600:
        continue
    pid = f"P{r['timeslot_id']}"
    periods.append({
        "id": pid,
        "start": f"2025-12-{(int(r['timeslot_id']) % 10)+10:02d}T{start.strftime('%H:%M')}",
        "end": f"2025-12-{(int(r['timeslot_id']) % 10)+10:02d}T{end.strftime('%H:%M')}",
        "duration_minutes": dur
    })

periods_df = pd.DataFrame(periods)
periods_df.to_csv(os.path.join(OUT_PATH, "periods.csv"), index=False)
print(f"⏰ periods.csv saved ({len(periods_df)} valid slots)")

# -----------------------------
# STEP 4: Generate durations.csv
# -----------------------------
durations_df = pd.DataFrame({
    "exam_id": [f"E{cid}" for cid in courses["course_id"]],
    "duration_minutes": [random.choice([60, 90, 120, 180]) for _ in range(len(courses))]
})
durations_df.to_csv(os.path.join(OUT_PATH, "durations.csv"), index=False)
print(f"⌛ durations.csv saved ({len(durations_df)} exams)")

# -----------------------------
# STEP 5: Build enrollments.stu from schedule.csv
# -----------------------------
enrollments = schedule.groupby("student_id")["course_id"].apply(list).reset_index()

stu_lines = []
for _, row in enrollments.iterrows():
    courses_list = [str(c) for c in row["course_id"]]
    stu_lines.append(" ".join(courses_list))

stu_path = os.path.join(OUT_PATH, "enrollments.stu")
with open(stu_path, "w") as f:
    f.write("\n".join(stu_lines))

print(f"🧑‍🎓 enrollments.stu saved ({len(stu_lines)} students)")

# -----------------------------
# STEP 6: Summary
# -----------------------------
summary = f"""
ExamTime-Compatible Dataset Generated 🎓
----------------------------------------
Students:   {len(students)}
Courses:    {len(courses)}
Rooms:      {len(rooms_df)}
Periods:    {len(periods_df)}
Exams:      {len(durations_df)}
Enrollments:{len(stu_lines)}
Output Path:{os.path.abspath(OUT_PATH)}
"""
with open(os.path.join(OUT_PATH, "summary.txt"), "w") as f:
    f.write(summary)
print(summary)
print("✅ All files successfully generated for ExamTime!")
