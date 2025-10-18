#!/usr/bin/env python3
"""
Full Dataset Builder for Exam Scheduling
========================================
Generates a complete synthetic exam dataset from base inputs:
students.csv, courses.csv, classrooms.csv, timeslots.csv

Outputs:
  - enrollments.csv
  - exam_schedule.csv
  - student_schedule.csv
  - enrollments.stu
  - conflict_edges.csv
  - summary.txt
"""

import os
import random
import pandas as pd
from itertools import combinations
from datetime import datetime

# -----------------------------
# CONFIGURATION
# -----------------------------
BASE_PATH = "data"
OUTPUT_PATH = os.path.join(BASE_PATH, "full_dataset")
os.makedirs(OUTPUT_PATH, exist_ok=True)

RANDOM_SEED = 42
MIN_COURSES = 3
MAX_COURSES = 6
random.seed(RANDOM_SEED)


# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------
def parse_time(t):
    """Parse HH:MM string safely."""
    try:
        return datetime.strptime(str(t).strip(), "%H:%M")
    except Exception:
        return None


def clean_timeslots(df):
    """Fix invalid or swapped start/end times and drop duplicates."""
    cleaned = []
    seen = set()
    for _, r in df.iterrows():
        start = parse_time(r["start_time"])
        end = parse_time(r["end_time"])
        if not start or not end:
            continue
        # Swap if needed
        if end <= start:
            start, end = end, start
        duration = (end - start).seconds / 3600
        if duration <= 0:
            continue
        tup = (r["day"], start.strftime("%H:%M"), end.strftime("%H:%M"))
        if tup in seen:
            continue
        seen.add(tup)
        cleaned.append({
            "timeslot_id": len(cleaned) + 1,
            "day": r["day"],
            "start_time": start.strftime("%H:%M"),
            "end_time": end.strftime("%H:%M"),
            "duration_hrs": duration
        })
    return pd.DataFrame(cleaned)


# -----------------------------
# STEP 1: LOAD BASE DATA
# -----------------------------
students = pd.read_csv(os.path.join(BASE_PATH, "students.csv"))
courses = pd.read_csv(os.path.join(BASE_PATH, "courses.csv"))
classrooms = pd.read_csv(os.path.join(BASE_PATH, "classrooms.csv"))
timeslots = pd.read_csv(os.path.join(BASE_PATH, "timeslots.csv"))
timeslots = clean_timeslots(timeslots)

print(f"✅ Loaded: {len(students)} students, {len(courses)} courses, "
      f"{len(classrooms)} rooms, {len(timeslots)} valid timeslots.")


# -----------------------------
# STEP 2: GENERATE ENROLLMENTS
# -----------------------------
enrollments = []
for _, s in students.iterrows():
    n = random.randint(MIN_COURSES, MAX_COURSES)
    chosen = random.sample(list(courses["course_id"]), n)
    for cid in chosen:
        enrollments.append({"student_id": s["student_id"], "course_id": cid})

enroll_df = pd.DataFrame(enrollments)
enroll_df.to_csv(f"{OUTPUT_PATH}/enrollments.csv", index=False)
print(f"📘 Enrollments created: {len(enroll_df)} records")


# -----------------------------
# STEP 3: BUILD EXAM SCHEDULE
# -----------------------------
used_room_slots = set()
exam_schedule = []

for _, course in courses.iterrows():
    tries = 0
    while tries < 200:
        room = classrooms.sample(1).iloc[0]
        slot = timeslots.sample(1).iloc[0]
        key = (room["classroom_id"], slot["timeslot_id"])
        if key not in used_room_slots:
            used_room_slots.add(key)
            break
        tries += 1

    enrolled = enroll_df.query("course_id == @course.course_id").shape[0]
    exam_schedule.append({
        "course_id": course["course_id"],
        "course_name": course["course_name"],
        "department": course["department"],
        "classroom_id": room["classroom_id"],
        "room_capacity": room["capacity"],
        "enrolled_students": enrolled,
        "day": slot["day"],
        "start_time": slot["start_time"],
        "end_time": slot["end_time"],
        "timeslot_id": slot["timeslot_id"]
    })

exam_df = pd.DataFrame(exam_schedule)
exam_df.to_csv(f"{OUTPUT_PATH}/exam_schedule.csv", index=False)
print(f"🗓️  Exam schedule created: {len(exam_df)} courses assigned.")


# -----------------------------
# STEP 4: STUDENT SCHEDULE
# -----------------------------
student_sched = (
    enroll_df.groupby("student_id")["course_id"]
    .apply(lambda x: list(map(str, x)))
    .reset_index()
)
student_sched["course_count"] = student_sched["course_id"].apply(len)
student_sched["courses"] = student_sched["course_id"].apply(lambda x: " ".join(x))
student_sched.drop(columns=["course_id"], inplace=True)
student_sched.to_csv(f"{OUTPUT_PATH}/student_schedule.csv", index=False)
print("📋 Student schedule summary generated.")


# -----------------------------
# STEP 5: .STU + Conflict Graph
# -----------------------------
# .stu file
with open(f"{OUTPUT_PATH}/enrollments.stu", "w") as f:
    for row in student_sched["courses"]:
        f.write(row + "\n")
print("📘 Created enrollments.stu for graph-coloring input.")

# Conflict edges
edges = set()
for _, g in enroll_df.groupby("student_id"):
    for a, b in combinations(sorted(g["course_id"].unique()), 2):
        edges.add((a, b))
edges_df = pd.DataFrame(list(edges), columns=["course1", "course2"])
edges_df.to_csv(f"{OUTPUT_PATH}/conflict_edges.csv", index=False)
print(f"⚡ Conflict graph built with {len(edges_df)} edges.")


# -----------------------------
# STEP 6: SUMMARY FILE
# -----------------------------
summary = f"""
==== Exam Scheduling Synthetic Dataset ====
Students:          {len(students)}
Courses:           {len(courses)}
Classrooms:        {len(classrooms)}
Valid Timeslots:   {len(timeslots)}
Enrollments:       {len(enroll_df)}
Conflicts:         {len(edges_df)}

Average Courses/Student: {enroll_df.groupby('student_id').size().mean():.2f}
Max Room Capacity:       {classrooms['capacity'].max()}
Min Room Capacity:       {classrooms['capacity'].min()}
-------------------------------------------
Output Directory: {os.path.abspath(OUTPUT_PATH)}
"""
with open(f"{OUTPUT_PATH}/summary.txt", "w") as f:
    f.write(summary)
print(summary)
