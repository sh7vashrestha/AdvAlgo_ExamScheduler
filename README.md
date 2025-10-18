# ExamTime – Room & Duration Aware Exam Timetabling

A Python toolkit and minimal UI for exam timetabling that:
- Builds a conflict graph from student enrollments (Toronto `.stu`) or an edge list
- Assigns **conflict-free periods** (timeslots) using **DSATUR** or **Greedy**
- Enforces **exam duration** feasibility vs. period durations
- Assigns **rooms** by capacity (greedy best-fit)
- Exports schedule and rooming CSVs
- Optional Streamlit UI for interactive runs

## Install
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

## Quick Start (CLI)
Toronto `.stu` + rooms + periods:
```bash
python main.py --stu path/to/yor-f-83.stu   --rooms sample_data/rooms.csv --periods sample_data/periods.csv   --algo dsatur --out_schedule schedule.csv --out_rooming rooming.csv
```

Edge list only (defaults to 10 x 120-min periods, random enrollments):
```bash
python main.py --edges path/to/edges.csv --algo greedy --out_schedule sch.csv
```

Synthetic graph:
```bash
python main.py --generate 200 --density 0.15 --algo dsatur --out_schedule sch.csv
```

Optional per-exam durations:
```bash
python main.py --stu path/to/file.stu --durations path/to/durations.csv   --rooms sample_data/rooms.csv --periods sample_data/periods.csv
```

## Data Formats
**rooms.csv**
```
id,capacity
R101,150
R102,120
R201,80
R202,60
R301,40
```

**periods.csv**
```
id,start,end,duration_minutes
P1,2025-12-10T09:00,2025-12-10T11:00,120
P2,2025-12-10T13:00,2025-12-10T15:00,120
P3,2025-12-11T09:00,2025-12-11T12:00,180
P4,2025-12-11T13:00,2025-12-11T15:00,120
```

**durations.csv** (optional)
```
exam_id,duration_minutes
E12,180
E37,90
```

## Streamlit UI
```bash
streamlit run app/streamlit_app.py
```
- Upload `.stu` or edge list, plus rooms/periods/durations CSVs as needed
- Choose algorithm and run
- Download `schedule.csv` and `rooming.csv`

## Notes
- This project focuses on a simple two-stage approach: **period assignment** (coloring) then **room assignment** (greedy). For stricter feasibility and better optimization (e.g., minimizing number of periods or balancing room usage), consider adding an **ILP/CP** layer (PuLP, OR-Tools).
- Toronto `.stu` files don’t include exam durations; use `--durations` to provide them if needed.
