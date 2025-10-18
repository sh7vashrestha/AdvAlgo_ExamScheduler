import csv
import io
import os
from typing import Dict, List, Tuple, Iterable, Set, Union, IO
from collections import defaultdict

from .models import Exam, Room, Period

TextOrPath = Union[str, os.PathLike, IO]


def _open_text(src: TextOrPath):
    """Return a text-mode file handle and a flag indicating whether to close it.

    Accepts a filesystem path, a text IO object, or a BytesIO buffer.
    Ensures CSV readers get strings, not bytes.
    """
    # Path-like: open in text mode
    if isinstance(src, (str, os.PathLike)):
        f = open(src, 'r', newline='')
        return f, True
    # BytesIO -> wrap in TextIOWrapper
    if isinstance(src, io.BytesIO):
        try:
            src.seek(0)
        except Exception:
            pass
        f = io.TextIOWrapper(src, encoding='utf-8', newline='')
        return f, True
    # Assume it's already a text IO (e.g., StringIO or file object)
    if hasattr(src, 'read'):
        try:
            src.seek(0)
        except Exception:
            pass
        return src, False
    raise TypeError("Unsupported input type; expected path or file-like object")


def load_rooms(src: TextOrPath) -> Dict[str, Room]:
    rooms: Dict[str, Room] = {}
    f, should_close = _open_text(src)
    try:
        r = csv.DictReader(f)
        for row in r:
            rid = str(row['id']).strip()
            cap = int(row['capacity'])
            rooms[rid] = Room(id=rid, capacity=cap)
    finally:
        if should_close:
            f.close()
    return rooms


def load_periods(src: TextOrPath) -> Dict[str, Period]:
    periods: Dict[str, Period] = {}
    f, should_close = _open_text(src)
    try:
        r = csv.DictReader(f)
        for row in r:
            pid = str(row['id']).strip()
            duration = int(row.get('duration_minutes', row.get('duration_min', 120)))
            periods[pid] = Period(
                id=pid,
                start=row.get('start'),
                end=row.get('end'),
                duration_min=duration,
            )
    finally:
        if should_close:
            f.close()
    return periods


def load_exam_durations(src: TextOrPath) -> Dict[str, int]:
    """Optional CSV file exam_id,duration_minutes."""
    durations: Dict[str, int] = {}
    f, should_close = _open_text(src)
    try:
        r = csv.DictReader(f)
        for row in r:
            durations[str(row['exam_id']).strip()] = int(row['duration_minutes'])
    finally:
        if should_close:
            f.close()
    return durations


def load_toronto_stu(src: TextOrPath):
    """Return (student->set(exams)), and inferred enrollment counts per exam."""
    enroll_per_exam: Dict[str, int] = defaultdict(int)
    students: Dict[str, Set[str]] = {}
    f, should_close = _open_text(src)
    try:
        idx = 0
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            exams = line.replace('\t', ' ').split()
            sid = f"stu_{idx}"
            idx += 1
            students[sid] = set(exams)
            for ex in exams:
                enroll_per_exam[ex] += 1
    finally:
        if should_close:
            f.close()
    return students, dict(enroll_per_exam)


def load_edge_list_csv(src: TextOrPath):
    edges: List[Tuple[str, str]] = []
    f, should_close = _open_text(src)
    try:
        reader = csv.reader(f)
        for row in reader:
            if not row or (isinstance(row[0], str) and row[0].startswith('#')):
                continue
            if len(row) >= 2:
                u, v = str(row[0]).strip(), str(row[1]).strip()
                if u != v:
                    edges.append((u, v))
    finally:
        if should_close:
            f.close()
    return edges


def save_schedule_csv(path: str, schedule_map: Dict[str, str]):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['exam_id', 'period_id'])
        for exam_id, pid in sorted(schedule_map.items()):
            w.writerow([exam_id, pid])


def save_rooming_csv(path: str, rooming: Dict[str, List[Tuple[str, str]]]):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['period_id', 'exam_id', 'room_id'])
        for pid, pairs in rooming.items():
            for exam_id, room_id in pairs:
                w.writerow([pid, exam_id, room_id])
