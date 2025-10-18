from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple

@dataclass
class Exam:
    id: str
    duration_min: int = 120
    enrollment: int = 0  # number of students taking it

@dataclass
class Room:
    id: str
    capacity: int

@dataclass
class Period:
    id: str
    start: Optional[str] = None  # ISO-like string for display
    end: Optional[str] = None
    duration_min: int = 120

@dataclass
class Schedule:
    # timeslot assignment: exam_id -> period_id
    period_of: Dict[str, str] = field(default_factory=dict)
    # room assignment per period: period_id -> list of (exam_id, room_id)
    rooming: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)
