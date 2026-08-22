from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Literal

import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
SCHEDULE_FILE = DATA_DIR / "schedule.json"
CHROMA_DIR = APP_DIR / "chroma_db"
STATIC_DIR = APP_DIR / "static"
LOCK = threading.RLock()

# Keep the embedding deliberately lightweight so the service stays suitable
# for small/free instances. It is a deterministic vector embedding used by
# ChromaDB; no large ML model is downloaded at deploy time.
EMBED_DIM = 128
WINDOW_DAYS = 30


def embed_text(text: str) -> list[float]:
    """Create a deterministic normalized hash embedding."""
    vec = [0.0] * EMBED_DIM
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if not tokens:
        return vec
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for offset in (0, 4, 8):
            idx = int.from_bytes(digest[offset:offset + 4], "little") % EMBED_DIM
            sign = 1.0 if digest[offset + 12] % 2 == 0 else -1.0
            vec[idx] += sign
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec] if norm else vec


class Event(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    event_type: Literal["meeting", "workshop", "task", "appointment", "other"] = "other"
    date: str
    start_time: str
    end_time: str
    location: str = ""
    notes: str = ""


class UpdateRequest(BaseModel):
    action: Literal["add", "update", "remove"]
    event: Event | None = None
    event_id: str | None = None
    changes: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    message: str


class AgentResponse(BaseModel):
    answer: str
    tool_used: str
    events: list[Event] = Field(default_factory=list)


def today() -> date:
    return datetime.now().date()


def parse_date(text: str, reference: date | None = None) -> date | None:
    ref = reference or today()
    low = text.lower()

    if "today" in low:
        return ref
    if "tomorrow" in low:
        return ref + timedelta(days=1)
    if "day after tomorrow" in low:
        return ref + timedelta(days=2)

    weekdays = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
    }
    for name, idx in weekdays.items():
        if re.search(rf"\b{re.escape(name)}\b", low):
            delta = (idx - ref.weekday()) % 7
            # "this Friday" means today if today is Friday; otherwise next occurrence.
            return ref + timedelta(days=delta)

    m = re.search(
        r"\b(january|february|march|april|may|june|july|august|september|"
        r"october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s*,?\s*(\d{4}))?",
        low,
    )
    if m:
        months = {
            "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
            "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
            "november": 11, "december": 12,
        }
        year = int(m.group(3) or ref.year)
        month, day_num = months[m.group(1)], int(m.group(2))
        try:
            result = date(year, month, day_num)
        except ValueError:
            return None
        # If no year was given and the date has already passed, treat it as next year.
        if not m.group(3) and result < ref:
            result = date(ref.year + 1, month, day_num)
        return result

    iso = re.search(r"\b(20\d{2}-\d{1,2}-\d{1,2})\b", low)
    if iso:
        try:
            return date.fromisoformat(iso.group(1))
        except ValueError:
            return None
    return None


def parse_time(text: str, default: str = "09:00") -> str:
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", text.lower())
    if not m:
        return default
    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    meridiem = m.group(3)
    if meridiem == "pm" and hour != 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    if hour > 23 or minute > 59:
        raise ValueError("Invalid time.")
    return f"{hour:02d}:{minute:02d}"


def normalize_event(event: Event) -> Event:
    try:
        datetime.strptime(event.date, "%Y-%m-%d")
        datetime.strptime(event.start_time, "%H:%M")
        datetime.strptime(event.end_time, "%H:%M")
    except ValueError as exc:
        raise ValueError("date must be YYYY-MM-DD and time must be HH:MM.") from exc
    if event.end_time <= event.start_time:
        raise ValueError("end_time must be later than start_time.")
    return event


def in_window(event_date: date) -> bool:
    start = today()
    return start <= event_date <= start + timedelta(days=WINDOW_DAYS)


def sample_events() -> list[dict[str, Any]]:
    base = today()

    def d(offset: int) -> str:
        return (base + timedelta(days=offset)).isoformat()

    return [
        {"id": str(uuid.uuid4()), "title": "Team Stand-up", "event_type": "meeting",
         "date": d(1), "start_time": "09:30", "end_time": "10:00",
         "location": "Online", "notes": "Daily engineering sync."},
        {"id": str(uuid.uuid4()), "title": "RAG Project Workshop", "event_type": "workshop",
         "date": d(3), "start_time": "11:00", "end_time": "13:00",
         "location": "Lab 2", "notes": "Build and test retrieval pipeline."},
        {"id": str(uuid.uuid4()), "title": "Submit Project Report", "event_type": "task",
         "date": d(5), "start_time": "16:00", "end_time": "17:00",
         "location": "", "notes": "Final PDF submission."},
        {"id": str(uuid.uuid4()), "title": "Doctor Appointment", "event_type": "appointment",
         "date": d(7), "start_time": "10:30", "end_time": "11:30",
         "location": "City Clinic", "notes": "Bring previous reports."},
        {"id": str(uuid.uuid4()), "title": "Client Meeting", "event_type": "meeting",
         "date": d(9), "start_time": "14:00", "end_time": "15:00",
         "location": "Online", "notes": "Discuss project milestone."},
        {"id": str(uuid.uuid4()), "title": "Python Workshop", "event_type": "workshop",
         "date": d(12), "start_time": "10:00", "end_time": "12:00",
         "location": "Seminar Hall", "notes": "FastAPI and deployment."},
        {"id": str(uuid.uuid4()), "title": "Database Design Task", "event_type": "task",
         "date": d(16), "start_time": "15:00", "end_time": "16:30",
         "location": "", "notes": "Review ChromaDB schema."},
        {"id": str(uuid.uuid4()), "title": "Project Review", "event_type": "meeting",
         "date": d(20), "start_time": "13:00", "end_time": "14:00",
         "location": "Online", "notes": "Demo the schedule assistant."},
        {"id": str(uuid.uuid4()), "title": "Career Appointment", "event_type": "appointment",
         "date": d(24), "start_time": "11:00", "end_time": "12:00",
         "location": "Career Center", "notes": "Resume review."},
        {"id": str(uuid.uuid4()), "title": "Release Planning", "event_type": "meeting",
         "date": d(28), "start_time": "15:30", "end_time": "16:30",
         "location": "Online", "notes": "Plan the next release."},
    ]


def load_events() -> list[dict[str, Any]]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not SCHEDULE_FILE.exists():
        events = sample_events()
        SCHEDULE_FILE.write_text(json.dumps(events, indent=2), encoding="utf-8")
        return events
    try:
        return json.loads(SCHEDULE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        events = sample_events()
        SCHEDULE_FILE.write_text(json.dumps(events, indent=2), encoding="utf-8")
        return events


def save_events(events: list[dict[str, Any]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SCHEDULE_FILE.write_text(json.dumps(events, indent=2), encoding="utf-8")


class ScheduleRAG:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            self.collection = self.client.get_or_create_collection(
                name="schedule",
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            self.client.delete_collection("schedule")
            self.collection = self.client.get_or_create_collection(
                name="schedule",
                metadata={"hnsw:space": "cosine"},
            )
        self.rebuild()

    def rebuild(self) -> None:
        with LOCK:
            ids = self.collection.get().get("ids", [])
            if ids:
                self.collection.delete(ids=ids)
            events = load_events()
            if not events:
                return
            ids = [e["id"] for e in events]
            docs = [self.event_text(e) for e in events]
            embeddings = [embed_text(x) for x in docs]
            self.collection.add(ids=ids, documents=docs, embeddings=embeddings)

    @staticmethod
    def event_text(e: dict[str, Any]) -> str:
        return (
            f"{e['title']} {e['event_type']} {e['date']} "
            f"{e['start_time']} {e['end_time']} {e.get('location', '')} {e.get('notes', '')}"
        )

    def search(self, query: str, n: int = 8) -> list[dict[str, Any]]:
        events = load_events()
        if not events:
            return []
        result = self.collection.query(
            query_embeddings=[embed_text(query)],
            n_results=min(n, len(events)),
            include=["documents", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        by_id = {e["id"]: e for e in events}
        return [by_id[event_id] for event_id in ids if event_id in by_id]


rag = ScheduleRAG()


def get_schedule(query: str = "", target_date: date | None = None,
                 start: str | None = None, end: str | None = None) -> list[Event]:
    """Tool 1: retrieve relevant schedule details using ChromaDB RAG + filters."""
    with LOCK:
        events = load_events()
        if target_date:
            selected = [e for e in events if e["date"] == target_date.isoformat()]
        else:
            selected = rag.search(query or "schedule", n=10)

        if start or end:
            lo = start or "00:00"
            hi = end or "23:59"
            selected = [
                e for e in selected
                if e["start_time"] < hi and e["end_time"] > lo
            ]

        # Preserve chronological order for schedule/availability answers.
        selected.sort(key=lambda e: (e["date"], e["start_time"]))
        return [Event(**e) for e in selected]


def update_schedule(action: str, event: Event | None = None,
                    event_id: str | None = None,
                    changes: dict[str, Any] | None = None) -> Event | None:
    """Tool 2: add, update, or remove schedule entries."""
    with LOCK:
        events = load_events()
        changes = changes or {}

        if action == "add":
            if event is None:
                raise ValueError("event is required for add.")
            event = normalize_event(event)
            if not in_window(date.fromisoformat(event.date)):
                raise ValueError("Events must be within the next 30 days.")
            events.append(event.model_dump())
            save_events(events)
            rag.rebuild()
            return event

        if action == "update":
            if not event_id:
                raise ValueError("event_id is required for update.")
            found = next((e for e in events if e["id"] == event_id), None)
            if not found:
                raise ValueError("Event not found.")
            allowed = {"title", "event_type", "date", "start_time", "end_time", "location", "notes"}
            for key, value in changes.items():
                if key in allowed and value is not None:
                    found[key] = value
            updated = normalize_event(Event(**found))
            if not in_window(date.fromisoformat(updated.date)):
                raise ValueError("Events must be within the next 30 days.")
            save_events(events)
            rag.rebuild()
            return updated

        if action == "remove":
            if not event_id:
                raise ValueError("event_id is required for remove.")
            before = len(events)
            removed = next((e for e in events if e["id"] == event_id), None)
            events = [e for e in events if e["id"] != event_id]
            if len(events) == before:
                raise ValueError("Event not found.")
            save_events(events)
            rag.rebuild()
            return Event(**removed) if removed else None

        raise ValueError("Unknown action.")


def find_matching_event(message: str) -> Event | None:
    """Use RAG first, then lexical fallback for robust update matching."""
    candidates = get_schedule(message, n=8) if False else rag.search(message, n=8)
    low = message.lower()
    for e in candidates:
        if e["title"].lower() in low:
            return Event(**e)

    # Strong matching for time references such as "meeting from 2 PM".
    times = re.findall(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b", low)
    target_date = parse_date(message)
    for e in load_events():
        if target_date and e["date"] != target_date.isoformat():
            continue
        if times and e["start_time"] == parse_time(times[0]):
            return Event(**e)
        if e["event_type"] in low or e["title"].split()[0].lower() in low:
            return Event(**e)
    return None


def make_event_from_add(message: str) -> Event:
    target_date = parse_date(message)
    if not target_date:
        raise ValueError("Please include a date such as 'tomorrow' or 'August 25'.")
    if not in_window(target_date):
        raise ValueError("Please choose a date within the next 30 days.")

    start = parse_time(message, "09:00")
    # Use an explicit end time if the user gives "from X to Y" or "X-Y".
    time_matches = re.findall(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b", message.lower())
    end = start
    if len(time_matches) >= 2:
        end = parse_time(time_matches[1])
    else:
        h, m = map(int, start.split(":"))
        end_dt = datetime.combine(target_date, time(h, m)) + timedelta(hours=1)
        end = end_dt.strftime("%H:%M")

    title_match = re.search(
        r"(?:add|schedule|create|book)\s+(?:a|an)?\s*(meeting|workshop|task|appointment)\s*(?:called|named)?\s*(.*?)(?:\s+on\s+|\s+for\s+|\s+at\s+|\s+from\s+|\s+tomorrow\b|\s+today\b|$)",
        message,
        re.IGNORECASE,
    )
    if title_match:
        event_type = title_match.group(1).lower()
        tail = title_match.group(2).strip(" .")
        title = tail or event_type.title()
    else:
        event_type = "meeting" if "meeting" in message.lower() else "other"
        title = "New Schedule Event"

    return Event(
        title=title,
        event_type=event_type if event_type in {"meeting", "workshop", "task", "appointment"} else "other",
        date=target_date.isoformat(),
        start_time=start,
        end_time=end,
        notes="Added by the schedule assistant.",
    )


def format_event(e: Event) -> str:
    return f"{e.date} {e.start_time}-{e.end_time} — {e.title} ({e.event_type})"


def handle_chat(message: str) -> AgentResponse:
    low = message.lower().strip()

    # Agentic routing: mutations go to update_schedule; questions go to RAG retrieval.
    mutation_words = ("add ", "schedule ", "create ", "book ", "move ", "change ", "update ", "remove ", "delete ", "cancel ")
    if low.startswith(mutation_words):
        if low.startswith(("add ", "schedule ", "create ", "book ")):
            event = make_event_from_add(message)
            saved = update_schedule("add", event=event)
            return AgentResponse(
                answer=f"Added: {format_event(saved)}",
                tool_used="update_schedule",
                events=[saved],
            )

        if low.startswith(("remove ", "delete ", "cancel ")):
            found = find_matching_event(message)
            if not found:
                return AgentResponse(
                    answer="I could not identify the event to remove. Include its title, date, or time.",
                    tool_used="get_schedule",
                )
            removed = update_schedule("remove", event_id=found.id)
            return AgentResponse(
                answer=f"Removed: {format_event(removed)}",
                tool_used="update_schedule",
                events=[removed],
            )

        # Move/change/update: locate the event, then infer the new time/date.
        found = find_matching_event(message)
        if not found:
            return AgentResponse(
                answer="I could not identify the event to update. Include its title, date, or current time.",
                tool_used="get_schedule",
            )
        changes: dict[str, Any] = {}
        times = re.findall(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b", low)
        if len(times) >= 2:
            changes["start_time"] = parse_time(times[-1])
            h, m = map(int, changes["start_time"].split(":"))
            changes["end_time"] = (datetime.combine(today(), time(h, m)) + timedelta(hours=1)).strftime("%H:%M")
        elif "to " in low:
            candidate = low.split("to ", 1)[1]
            changes["start_time"] = parse_time(candidate, found.start_time)
            h, m = map(int, changes["start_time"].split(":"))
            changes["end_time"] = (datetime.combine(today(), time(h, m)) + timedelta(hours=1)).strftime("%H:%M")
        new_date = parse_date(message)
        if new_date and any(x in low for x in ("tomorrow", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december")):
            changes["date"] = new_date.isoformat()

        updated = update_schedule("update", event_id=found.id, changes=changes)
        return AgentResponse(
            answer=f"Updated: {format_event(updated)}",
            tool_used="update_schedule",
            events=[updated],
        )

    # Retrieval path.
    target_date = parse_date(message)
    if target_date:
        events = get_schedule(message, target_date=target_date)
    else:
        events = get_schedule(message)

    if "free" in low or "available" in low:
        # Afternoon defaults to 12:00-17:00; morning 08:00-12:00; evening 17:00-21:00.
        if "afternoon" in low:
            start, end = "12:00", "17:00"
        elif "morning" in low:
            start, end = "08:00", "12:00"
        elif "evening" in low:
            start, end = "17:00", "21:00"
        else:
            start, end = "09:00", "18:00"
        if target_date:
            busy = get_schedule(message, target_date=target_date, start=start, end=end)
        else:
            busy = []
        if not busy:
            return AgentResponse(
                answer=f"Yes. You are free during {start}-{end} on {target_date.strftime('%A, %B %d')}." if target_date else "No matching busy events were found.",
                tool_used="get_schedule",
                events=[],
            )
        details = "; ".join(format_event(e) for e in busy)
        return AgentResponse(
            answer=f"You have {len(busy)} event(s) during that time: {details}",
            tool_used="get_schedule",
            events=busy,
        )

    if not events:
        return AgentResponse(
            answer="I could not find any matching schedule entries.",
            tool_used="get_schedule",
            events=[],
        )

    if target_date:
        details = "\n".join(f"• {e.start_time}-{e.end_time}: {e.title} ({e.event_type})" for e in events)
        return AgentResponse(
            answer=f"Schedule for {target_date.strftime('%A, %B %d, %Y')}:\n{details}",
            tool_used="get_schedule",
            events=events,
        )

    details = "\n".join(f"• {format_event(e)}" for e in events[:8])
    return AgentResponse(
        answer=f"Here are the most relevant schedule entries:\n{details}",
        tool_used="get_schedule",
        events=events[:8],
    )


app = FastAPI(
    title="Agentic RAG Schedule Assistant",
    version="1.0.0",
    description="30-day schedule assistant with ChromaDB RAG and two agent tools.",
)


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "agentic-rag-schedule-assistant"}


@app.get("/api/schedule", response_model=list[Event])
def api_schedule(date_str: str | None = None) -> list[Event]:
    target = None
    if date_str:
        try:
            target = date.fromisoformat(date_str)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Use YYYY-MM-DD.") from exc
    return get_schedule(target_date=target, query=date_str or "schedule")


@app.post("/api/chat", response_model=AgentResponse)
def api_chat(req: ChatRequest) -> AgentResponse:
    try:
        return handle_chat(req.message)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/tools/get_schedule", response_model=list[Event])
def tool_get_schedule(req: ChatRequest) -> list[Event]:
    target = parse_date(req.message)
    return get_schedule(req.message, target_date=target)


@app.post("/tools/update_schedule", response_model=Event | None)
def tool_update_schedule(req: UpdateRequest) -> Event | None:
    try:
        return update_schedule(
            req.action,
            event=req.event,
            event_id=req.event_id,
            changes=req.changes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/reindex")
def reindex() -> dict[str, str]:
    rag.rebuild()
    return {"status": "ok", "message": "ChromaDB index rebuilt."}
