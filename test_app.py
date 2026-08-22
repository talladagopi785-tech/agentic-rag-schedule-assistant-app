from datetime import date, timedelta
from app import parse_date, parse_time, in_window, load_events, handle_chat

def test_date_parsing():
    t = date.today()
    assert parse_date("today") == t
    assert parse_date("tomorrow") == t + timedelta(days=1)

def test_time_parsing():
    assert parse_time("3 PM") == "15:00"
    assert parse_time("10:30 AM") == "10:30"

def test_sample_data():
    events = load_events()
    assert len(events) >= 8
    assert all("title" in e and "date" in e for e in events)

def test_chat_retrieval():
    result = handle_chat("What do I have scheduled tomorrow?")
    assert result.tool_used == "get_schedule"
