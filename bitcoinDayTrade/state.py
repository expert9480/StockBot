from __future__ import annotations

import json
import os
import tempfile
import datetime as dt

STATE_FILE = "state.json"


def _to_iso(t: dt.datetime | None) -> str | None:
    if t is None:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return t.isoformat()


def _from_iso(s: str | None) -> dt.datetime | None:
    if not s:
        return None
    return dt.datetime.fromisoformat(s)


def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_state(state: dict) -> None:
    state = dict(state)
    dir_name = os.path.dirname(os.path.abspath(STATE_FILE)) or "."
    fd, temp_path = tempfile.mkstemp(prefix="state_", suffix=".tmp", dir=dir_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, STATE_FILE)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def get_dt(state: dict, key: str) -> dt.datetime | None:
    return _from_iso(state.get(key))


def set_dt(state: dict, key: str, t: dt.datetime | None) -> None:
    state[key] = _to_iso(t)
