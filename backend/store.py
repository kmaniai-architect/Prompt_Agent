import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "prompts.db"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init() -> None:
    with _conn() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                role TEXT NOT NULL,
                context TEXT NOT NULL,
                input TEXT NOT NULL,
                output TEXT NOT NULL,
                prompt TEXT NOT NULL,
                mode TEXT NOT NULL,
                transcript TEXT NOT NULL DEFAULT '[]',
                reverse TEXT,
                created_at TEXT NOT NULL
            )"""
        )
        cols = {r["name"] for r in c.execute("PRAGMA table_info(prompts)")}
        for name, ddl in (("strategy", "TEXT NOT NULL DEFAULT ''"), ("length", "TEXT NOT NULL DEFAULT ''")):
            if name not in cols:
                c.execute(f"ALTER TABLE prompts ADD COLUMN {name} {ddl}")


def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["transcript"] = json.loads(d["transcript"] or "[]")
    d["reverse"] = json.loads(d["reverse"]) if d["reverse"] else None
    return d


def create(data: dict) -> dict:
    with _conn() as c:
        cur = c.execute(
            """INSERT INTO prompts (title, role, context, input, output, prompt, mode, transcript,
                                    created_at, strategy, length)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                data["title"], data["role"], data["context"], data["input"], data["output"],
                data["prompt"], data["mode"], json.dumps(data.get("transcript", [])),
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                data.get("strategy", ""), data.get("length", ""),
            ),
        )
        pid = cur.lastrowid
    return get(pid)


def get(pid: int) -> dict | None:
    with _conn() as c:
        r = c.execute("SELECT * FROM prompts WHERE id=?", (pid,)).fetchone()
        return _row(r) if r else None


def list_all() -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM prompts ORDER BY id DESC").fetchall()
        return [_row(r) for r in rows]


def save_reverse(pid: int, reverse: dict) -> None:
    with _conn() as c:
        c.execute("UPDATE prompts SET reverse=? WHERE id=?", (json.dumps(reverse), pid))


def delete(pid: int) -> bool:
    with _conn() as c:
        return c.execute("DELETE FROM prompts WHERE id=?", (pid,)).rowcount > 0
