import json
import os
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = ROOT / "data" / "sample_openings.json"
SHORTLIST_PATH = Path(os.getenv("SHORTLIST_PATH", ROOT / "data" / "shortlist.json"))

OPENING_COLUMNS = (
    "key, company, title, locations, regions, remote, salary, apply_url, posted_at, description"
)


def db_path() -> Path | None:
    """The jobscout SQLite database, if it exists on this machine."""
    path = Path(os.getenv("JOBSCOUT_DB", ROOT.parent / "jobscout" / "data" / "jobscout.db"))
    return path.resolve() if path.exists() else None


def load_sample() -> list[dict]:
    return json.loads(SAMPLE_PATH.read_text())


def query_openings(
    keyword: str = "", market: str | None = None, salary_only: bool = False, limit: int = 5
) -> list[dict]:
    """Open job openings matching the filters, from the jobscout DB or the bundled sample."""
    if db_path():
        return _query_db(keyword, market, salary_only, limit)
    rows = load_sample()
    rows = [r for r in rows if keyword.lower() in r["title"].lower()]
    if market:
        rows = [r for r in rows if market.upper() in json.loads(r["regions"])]
    if salary_only:
        rows = [r for r in rows if r["salary"]]
    return rows[:limit]


def get_opening(key: str) -> dict | None:
    if db_path():
        with sqlite3.connect(f"file:{db_path()}?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            sql = f"SELECT {OPENING_COLUMNS} FROM openings WHERE key = ?"
            row = db.execute(sql, (key,)).fetchone()
            return dict(row) if row else None
    return next((r for r in load_sample() if r["key"] == key), None)


def _query_db(keyword: str, market: str | None, salary_only: bool, limit: int) -> list[dict]:
    sql = f"SELECT {OPENING_COLUMNS} FROM openings WHERE closed_at IS NULL AND title LIKE ?"
    params: list = [f"%{keyword}%"]
    if market:
        sql += " AND regions LIKE ?"
        params.append(f'%"{market.upper()}"%')
    if salary_only:
        sql += " AND salary IS NOT NULL AND salary != ''"
    sql += " ORDER BY last_seen DESC LIMIT ?"
    params.append(limit)
    with sqlite3.connect(f"file:{db_path()}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        return [dict(r) for r in db.execute(sql, params)]


def read_shortlist() -> list[dict]:
    if not SHORTLIST_PATH.exists():
        return []
    return json.loads(SHORTLIST_PATH.read_text())


def write_shortlist(entries: list[dict]) -> None:
    SHORTLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    SHORTLIST_PATH.write_text(json.dumps(entries, indent=1, ensure_ascii=False))
