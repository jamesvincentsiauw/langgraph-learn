"""Checkpointer helpers shared by the lessons that keep conversation threads."""

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from agentlab.data import ROOT

THREADS_PATH = ROOT / "data" / "threads.sqlite"


def open_sqlite_saver(path: Path = THREADS_PATH) -> SqliteSaver:
    """A SqliteSaver over one file. The connection stays open for the life of the process."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # LangGraph may run nodes on worker threads, so the connection cannot be thread-bound.
    conn = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(conn)
