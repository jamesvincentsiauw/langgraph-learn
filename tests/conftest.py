import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lessons"))


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Tests never touch the real jobscout DB, the shortlist, or an API key."""
    monkeypatch.setenv("JOBSCOUT_DB", str(tmp_path / "missing.db"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    from agentlab import data

    monkeypatch.setattr(data, "SHORTLIST_PATH", tmp_path / "shortlist.json")
    os.environ.pop("MODEL", None)
