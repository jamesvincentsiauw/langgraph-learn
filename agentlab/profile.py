import os
from pathlib import Path

from agentlab.data import ROOT

PROFILE_PATH = Path(os.getenv("PROFILE_PATH", ROOT / "data" / "profile.example.md"))


def load_profile() -> str:
    """The candidate profile the scoring lessons judge openings against."""
    return PROFILE_PATH.read_text().strip()
