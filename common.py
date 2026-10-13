import json
import os
import sqlite3
from pathlib import Path

import httpx
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

load_dotenv()

DB_PATH = Path(os.getenv("JOBSCOUT_DB", "../jobscout/data/jobscout.db")).resolve()

SYSTEM = (
    "You help Vincent, a backend engineer in Jakarta, look at job openings abroad. "
    "Use the tools to find openings and convert salaries to IDR. "
    "Only state salaries the tools returned. Answer in casual Indonesian."
)


def make_model():
    if os.getenv("ANTHROPIC_API_KEY"):
        return init_chat_model(
            os.getenv("MODEL", "claude-haiku-5-5"), model_provider="anthropic", max_tokens=2048
        )
    # OpenRouter speaks the OpenAI API, so the openai provider works with a different base_url.
    return init_chat_model(
        os.getenv("MODEL", "anthropic/claude-haiku-5.5"),
        model_provider="openai",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        max_tokens=2048,
    )


@tool
def search_openings(keyword: str, market: str | None = None, salary_only: bool = False) -> str:
    """Search open job openings by title keyword.

    Args:
        keyword: word to match in the job title, e.g. "backend" or "golang".
        market: one of SG, JP, NZ, REMOTE. Leave empty for all markets.
        salary_only: only return openings that state a salary.
    """
    sql = "SELECT key, company, title, locations, remote, salary FROM openings WHERE closed_at IS NULL AND title LIKE ?"
    params: list = [f"%{keyword}%"]
    if market:
        sql += " AND regions LIKE ?"
        params.append(f'%"{market.upper()}"%')
    if salary_only:
        sql += " AND salary IS NOT NULL AND salary != ''"
    sql += " ORDER BY last_seen DESC LIMIT 5"

    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(r) for r in db.execute(sql, params)]
    return json.dumps(rows) if rows else "No openings found."


@tool
def convert_to_idr(amount: float, currency: str) -> str:
    """Convert an amount in a foreign currency (SGD, JPY, NZD, USD, ...) to Indonesian rupiah."""
    resp = httpx.get(
        "https://api.frankfurter.dev/v1/latest",
        params={"from": currency.upper(), "to": "IDR"},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    rate = data["rates"]["IDR"]
    return f"{amount:,.0f} {currency.upper()} = Rp{amount * rate:,.0f} (rate {rate}, {data['date']})"


TOOLS = [search_openings, convert_to_idr]


def print_updates(stream) -> None:
    """Print each graph step so the model -> tools -> model loop is visible."""
    for update in stream:
        for node, output in update.items():
            for msg in output["messages"]:
                if getattr(msg, "tool_calls", None):
                    for call in msg.tool_calls:
                        print(f"[{node}] calls {call['name']}({call['args']})")
                elif msg.type == "tool":
                    print(f"[{node}] {msg.name} -> {str(msg.content)[:160]}")
                else:
                    print(f"[{node}] answer:\n{msg.content}")
