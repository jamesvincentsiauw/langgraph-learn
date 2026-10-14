import json
from datetime import date

import httpx
from langchain_core.tools import tool

from agentlab import data


@tool
def search_openings(keyword: str, market: str | None = None, salary_only: bool = False) -> str:
    """Search open job openings by title keyword.

    Args:
        keyword: word to match in the job title, e.g. "backend" or "golang".
        market: one of SG, JP, NZ, REMOTE. Leave empty for all markets.
        salary_only: only return openings that state a salary.
    """
    rows = data.query_openings(keyword, market, salary_only)
    # Descriptions are long; the model gets the short fields and can ask for one opening in full.
    slim = [{k: v for k, v in r.items() if k != "description"} for r in rows]
    return json.dumps(slim) if slim else "No openings found."


@tool
def get_opening_details(key: str) -> str:
    """Full description of one opening, by the key returned from search_openings."""
    row = data.get_opening(key)
    return json.dumps(row) if row else f"No opening with key {key}."


@tool
def convert_to_idr(amount: float, currency: str) -> str:
    """Convert an amount in a foreign currency (SGD, JPY, NZD, USD, ...) to Indonesian rupiah."""
    resp = httpx.get(
        "https://api.frankfurter.dev/v1/latest",
        params={"from": currency.upper(), "to": "IDR"},
        timeout=10,
    )
    resp.raise_for_status()
    body = resp.json()
    rate = body["rates"]["IDR"]
    converted = f"Rp{amount * rate:,.0f}"
    return f"{amount:,.0f} {currency.upper()} = {converted} (rate {rate}, {body['date']})"


@tool
def save_to_shortlist(key: str, note: str) -> str:
    """Save an opening to Vincent's shortlist file with a short note on why it fits."""
    entries = data.read_shortlist()
    if any(e["key"] == key for e in entries):
        return f"{key} is already on the shortlist."
    entries.append({"key": key, "note": note, "saved_on": date.today().isoformat()})
    data.write_shortlist(entries)
    return f"Saved {key} to the shortlist ({len(entries)} entries)."


TOOLS = [search_openings, get_opening_details, convert_to_idr]
