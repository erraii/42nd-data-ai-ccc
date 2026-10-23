"""Shared readers for the two tournament exports supplied with this project."""
from __future__ import annotations

import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = sorted((ROOT / "data/tournament/raw").glob("*.json"))
ACTION = re.compile(
    r"^\s*(\d+)\.\s+(CYAN|MAGENTA)\s+(\w+)\s+"
    r"([A-H][1-8])\s*->\s*([A-H][1-8])\s+(MOVE|SHOOT)", re.MULTILINE
)
WINNER = re.compile(r"\b(CYAN|MAGENTA) WINS\b")


def load_matches(paths=None):
    matches = {}
    for path in paths if paths is not None else DEFAULT_INPUTS:
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected a list of matches: {path}")
        for row in data:
            match_id = row["match_id"]
            if row.get("you") not in ("CYAN", "MAGENTA"):
                raise ValueError(f"Unknown side in match {match_id}")
            if match_id in matches:
                existing, sources = matches[match_id]
                if existing != row:
                    raise ValueError(f"Conflicting exports for match {match_id}")
                sources.append(path.name)
            else:
                matches[match_id] = (row, [path.name])
    return sorted(matches.values(), key=lambda pair: (pair[0]["played_at"], pair[0]["match_id"]))


def winner_from_protocol(row):
    winners = WINNER.findall(row.get("protocol") or "")
    return winners[-1] if winners else None


def match_result(row):
    winner = winner_from_protocol(row)
    return "unknown" if winner is None else ("win" if winner == row["you"] else "loss")


def square(text):
    return 8 - int(text[1]), ord(text[0]) - ord("A")
