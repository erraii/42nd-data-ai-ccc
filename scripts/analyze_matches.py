#!/usr/bin/env python3
"""Produce a deterministic metadata CSV and summary from the saved exports."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from match_io import ACTION, ROOT, load_matches, match_result, winner_from_protocol


def analyze(paths=None):
    matches = load_matches(paths)
    rows = []
    results = Counter()
    by_colour = {side: Counter() for side in ("CYAN", "MAGENTA")}
    score_agrees_with_cyan = 0
    for match, sources in matches:
        winner = winner_from_protocol(match)
        result = match_result(match)
        results[result] += 1
        by_colour[match["you"]][result] += 1
        if winner:
            score_agrees_with_cyan += match.get("score") == int(winner == "CYAN")
        rows.append({
            "match_id": match["match_id"], "played_at_utc": match["played_at"],
            "our_side": match["you"], "opponent": match.get("opponent") or "",
            "source_score": match.get("score"), "winner": winner or "",
            "our_result": result, "has_protocol": bool(match.get("protocol")),
            "recorded_actions": len(ACTION.findall(match.get("protocol") or "")),
            "source_exports": ";".join(sources),
        })
    summary = {
        "unique_matches": len(matches),
        "protocol_results": {k: results[k] for k in ("win", "loss", "unknown")},
        "by_colour": {side: {k: counts[k] for k in ("win", "loss", "unknown")}
                      for side, counts in by_colour.items()},
        "source_score_matches_cyan_outcome": score_agrees_with_cyan,
        "scope": "Two saved exports; not the complete tournament history or one model's evaluation set.",
        "result_method": "Winner text in protocol compared with 'you'. Missing protocols remain unknown.",
        "source_score_note": "For all records with a winner, score matches CYAN's outcome. Do not treat it as our score.",
    }
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    rows, summary = analyze(args.inputs)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "matches.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["match_id"])
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "match-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
