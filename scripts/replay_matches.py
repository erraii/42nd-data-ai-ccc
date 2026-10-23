#!/usr/bin/env python3
"""Replay exported action coordinates with the original engine, or show one game."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from match_io import ACTION, ROOT, load_matches, match_result, square, winner_from_protocol

sys.path.insert(0, str(ROOT))
import numpy as np
from colony_one import ColonyOneEnv


def replay(match, render=False):
    if not match.get("protocol") or "starting_observation" not in match:
        return {"match_id": match["match_id"], "status": "metadata_only"}
    env = ColonyOneEnv(scene=np.asarray(match["starting_observation"], dtype=np.float32))
    actions = ACTION.findall(match["protocol"])
    if not actions:
        raise ValueError("Protocol contains no recognized actions")
    if render:
        print(env.render())
    for number, side, abbreviation, start, end, logged_kind in actions:
        if side != env.current_team or int(number) != env.action:
            raise ValueError(f"Unexpected team/action number at action {number}")
        piece = env.board.piece_at(*square(start))
        if piece is None or piece.abbr != abbreviation:
            raise ValueError(f"Unexpected piece at action {number}")
        # A redirected Kernel attack may be logged as SHOOT although its legal
        # engine action is MOVE. Coordinates and the engine determine the kind.
        legal = [a for a in env.legal_actions()
                 if a.from_pos == square(start) and a.to_pos == square(end)]
        if len(legal) != 1:
            raise ValueError(f"No unique legal action for {start}->{end} at {number}")
        env.step(legal[0])
        if render:
            print(f"\n{number}. {side} {abbreviation} {start}->{end} (logged {logged_kind})")
            print(env.render())
    winner = winner_from_protocol(match)
    if winner is None or not env.terminated or env.winner != winner:
        raise ValueError(f"Final result mismatch: engine={env.winner}, protocol={winner}")
    return {"match_id": match["match_id"], "status": "verified", "actions": len(actions),
            "winner": winner, "our_side": match["you"], "our_result": match_result(match)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", type=int)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--inputs", nargs="+", type=Path)
    parser.add_argument("--report", type=Path, default=ROOT / "results/replay-validation.json")
    args = parser.parse_args()
    selected = [row for row, _ in load_matches(args.inputs)
                if args.match_id is None or row["match_id"] == args.match_id]
    if not selected:
        parser.error("No matching game found")
    if args.render and args.match_id is None:
        parser.error("Use --match-id with --render to avoid dumping every game")
    results = []
    for match in selected:
        try:
            results.append(replay(match, args.render))
        except Exception as exc:
            results.append({"match_id": match["match_id"], "status": "failed", "reason": str(exc)})
    if args.match_id is None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(results, indent=2) + "\n")
    totals = {status: sum(row["status"] == status for row in results)
              for status in ("verified", "metadata_only", "failed")}
    print(json.dumps(totals, indent=2))
    return int(totals["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
