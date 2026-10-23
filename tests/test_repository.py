"""Meaningful checks for result semantics, replay compatibility and curation."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from analyze_matches import analyze
from collect_training_artifacts import collect, digest
from match_io import load_matches, match_result, winner_from_protocol
from replay_matches import replay


class TournamentRecords(unittest.TestCase):
    def test_unique_records_and_correct_perspective(self):
        records = load_matches()
        self.assertEqual(len(records), 60)
        self.assertEqual(len({r[0]["match_id"] for r in records}), 60)
        sample = next(row for row, _ in records if row["match_id"] == 15111)
        self.assertEqual(sample["you"], "MAGENTA")
        self.assertEqual(sample["score"], 0)
        self.assertEqual(winner_from_protocol(sample), "MAGENTA")
        self.assertEqual(match_result(sample), "win")

    def test_metadata_only_is_not_guessed(self):
        metadata = [row for row, _ in load_matches() if not row.get("protocol")]
        self.assertEqual(len(metadata), 7)
        for row in metadata:
            self.assertEqual(match_result(row), "unknown")
            self.assertEqual(replay(row)["status"], "metadata_only")

    def test_all_complete_protocols_replay(self):
        complete = [row for row, _ in load_matches() if row.get("protocol")]
        self.assertEqual(len(complete), 53)
        for row in complete:
            with self.subTest(match_id=row["match_id"]):
                self.assertEqual(replay(row)["status"], "verified")

    def test_summary_uses_protocols_not_raw_score(self):
        rows, summary = analyze()
        self.assertEqual(len(rows), 60)
        self.assertEqual(summary["protocol_results"], {"win": 37, "loss": 16, "unknown": 7})
        self.assertEqual(summary["source_score_matches_cyan_outcome"], 53)

    def test_conflicting_duplicate_is_rejected(self):
        row = load_matches()[0][0]
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.json", Path(directory) / "b.json"
            a.write_text(json.dumps([row]))
            b.write_text(json.dumps([{**row, "opponent": "different"}]))
            with self.assertRaisesRegex(ValueError, "Conflicting exports"):
                load_matches([a, b])


class ArtifactCollector(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name)
        self.source, self.repo = base / "training", base / "repo"
        self.repo.mkdir()
        (self.repo / "pyproject.toml").write_text("")
        (self.repo / "LICENSE").write_text("test fixture")
        self.write("selfplay/best.onnx", b"latest model")
        self.write("selfplay/best_7.onnx", b"submitted model")
        self.write("selfplay/best.pt", b"unrelated latest weights")
        self.write("selfplay/iteration-00005/candidate.onnx", b"submitted model")
        self.write("selfplay/iteration-00005/candidate.pt", b"matching weights")
        self.write("selfplay/iteration-00005/replay.npz", b"not for git")
        self.write("selfplay/iteration-00005/validation.log", f"model: {self.source}/selfplay/iteration-00005/candidate.onnx\nACCEPTED\n".encode())
        self.write("selfplay/other_models/rival.onnx", b"someone else's model")
        self.write("selfplay/state.json", json.dumps({"completed_games": 2500, "out": str(self.source / "selfplay")}).encode())
        self.write("selfplay/history.jsonl", b'{"iteration":5,"promoted":true}\n')
        self.write("level4-v1/model.onnx", b"baseline")

    def write(self, relative, content):
        path = self.source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def do_collect(self, winner=None, include_weights=False, **options):
        with contextlib.redirect_stdout(io.StringIO()):
            return collect(self.source, self.repo, winner, include_weights, **options)

    def test_default_selection_does_not_invent_winner_or_change_source(self):
        before = {p.relative_to(self.source): digest(p) for p in self.source.rglob("*") if p.is_file()}
        summary = self.do_collect()
        self.assertFalse(summary["winning_model_explicitly_selected"])
        self.assertFalse((self.repo / "models/submission.onnx").exists())
        self.assertFalse(list(self.repo.rglob("*.npz")))
        self.assertFalse(list(self.repo.rglob("rival.onnx")))
        self.assertNotIn(str(self.source), (self.repo / "results/training/state-summary.json").read_text())
        after = {p.relative_to(self.source): digest(p) for p in self.source.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_submission_weights_are_matched_by_onnx_hash(self):
        winner = self.source / "selfplay/best_7.onnx"
        self.do_collect(winner, True)
        self.assertEqual((self.repo / "models/submission.onnx").read_bytes(), b"submitted model")
        self.assertEqual((self.repo / "models/submission.pt").read_bytes(), b"matching weights")
        manifest = json.loads((self.repo / "models/manifest.json").read_text())
        weights = next(r for r in manifest["files"] if r["path"] == "models/submission.pt")
        self.assertEqual(weights["paired_onnx_sha256"], digest(winner))

    def test_changing_submission_does_not_retain_stale_weights(self):
        self.do_collect(self.source / "selfplay/best_7.onnx", True)
        self.do_collect(self.source / "selfplay/best.onnx", False)
        self.assertFalse((self.repo / "models/submission.pt").exists())

    def test_unmatched_weights_are_not_substituted(self):
        self.do_collect(self.source / "selfplay/best.onnx", True)
        self.assertFalse((self.repo / "models/submission.pt").exists())

    def test_missing_selected_file_and_unselected_weights_fail(self):
        with self.assertRaises(ValueError):
            self.do_collect(self.source / "missing.onnx", False)
        with self.assertRaises(ValueError):
            self.do_collect(None, True)

    def test_evolution_preserves_numbered_snapshots_and_hash_mapping(self):
        self.write("selfplay/best_10.onnx", b"later model")
        self.write("selfplay/best_2.onnx", b"earlier model")
        self.do_collect(self.source / "selfplay/best_7.onnx", include_snapshots=True)
        import csv
        with (self.repo / "results/training/model-evolution.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([r["snapshot"] for r in rows], ["best_2.onnx", "best_7.onnx", "best_10.onnx"])
        self.assertEqual(rows[1]["same_hash_as_submission"], "True")
        self.assertEqual(rows[1]["matching_candidates"], "selfplay/iteration-00005/candidate.onnx")
        self.assertTrue((self.repo / "models/evolution/best_10.onnx").exists())

    def test_shared_models_credit_teammate_and_peers_separately(self):
        self.write("selfplay/other_models/tobias/model_tobias.onnx", b"teammate model")
        self.write("selfplay/other_models/zak/apex_predator.onnx", b"peer model")
        self.do_collect(include_shared_models=True)
        self.assertTrue((self.repo / "models/team/tobias/model_tobias.onnx").exists())
        self.assertTrue((self.repo / "models/community/zak/apex_predator.onnx").exists())
        self.assertTrue((self.repo / "models/community/unattributed/rival.onnx").exists())
        entries = json.loads((self.repo / "models/manifest.json").read_text())["files"]
        teammate = next(r for r in entries if r["path"] == "models/team/tobias/model_tobias.onnx")
        self.assertEqual(teammate["contributor"], "tobias")
        self.assertIn("Teammate", teammate["role"])


if __name__ == "__main__":
    unittest.main()
