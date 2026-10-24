#!/usr/bin/env python3
"""Copy a small, explicit selection from a local training folder into this repo.

Never deletes or changes training files. Does not infer which model won.
Model files copied here must still be checked with the official validator.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def sanitize(value, source):
    if isinstance(value, dict):
        return {key: sanitize(item, source) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item, source) for item in value]
    if isinstance(value, str):
        return value.replace(str(source), "<training-root>")
    return value


def copy_artifact(source, relative, role, origin, manifest, repo):
    source = Path(source)
    if not source.is_file():
        return None
    if source.stat().st_size > 10 * 1024 * 1024:
        raise ValueError(f"Unexpectedly large curated model: {source}")
    target = repo / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    entry = {"path": relative, "role": role, "source": origin,
             "bytes": target.stat().st_size, "sha256": digest(target),
             "validation": "not run by the collector"}
    manifest[relative] = entry
    return entry


def collect(source, repo, winner=None, include_weights=False,
            include_snapshots=False, include_shared_models=False):
    source, repo = Path(source).resolve(), Path(repo).resolve()
    if not source.is_dir():
        raise ValueError(f"Training root does not exist: {source}")
    if not (repo / "pyproject.toml").is_file() or not (repo / "LICENSE").is_file():
        raise ValueError("Destination must be the prepared repository root")
    source_relative_to_repo = source == repo or repo in source.parents
    if source_relative_to_repo:
        raise ValueError("Keep the uncurated training root outside this repository")
    if winner is not None and not Path(winner).is_file():
        raise ValueError(f"Selected final submission does not exist: {winner}")
    if include_weights and winner is None:
        raise ValueError("--include-weights requires an explicitly selected --winner")
    models_dir, results_dir = repo / "models", repo / "results/training"
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = models_dir / "manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": []}
    manifest = {row["path"]: row for row in old.get("files", [])}
    selection = [
        ("smoke/model_fixed.onnx", "models/baseline-level3.onnx", "Level 3 baseline"),
        ("level4-v1/model.onnx", "models/baseline-level4.onnx", "First Level 4 baseline"),
        ("selfplay/best.onnx", "models/latest-selfplay.onnx", "Latest local champion; not automatically the final submission"),
    ]
    for origin, target, role in selection:
        copy_artifact(source / origin, target, role, origin, manifest, repo)
    for origin, target in [
        ("selfplay/history.jsonl", "history.jsonl"),
        ("selfplay/state.json", "state-summary.json"),
        ("level4-v1/results.json", "baseline-level4-results.json"),
    ]:
        path = source / origin
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            text = json.dumps(sanitize(json.loads(text), source), indent=2) + "\n"
        else:
            text = text.replace(str(source), "<training-root>")
        (results_dir / target).write_text(text, encoding="utf-8")
    validation_dir = results_dir / "validation"
    for path in sorted((source / "selfplay").glob("iteration-*/validation.log")):
        validation_dir.mkdir(exist_ok=True)
        destination = validation_dir / f"{path.parent.name}.log"
        destination.write_text(path.read_text(errors="replace").replace(str(source), "<training-root>"))
    # Keep a hash index, rather than copying every 1.3 MiB checkpoint.
    candidates = sorted((source / "selfplay").glob("iteration-*/candidate.onnx"))
    snapshots = sorted((source / "selfplay").glob("best*.onnx"))
    index = [{"source": p.relative_to(source).as_posix(), "bytes": p.stat().st_size,
              "sha256": digest(p)} for p in candidates + snapshots]
    with (results_dir / "checkpoint-index.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(index)
    if winner is not None:
        winner = Path(winner).resolve()
        previous_submission = manifest.get("models/submission.onnx")
        previous_weights = manifest.get("models/submission.pt")
        if (previous_submission and previous_weights
                and previous_submission["sha256"] != digest(winner)):
            old_weights_path = repo / "models/submission.pt"
            if old_weights_path.exists():
                if digest(old_weights_path) != previous_weights["sha256"]:
                    raise ValueError("Existing submission.pt was changed locally; move it aside before selecting another winner")
                old_weights_path.unlink()
            del manifest["models/submission.pt"]
            print("Removed the previous curated weights because the selected ONNX changed.")
        try:
            origin = winner.relative_to(source).as_posix()
        except ValueError:
            origin = winner.name
        entry = copy_artifact(winner, "models/submission.onnx", "Final tournament submission selected by the user",
                              origin, manifest, repo)
        matching = [row["source"] for row in index if row["sha256"] == entry["sha256"]]
        entry["same_hash_as"] = matching
        if include_weights:
            weights = next((p.with_suffix(".pt") for p in candidates
                            if digest(p) == entry["sha256"] and p.with_suffix(".pt").is_file()), None)
            if weights is None:
                print("No byte-identical candidate ONNX with matching candidate.pt found; weights omitted.")
            else:
                weights_entry = copy_artifact(weights, "models/submission.pt", "Weights paired with a byte-identical candidate ONNX",
                                              weights.relative_to(source).as_posix(), manifest, repo)
                weights_entry["paired_onnx_sha256"] = entry["sha256"]
    if include_snapshots:
        numbered = [p for p in snapshots if re.fullmatch(r"best_\d+\.onnx", p.name)]
        numbered.sort(key=lambda p: int(p.stem.split("_")[-1]))
        submitted = manifest.get("models/submission.onnx", {}).get("sha256")
        evolution = []
        for path in numbered:
            origin = path.relative_to(source).as_posix()
            entry = copy_artifact(path, f"models/latest_models/{path.name}", "Numbered self-play snapshot",
                                  origin, manifest, repo)
            matching = [r["source"] for r in index if r["sha256"] == entry["sha256"]
                        and "/iteration-" in r["source"]]
            entry["same_hash_as"] = matching
            evolution.append({"snapshot": path.name, "sha256": entry["sha256"],
                              "bytes": entry["bytes"], "matching_candidates": ";".join(matching),
                              "same_hash_as_submission": bool(submitted and submitted == entry["sha256"])})
        with (results_dir / "model-evolution.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["snapshot", "sha256", "bytes", "matching_candidates", "same_hash_as_submission"])
            writer.writeheader()
            writer.writerows(evolution)
    if include_shared_models:
        shared_root = source / "selfplay/other_models"
        for path in sorted(shared_root.rglob("*.onnx")):
            relative = path.relative_to(shared_root)
            owner = relative.parts[0] if len(relative.parts) > 1 else "unattributed"
            public_owner = {
                "tobias": "tkern",
                "tkern": "tkern",
                "zak": "malshare",
                "malshare": "malshare",
            }.get(owner.lower(), owner)
            public_relative = (
                Path(public_owner).joinpath(*relative.parts[1:])
                if len(relative.parts) > 1
                else Path(public_owner) / relative
            )
            category = "team" if owner.lower() in {"tobias", "tkern"} else "community"
            role = "Teammate model contributed by Tobias" if category == "team" else f"Peer-shared model, source attribution: {owner}"
            entry = copy_artifact(path, f"models/{category}/{public_relative.as_posix()}", role,
                                  path.relative_to(source).as_posix(), manifest, repo)
            entry["contributor"] = owner
            entry["training_provenance"] = "Contributor-supplied; not claimed as output of this repository's self-play run"
    summary = {"models_copied": sorted(manifest), "indexed_checkpoints_and_snapshots": len(index),
               "winning_model_explicitly_selected": "models/submission.onnx" in manifest,
               "excluded": ["all .npz replay/dataset files", "all unselected candidate.pt / candidate.onnx files",
                            *([] if include_shared_models else ["selfplay/other_models (not selected for this import)"]),
                            "virtual environments"],
               "state_note": "state-summary.json is redacted documentation, not a resumable checkpoint."}
    manifest_path.write_text(json.dumps({"files": sorted(manifest.values(), key=lambda r: r["path"])}, indent=2) + "\n")
    (results_dir / "artifact-import-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Training root containing selfplay/, smoke/, level4-v1/")
    parser.add_argument("--winner", type=Path, help="The exact ONNX file actually submitted at the end of the contest")
    parser.add_argument("--include-weights", action="store_true", help="Include candidate.pt only when its paired ONNX hash matches --winner")
    parser.add_argument("--include-snapshots", action="store_true", help="Archive all numbered best_N.onnx snapshots and create an evolution CSV")
    parser.add_argument("--include-shared-models", action="store_true", help="Import shared ONNX files with contributor attribution; Tobias is recorded as a teammate")
    args = parser.parse_args()
    try:
        summary = collect(args.source, ROOT, args.winner, args.include_weights,
                          args.include_snapshots, args.include_shared_models)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
