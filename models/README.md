# Selected model artifacts

No trained model binaries were present in `ccc.zip`. This folder is intentionally
ready for selected real artifacts; `play/random.onnx` is only the supplied example.

Run `scripts/collect_training_artifacts.py` to populate:

| File | Meaning |
|---|---|
| `baseline-level3.onnx` | `smoke/model_fixed.onnx` |
| `baseline-level4.onnx` | `level4-v1/model.onnx` |
| `latest-selfplay.onnx` | The external self-play folder's current `best.onnx`; not automatically the final submission |
| `submission.onnx` | Exact final contest file explicitly selected with `--winner` |
| `submission.pt` | Optional checkpoint paired with a byte-identical candidate ONNX; never inferred from an unrelated `best.pt` |
| `manifest.json` | SHA-256, size, role and source-relative path for each imported file |

Numbered `best_*.onnx` snapshots and all iteration candidates are indexed by hash,
not all copied by default. Use `--include-snapshots` to archive every available
numbered snapshot in `models/evolution/` and write `results/training/model-evolution.csv`.
This preserves the model versions without copying every iteration's weights and
replay buffer. Snapshot numbers are version labels, not exact training-game counts.

Use `--include-shared-models` to copy contributor-supplied models with attribution:

| Folder | Attribution |
|---|---|
| `team/tkern/` | Tobias, a teammate of Eray in 42Heilbronn_GTA6 |
| `community/malshare/` | community/malshare/ | Models shared by Mhd Zakarea Alshareef, who competed under the team name malshare |

Shared models retain contributor attribution and are not labeled as outputs of our
self-play trainer. Keep the contributors' publication permission and supplied
licensing terms with the files. The collector does not certify model format:
run the official validator before describing a model as a validated submission.
