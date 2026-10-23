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
not all copied. Another team's ONNX files are not collected. The collector does
not certify model format: run the official validator on a selected model before
describing it as a validated submission.
