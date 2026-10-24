# Add ranking evidence, model evolution and shared contributions

Extract the update archive into the existing repository root, replacing its
included files. Existing model binaries and imported training records are retained.

The update includes the Germany-filtered result screenshot, an updated English
README, expanded collector options, model attribution notes and tests.

Run this command from the repository root:

```bash
python3 scripts/collect_training_artifacts.py \
  --source "/goinfre/$USER/colony-training" \
  --winner "/goinfre/$USER/colony-training/selfplay/best_8.onnx" \
  --include-weights \
  --include-snapshots \
  --include-shared-models
```

| Destination | Contents |
|---|---|
| `models/submission.onnx` | Confirmed final tournament submission, `best_8.onnx` |
| `models/latest_models/best_*.onnx` | Every available numbered self-play snapshot |
| `results/training/model-evolution.csv` | Snapshot hashes and matching saved candidates |
| `models/team/tkern/` | Tobias's teammate contributions |
| `models/community/malshare/` | Peer-shared models from the source folder named `zak` |
| `models/manifest.json` | File sizes, hashes, contributor roles and source paths |

Version names alone do not prove a monotonic improvement in playing strength.
The archive preserves the actual saved versions; any performance curve should
come from measured comparisons. Keep the original complete training directory
outside Git as a resumable backup. The collector leaves it unchanged.

After importing, run:

```bash
python3 -m unittest discover -s tests -v
git add .
git status --short
git diff --cached --stat
git commit -m "Add Germany ranking, model evolution and team contributions"
git push
```

Retain the contributors' publication permission and any supplied model-license
terms for peer-shared binaries. Crediting a contributor does not assign their
model the bundled engine's license.
