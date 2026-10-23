# Which training artifacts belong on GitHub?

This repository was prepared from the actual source files in `ccc.zip`. Only a
file listing was supplied for the separate training directory, so the model and
log files below have not yet been added. That directory is approximately 320 MB;
the public repository only needs a selected subset.

## Collect the selected files automatically

Run this from the prepared repository root. It does not modify the source training files:

```bash
python3 scripts/collect_training_artifacts.py \
  --source "/goinfre/$USER/colony-training"
```

| Source | Repository destination | Purpose |
|---|---|---|
| `smoke/model_fixed.onnx` | `models/baseline-level3.onnx` | First valid Level 3 baseline |
| `level4-v1/model.onnx` | `models/baseline-level4.onnx` | First Level 4 comparison model |
| `selfplay/best.onnx` | `models/latest-selfplay.onnx` | Latest local champion; not assumed to be the final tournament submission |
| `selfplay/history.jsonl` | `results/training/history.jsonl` | Game counts, iterations, promotions and timings |
| `selfplay/state.json` | `results/training/state-summary.json` | Training summary with shortened paths; not a resume checkpoint |
| `level4-v1/results.json` | `results/training/baseline-level4-results.json` | Initial benchmark results |
| `selfplay/iteration-*/validation.log` | `results/training/validation/iteration-*.log` | Small validation and error records |
| `selfplay/iteration-*/candidate.onnx`, `selfplay/best*.onnx` | `results/training/checkpoint-index.csv` | SHA-256 traceability without copying every model binary |

## Select the exact final tournament submission

The filename `best.onnx`, a numbered snapshot, or the highest iteration number
does not by itself identify the winning submission. Select the exact file that
was uploaded at the end of the contest.

For example, **only if** your final upload was `best_7.onnx`:

```bash
python3 scripts/collect_training_artifacts.py \
  --source "/goinfre/$USER/colony-training" \
  --winner "/goinfre/$USER/colony-training/selfplay/best_7.onnx" \
  --include-weights
```

`best_7.onnx` is an example. Replace it with your actual final submission.
The collector copies it to `models/submission.onnx`, compares its hash with the
saved candidates, and adds `models/submission.pt` only if a matching candidate
ONNX has an accompanying `candidate.pt`. It does not pair an old submitted model
with unrelated latest `best.pt` weights.

Validate the selected model separately:

```bash
python tools/validate_submission.py models/submission.onnx
```

## Keep these in a local backup, outside GitHub

| Files | Decision |
|---|---|
| Every `iteration-*/candidate.pt` and `candidate.onnx` | Keep the full experiment archive outside Git. Publish the selected models and their hash index. |
| `selfplay/iteration-00110/replay.npz` and other `.npz` files | Keep in the original run directory if you want to resume training. Omit from Git. |
| `selfplay/best_0.onnx` through `best_10.onnx` | Select the final submission rather than automatically publishing every snapshot. All available snapshots are indexed by hash. |
| `selfplay/other_models/zak/*`, `selfplay/other_models/tobias/*` | Other people's model binaries; omitted from this repository package. |
| `.venv`, installed Python packages, `__pycache__`, bytecode | Recreated during setup; omit from Git. |
| `smoke/dataset.npz`, `level4-v1/dataset.npz` | Not required in the main source repository, even though these particular datasets are small. |
| Original ZIP archives and superseded script copies | Preserve in the original backup; omit duplicates from the clean repository. |

Nothing excluded from this prepared copy was deleted from the original uploads
or your training directory. Keep a separate backup of the original run rather than
relying on its `/goinfre` copy alone.

## Before pushing

1. Confirm that `models/submission.onnx` is the exact final uploaded model.
2. If available, add a result screenshot with the Germany filter selected as
   `docs/images/germany-ranking.png`. The existing Hall of Fame screenshot shows
   the overall 24th-place result; the README identifies the national result as
   separately reported by the team.
3. Run `python -m unittest discover -s tests -v` and `python scripts/replay_matches.py`.
4. Use `git status --short` to check that the virtual environment, uncurated
   training directory and other teams' model files have not been added. Avoid
   overriding the ignore rules with `git add -f`.
5. Retain the root `LICENSE`: the supplied engine is not presented as a team-authored MIT project.

After creating a new GitHub repository, run these commands inside the clean folder:

```bash
git init
git add .
git status --short
git commit -m "Organize Colony One contest training, replays and results"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

Replace the remote URL with your own repository's address. No GitHub push was
performed while preparing this package.
