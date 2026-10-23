# CPU training on a storage-limited 42 Linux account

If the existing external environment is still available, activate it and install
the prepared repository in editable mode from this repository's root:

```bash
source "/goinfre/$USER/colony-training/.venv/bin/activate"
uv pip install -e .
```

For a fresh setup, keep both the environment and cache outside the small home quota:

```bash
export UV_CACHE_DIR="/goinfre/$USER/uv-cache"
export UV_LINK_MODE=copy
mkdir -p "/goinfre/$USER/colony-training"
uv venv "/goinfre/$USER/colony-training/.venv" --python 3.11
source "/goinfre/$USER/colony-training/.venv/bin/activate"
uv pip install -e .
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Start a new run in a distinct output folder so the contest archive is preserved:

```bash
python scripts/select_scene.py level4
python scripts/train_selfplay.py \
  --games 1000 --workers 4 \
  --out "/goinfre/$USER/colony-training/reproduction"
```

Do not delete the original self-play directory after collecting a few public
artifacts. Resuming that run needs its original state, referenced champion models,
rivals and replay buffer. Back it up separately from the small public repository.
