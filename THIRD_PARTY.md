# Supplied material and attribution

The official Colony: One environment shipped in the contest archive is copyright
Paul Puntschart, 2026. Its original license is copied verbatim to the repository's
root `LICENSE`. Keep that notice with copies and derivatives.

The supplied material comprises:

- `colony_one/`: engine, movement rules, board representation, encoding, MCTS,
  renderer, baseline agents, ONNX runner and submission checks.
- `tools/validate_submission.py`.
- `play/model_vs_model.py` and `play/random.onnx`.
- `pyproject.toml`, the original environment documentation and dependency snapshot.
- The starting scenes. The Level 4 scene is the selected default.

The engine and MCTS implementation are not authored by this team. The active
training and curation scripts live separately in `scripts/`. Their original
contest copies were found inside the engine directory and have been relocated.

The event screenshots and game exports were supplied by the repository owner.
The user-reported Germany placement is recorded separately from the screenshot's
overall placement. Other teams' model binaries listed in the local training folder
are **not** included in this repository.

The auxiliary `archive/incomplete/gen_data.py` was present in the supplied project
folder. Its authorship is not established by that folder alone. It requires an
absent `oracle.py` and is preserved as an incomplete historical reference, not as
a working component of this project's training pipeline.
