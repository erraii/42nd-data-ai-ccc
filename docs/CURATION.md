# Curation and provenance

This repository was prepared from `ccc.zip`, two event screenshots, and a text
listing of a separate training directory. The original uploads were not changed.

## Retained and relocated

| Original location inside `ccc.zip` | Repository location | Treatment |
|---|---|---|
| `level-3/colony-one-env/colony_one/` | `colony_one/` | Engine source retained; team training scripts removed from this package directory |
| `.../colony_one/train_colony.py` | `scripts/train_bootstrap.py` | Latest bootstrap version; explicit value clamp added consistently with the final self-play fix |
| `.../colony_one/train_colony_selfplay_new.py` | `scripts/train_selfplay.py` | Final provided self-play version, including value clamp and pending-candidate recovery; copied without algorithm changes |
| `.../manual_play.py` | `scripts/manual_play.py` | English prompts and a main guard; same random-opponent play loop |
| `.../colony_one/scene-level3.json` | `scenes/level3.json` | Retained separately |
| `.../colony_one/scene.json` | `scenes/level4.json` and `colony_one/scene.json` | Retained; Level 4 remains selected |
| `.../README.md` | `docs/upstream/README.md` | Original environment README retained as reference |
| `.../requirements.txt` | `docs/upstream/requirements-original.txt` | Supplied dependency snapshot retained; not labeled as a full training lockfile |
| `level-3/tournament-matches.json` | `data/tournament/raw/export-01.json` | Original bytes retained |
| `level-3/tournament-matches_2.json` | `data/tournament/raw/export-02.json` | Original bytes retained |
| `level-3/gen_data.py` | `archive/incomplete/gen_data.py` | Preserved but not runnable without its missing oracle dependency |

The engine source files compared with the original Level 3 environment are
unchanged; the selected `scene.json` differs because Level 4 was installed during
the contest. The upstream license remains unchanged. `docs/source-manifest.json`
records checksums of retained files at initial curation; authored documentation,
analysis tools and subsequent curation additions are not claimed as upstream files.

## Excluded from the clean repository

- The archived `.venv` and its installed dependencies, binaries and interpreter links.
- `__pycache__` and bytecode.
- `train_colony_old.py`: superseded bootstrap version before the Constant-node fix.
- `train_colony_selfplay.py`: superseded self-play version before the value clamp
  and pending-iteration recovery. The final `_new.py` version is the active one.
- Original ZIP/PDF packaging duplicates. The useful supplied rule/encoding/contract
  documents are already retained as Markdown.
- All unprovided training-directory bytes: models and logs are not manufactured
  from a filename listing. The collector adds selected real files later.

“Excluded” means omitted from this prepared copy. It does not mean deleting the
owner's originals or training backups.

## Result evidence

- `docs/images/hall-of-fame.png`: team `42Heilbronn_GTA6`, All countries filter,
  overall rank 24, score 2112.
- `docs/images/germany-ranking.png`, added subsequently, shows the Germany
  filter selected and the team leading the listed German entries. Its rank
  column retains overall rank numbers, including the team's overall rank 24.
- `docs/images/tournament-dashboard.png`: an earlier in-contest view; its selected
  Since last upload filter makes the displayed counts unsuitable as lifetime totals.
- The two game exports have no duplicate match IDs. All 53 available protocols
  replay successfully through the supplied engine. Seven rows contain metadata only.
- Raw `score` agrees with CYAN's winner in those 53 records; our result summaries
  are derived from protocol winner and `you`, leaving missing protocols unknown.

## Reproducibility limits

The prepared archive is sufficient to inspect the code, replay the exported games,
and start new training after installing dependencies. The team confirmed `best_8.onnx` as the final tournament submission. Training
continued to 55,000 games and the `best_10.onnx` snapshot. Actual model, weight and
log bytes are still imported from the owner's external training directory using
the collector; the latest local champion remains distinct from the final upload.
