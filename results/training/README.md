# Training records to add from the external run

The supplied text listing establishes filenames, not file contents. Run the
collector to add real `history.jsonl`, a redacted `state-summary.json`, validation
logs, and a hash index of local candidates and numbered snapshots.

`state-summary.json` is documentation only: machine-specific prefixes are redacted.
To resume the original run, retain the original full training directory outside
GitHub and pass its path to the trainer. A selected `submission.pt` can initialize
a new compatible run but is not a replacement for the old run's full resume state.
