# Incomplete auxiliary generator

`gen_data.py` was present beside the project in the uploaded archive. It imports
`oracle_evaluator` from `oracle.py`, which was not provided. Its original path
assumptions also refer to a different `train/` layout.

It is preserved here as historical source, not called by the README commands,
tests or self-play pipeline. Do not treat it as a reproducible alternative trainer
until its dependency and original context are supplied. No missing oracle was
invented during curation.
