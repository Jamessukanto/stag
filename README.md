# reciprocal-rec

A modular reciprocal recommendation system and configurable evaluation pipeline.
It compares a naive reciprocal aggregation model against the Reciprocal Embedding
Framework (REF), across random and popularity-biased negative sampling.

## Architecture

Three independently testable modules communicate only through serialized
artifacts on disk:

```
data/  ->  models/  ->  artifacts/*.json  ->  eval/
                 \________ src/core (shared contracts) ________/
```

- `src/core/` - stable shared contracts (types, interfaces, serialization, config). No business logic.
- `data/` - loads Libimseti, temporal splitting, negative sampling.
- `models/` - `naive.py` and `ref.py`, each producing a `ModelArtifact`.
- `eval/` - loads `ModelArtifact` files and computes reciprocal metrics.
- `experiments/` - orchestrates the modules into reproducible runs.

### Module boundary rules (enforced, not by convention)

- `eval/` must NEVER import `models/` (the hard invariant). It only reads `ModelArtifact` JSON.
- `eval/` must not import the `data/` implementation either.
- `data/` and `models/` are independent siblings; neither imports the other.
- `src/core/` depends on nothing downstream.

These are enforced by import-linter contracts in `pyproject.toml` and exercised
in `tests/test_import_boundaries.py`.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Checks

```bash
pytest            # tests (includes the import-boundary guard)
lint-imports      # architectural contracts
mypy              # type checking
ruff check .      # linting
```
