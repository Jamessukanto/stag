`0_initial_meta_prompt.md` plans every Cursor chat session. The numbered files are its outputs: one self-contained prompt per module.

## Workflow

Each `N_*_init.md` holds the **plan prompt** plus a **Build requirements** block (the TDD loop and hard rules the implementation must follow).

```
open N_*_init.md → paste into a fresh Plan Mode chat → review the plan → click Build → commit
```

There are no separate `*_followup.md` files: clicking **Build** is the implement step, and the Build requirements travel inside the init prompt. Project-wide TDD and boundary rules also live in `.cursor/rules/` so they apply on every Build automatically.

Each session owns one module. Sessions communicate only through committed artifacts and `core/` contracts.

## After each Build (your checklist)

1. `pytest`
2. `python -m importlinter.cli`
3. `mypy && ruff check .`
4. Confirm the module's public surface matches what downstream prompts expect (e.g. a model writes a `ModelArtifact` and passes the golden `verify_scorer_matches_directional` test).
5. Commit the module before starting the next parallel chat.

## If Build goes wrong

Don't re-run the init. Send a correction in the **same chat** (e.g. "revert X; write test Y first; it must fail for reason Z"). Re-running init restarts planning instead of fixing the build.

## Ordering

```
[1] Architecture
        │  commits core/ + conftest.py
        ▼
[2-5] parallel (depend only on Architecture)
    ├── Data
    ├── MF Model
    ├── NeuMF Model
    └── Evaluation
        │  each commits a stable, passing module
        ▼
[6] Experiments  →  [7] API  →  [8] Frontend
```

| # | Session | Prompt file |
|---|---------|-------------|
| 1 | Architecture | `1_architecture.md` |
| 2 | Data | `2_data_init.md` |
| 3 | MF Model | `3_mf_model_init.md` |
| 4 | NeuMF Model | `4_neumf_model_init.md` |
| 5 | Evaluation | `5_evaluation_init.md` |
| 6 | Experiments | `6_experiments_init.md` |
| 7 | API | `7_api_init.md` |
| 8 | Frontend | `8_frontend_init.md` |

Hard invariant: `eval/` never imports `models/` or `data/`; modules never import each other's internals. Enforced by `.cursor/rules/`, import-linter, and `tests/test_import_boundaries.py`.
