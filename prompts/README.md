`0_initial_meta_prompt.md` plans every Cursor chat session. All other files in the folder are its outputs: paste `*_init.md` into a fresh Plan Mode chat, approve the plan, then paste the matching `*_followup.md` to implement.

Each session owns one module. Sessions communicate only through committed artifacts and `src/core/` contracts. Every prompt enforces pytest, type hints, TDD, and never modifying tests to fit implementations.

## Ordering

```
[1] Architecture
        │  commits src/core/ + conftest.py
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

| # | Session | Init | Follow-up |
|---|---------|------|-----------|
| 1 | Architecture | `1_architecture.md` | in `1_architecture.md` |
| 2 | Data | `2_data_init.md` | `2_data_followup.md` |
| 3 | MF Model | `3_mf_model_init.md` | `3_mf_model_followup.md` |
| 4 | NeuMF Model | `4_neumf_model_init.md` | `4_neumf_model_followup.md` |
| 5 | Evaluation | `5_evaluation_init.md` | `5_evaluation_followup.md` |
| 6 | Experiments | `6_experiments_init.md` | `6_experiments_followup.md` |
| 7 | API | `7_api_init.md` | `7_api_followup.md` |
| 8 | Frontend | `8_frontend_init.md` | `8_frontend_followup.md` |

Hard invariant: `eval/` never imports `models/`; modules never import each other's internals.
