# Follow-up implementation prompt — Data

Paste this after you approve the plan, to move from planning to implementation under TDD.

```
The plan is approved. Implement the data/ module strictly test-first. Do not write implementation code before its test exists and fails.

Workflow, repeat per capability (loading, binarization, splitting, each negative-sampling strategy, downsampling):
1. Write the pytest tests first, using the shared synthetic fixture from conftest.py. Cover: binarization threshold at exactly 7, deterministic splits under a fixed seed, per-user stratification coverage, "random" and "popularity_biased" negatives, the downsample ratio, and that no train/val/test leakage occurs.
2. Run pytest and show me the tests failing for the right reason before writing any implementation.
3. Implement the minimum code to pass, conforming to the DataLoader Protocol and importing all shared types from src/core/.
4. Run pytest again and show all tests passing.

Hard rules:
- Never modify a test to make a failing implementation pass. If a test is wrong, explain why and fix the test deliberately, then re-verify it fails for the right reason.
- Do not redefine or edit anything in src/core/. Do not import from models/ or eval/.
- Keep the DataLoader as the only public surface; keep parsing/splitting/sampling as separate service functions with explicit inputs and structured outputs (code-structure skill).
- Type hints everywhere; keep mypy and ruff clean.

When done, commit the data/ module and report the public DataLoader signature so downstream chats can rely on it.
```
