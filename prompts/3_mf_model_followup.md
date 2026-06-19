# Follow-up implementation prompt — MF Model

Paste this after you approve the plan, to move from planning to implementation under TDD.

```
The plan is approved. Implement models/mf/ strictly test-first. Do not write implementation code before its test exists and fails.

Workflow, repeat per capability (embedding init, scoring, training step, fit, save/load):
1. Write the pytest tests first, using the shared synthetic fixture from conftest.py. Cover: directional_score equals a hand-computed p_u^T q_v on tiny fixed embeddings, training loss decreases over epochs on the fixture, training is deterministic under config.random_seed, and save/load round-trips reproduce directional_score within tolerance.
2. Run pytest and show me the tests failing for the right reason before writing any implementation.
3. Implement the minimum code to pass, conforming to the PreferenceModel Protocol and emitting a ModelArtifact with model_name "mf".
4. Run pytest again and show all tests passing.

Hard rules:
- Never modify a test to make a failing implementation pass. If a test is genuinely wrong, fix the test deliberately and re-verify it fails for the right reason first.
- Do not redefine or edit src/core/. Do not import eval/ or experiments/; do not reach into data/ internals.
- Implement no aggregation, reciprocal score, or metrics here.
- Keep fit/score/save/load as the public surface; keep the update step, loss, and (de)serialization as separate service functions (code-structure skill).
- Type hints everywhere; keep mypy and ruff clean; respect config.random_seed.

When done, commit models/mf/ and report the directional_score signature and the exact ModelArtifact fields written, so evaluation can consume the artifact without importing this module.
```
