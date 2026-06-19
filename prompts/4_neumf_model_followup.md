# Follow-up implementation prompt — NeuMF Model

Paste this after you approve the plan, to move from planning to implementation under TDD.

```
The plan is approved. Implement models/neumf/ strictly test-first. Do not write implementation code before its test exists and fails.

Workflow, repeat per capability (GMF branch, MLP branch, fusion/forward, fit with BCE, save/load, optional pretraining):
1. Write the pytest tests first, using the shared synthetic fixture from conftest.py. Cover: forward pass returns a scalar in (0,1) with the right shapes, BCE loss decreases over epochs on the fixture, training is deterministic under config.random_seed (seed torch/numpy/Python), and save/load round-trips reproduce directional_score within tolerance. If pretraining is included, test both the pretrained-init and from-scratch paths.
2. Run pytest and show me the tests failing for the right reason before writing any implementation.
3. Implement the minimum code to pass, conforming to the PreferenceModel Protocol and emitting a ModelArtifact (model_name "neumf") that stores model-specific tensors in the artifact's extra field.
4. Run pytest again and show all tests passing.

Hard rules:
- Never modify a test to make a failing implementation pass. If a test is genuinely wrong, fix it deliberately and re-verify it fails for the right reason first.
- Do not redefine or edit src/core/. Do not change the ModelArtifact schema. Do not import eval/, experiments/, models/mf/, or data/ internals.
- Implement no aggregation, reciprocal score, or metrics here.
- Keep fit/score/save/load as the public surface; keep branches, training step, loss, and (de)serialization as separate service functions (code-structure skill).
- Type hints everywhere; keep mypy and ruff clean; fully seed all RNGs.

When done, commit models/neumf/ and report the directional_score signature and exactly how the artifact (including extra) is laid out, so evaluation can score from the artifact without importing this module.
```
