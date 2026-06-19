# Chat name

MF Model

## Purpose

Implements the matrix-factorization preference model inside the REF framework: a linear, dot-product directional score. It is one interchangeable plug-in behind the shared preference-model interface and emits a standard ModelArtifact.

## Dependencies

- Architecture (committed): imports `ProcessedInteraction`, `UserIndex`, the `PreferenceModel` Protocol, `ModelArtifact`, and config from `core/`, plus the shared `conftest.py` fixture.
- May run in parallel with Data, NeuMF, and Evaluation. Forbidden from importing `data/` internals or `eval/`; it consumes `list[ProcessedInteraction]` through the contract, not the loader implementation.

## Initial Plan Mode prompt

```
You are planning the matrix-factorization preference model of the reciprocal-rec project, in models/mf/. This is a Plan Mode session: produce a plan only, no implementation code. After I review the plan and click Build, implement it under the Build requirements at the end of this prompt.

The shared contracts in core/ are frozen. Import and conform to them; do not modify them. You implement the PreferenceModel Protocol: fit(interactions: list[ProcessedInteraction]) -> None; directional_score(user_u: str, target_v: str) -> float; save(path) -> None; load(path) -> PreferenceModel. You emit a ModelArtifact with model_name "mf".

REF context (do not re-derive the framework, just implement this model's directional score):
- Each user u has a source embedding p_u and a target embedding q_u.
- This model's directional score is the inner product s(u->v) = p_u^T q_v.
- The aggregation f that fuses s(A->B) and s(B->A) into r(A,B) lives in the evaluation module, NOT here. This model only produces directional scores and embeddings. Do not implement aggregation or any reciprocal score here.

Scope, strictly limited to models/mf/:
1. A weighted matrix factorization trainer over the binary labels from ProcessedInteraction: minimize a regularized squared loss with L2 regularization on p_u and q_v, optimized with SGD/Adam. Hyperparameters (embedding_dim, learning_rate, epochs, L2 weight, negative_downsample_ratio) come from config.
2. Use the training split only (assert trained_on_split == "train"). Use the val split for early-stopping/monitoring if you choose; never touch test.
3. directional_score(u, v) = p_u^T q_v using the learned embeddings and the UserIndex.
4. save() writes a ModelArtifact (source_embeddings = p, target_embeddings = q, hyperparameters, sampling_strategy, user_index, model_name "mf"); load() reconstructs a working model from it. Round-trip save/load must reproduce directional_score exactly (within float tolerance). MF needs no score_program — the artifact default (dot product) already reproduces s(u->v).

Constraints:
- Import everything shared from core/. Do not redefine shared types or interfaces.
- Do not import from eval/ or experiments/. Do not import data/ internals; receive interactions as list[ProcessedInteraction] only.
- No aggregation, no reciprocal score, no metrics here. Those belong to evaluation.
- Apply the code-structure skill: the model class is the orchestration surface (fit/score/save/load); factor reusable mechanics (the SGD update step, loss computation, embedding init, artifact (de)serialization helpers) into explicit-input service functions.
- Type hints throughout. Deterministic given config.random_seed.

Produce: the file list with one-line descriptions, the training-loop design, the exact ModelArtifact contents this model writes, the test cases (loss decreases on the synthetic fixture, deterministic training under a seed, directional_score matches a hand-computed dot product on tiny embeddings, save/load round-trip preserves scores), and any decisions to confirm (loss weighting, how negatives enter the objective).

Build requirements (apply when implementing after approval):
Implement models/mf/ strictly test-first. Do not write implementation code before its test exists and fails.
Workflow, repeat per capability (embedding init, scoring, training step, fit, save/load):
1. Write the pytest tests first, using the shared synthetic fixture from conftest.py. Cover: directional_score equals a hand-computed p_u^T q_v on tiny fixed embeddings, training loss decreases over epochs on the fixture, training is deterministic under config.random_seed, and save/load round-trips reproduce directional_score within tolerance. Add a golden test: core.scoring.verify_scorer_matches_directional(artifact, model.directional_score, pairs) passes after save/load.
2. Run pytest and show the tests failing for the right reason before writing any implementation.
3. Implement the minimum code to pass, conforming to the PreferenceModel Protocol and emitting a ModelArtifact with model_name "mf".
4. Run pytest, then `python -m importlinter.cli`, mypy, and ruff; all must pass.
Hard rules:
- Never modify a test to make a failing implementation pass; if a test is genuinely wrong, fix it deliberately and re-verify it fails for the right reason first.
- Do not redefine or edit core/. Do not import eval/ or experiments/; do not reach into data/ internals.
- Implement no aggregation, reciprocal score, or metrics here.
- Keep fit/score/save/load as the public surface; keep the update step, loss, and (de)serialization as separate service functions (code-structure skill).
- Type hints everywhere; respect config.random_seed.
When done, commit models/mf/ and report the directional_score signature and the exact ModelArtifact fields written, so evaluation can consume the artifact without importing this module.
```

## Why this chat exists

MF is one of two interchangeable preference models, and the entire point of the project is to compare it against NeuMF with everything else held fixed. Giving it its own chat enforces that it touches nothing but the shared interface, so swapping models never leaks into data or evaluation. Keeping it linear and small also makes it the reference implementation that proves the artifact contract works.
