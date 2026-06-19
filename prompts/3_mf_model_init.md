# Chat name

MF Model

## Purpose

Implements the matrix-factorization preference model inside the REF framework: a linear, dot-product directional score. It is one interchangeable plug-in behind the shared preference-model interface and emits a standard ModelArtifact.

## Dependencies

- Architecture (committed): imports `ProcessedInteraction`, `UserIndex`, the `PreferenceModel` Protocol, `ModelArtifact`, and config from `src/core/`, plus the shared `conftest.py` fixture.
- May run in parallel with Data, NeuMF, and Evaluation. Forbidden from importing `data/` internals or `eval/`; it consumes `list[ProcessedInteraction]` through the contract, not the loader implementation.

## Initial Plan Mode prompt

```
You are implementing the matrix-factorization preference model of the reciprocal-rec project, in models/mf/. This is a Plan Mode session: produce a plan only, no implementation code.

The shared contracts in src/core/ are frozen. Import and conform to them; do not modify them. You implement the PreferenceModel Protocol: fit(interactions: list[ProcessedInteraction]) -> None; directional_score(user_u: str, target_v: str) -> float; save(path) -> None; load(path) -> PreferenceModel. You emit a ModelArtifact with model_name "mf".

REF context (do not re-derive the framework, just implement this model's directional score):
- Each user u has a source embedding p_u and a target embedding q_u.
- This model's directional score is the inner product s(u->v) = p_u^T q_v.
- The aggregation f that fuses s(A->B) and s(B->A) into r(A,B) lives in the evaluation module, NOT here. This model only produces directional scores and embeddings. Do not implement aggregation or any reciprocal score here.

Scope, strictly limited to models/mf/:
1. A weighted matrix factorization trainer over the binary labels from ProcessedInteraction: minimize a regularized squared loss with L2 regularization on p_u and q_v, optimized with SGD/Adam. Hyperparameters (embedding_dim, learning_rate, epochs, L2 weight, negative_downsample_ratio) come from config.
2. Use the training split only (assert trained_on_split == "train"). Use the val split for early-stopping/monitoring if you choose; never touch test.
3. directional_score(u, v) = p_u^T q_v using the learned embeddings and the UserIndex.
4. save() writes a ModelArtifact (source_embeddings = p, target_embeddings = q, hyperparameters, sampling_strategy, user_index, model_name "mf"); load() reconstructs a working model from it. Round-trip save/load must reproduce directional_score exactly (within float tolerance).

Constraints:
- Import everything shared from src/core/. Do not redefine shared types or interfaces.
- Do not import from eval/ or experiments/. Do not import data/ internals; receive interactions as list[ProcessedInteraction] only.
- No aggregation, no reciprocal score, no metrics here. Those belong to evaluation.
- Apply the code-structure skill: the model class is the orchestration surface (fit/score/save/load); factor reusable mechanics (the SGD update step, loss computation, embedding init, artifact (de)serialization helpers) into explicit-input service functions.
- Type hints throughout. Deterministic given config.random_seed.

Produce: the file list with one-line descriptions, the training-loop design, the exact ModelArtifact contents this model writes, the test cases (loss decreases on the synthetic fixture, deterministic training under a seed, directional_score matches a hand-computed dot product on tiny embeddings, save/load round-trip preserves scores), and any decisions to confirm (loss weighting, how negatives enter the objective).
```

## Why this chat exists

MF is one of two interchangeable preference models, and the entire point of the project is to compare it against NeuMF with everything else held fixed. Giving it its own chat enforces that it touches nothing but the shared interface, so swapping models never leaks into data or evaluation. Keeping it linear and small also makes it the reference implementation that proves the artifact contract works.
