# Chat name

NeuMF Model

## Purpose

Implements the NeuMF (NCF) preference model inside the REF framework: a learned, non-linear directional score fusing a GMF branch and an MLP branch. It is the second interchangeable plug-in behind the same preference-model interface and emits the same ModelArtifact schema.

## Dependencies

- Architecture (committed): imports `ProcessedInteraction`, `UserIndex`, the `PreferenceModel` Protocol, `ModelArtifact`, and config from `src/core/`, plus the shared `conftest.py` fixture.
- May run in parallel with Data, MF, and Evaluation. Forbidden from importing `data/` internals, `models/mf/`, or `eval/`. It must conform to the same interface and artifact contract as MF without sharing code with it.

## Initial Plan Mode prompt

```
You are implementing the NeuMF preference model (He et al., NCF 2017) of the reciprocal-rec project, in models/neumf/. This is a Plan Mode session: produce a plan only, no implementation code.

The shared contracts in src/core/ are frozen. Import and conform to them; do not modify them. You implement the SAME PreferenceModel Protocol that the matrix-factorization model implements: fit(interactions: list[ProcessedInteraction]) -> None; directional_score(user_u: str, target_v: str) -> float; save(path) -> None; load(path) -> PreferenceModel. You emit a ModelArtifact with model_name "neumf". Do not import or depend on models/mf/; the two models share only the interface and the artifact schema.

REF context (implement only this model's directional score):
- Each user u has a source embedding p_u and a target embedding q_u, but NeuMF keeps SEPARATE embeddings per branch:
  - GMF branch: element-wise product p_u^G ⊙ q_v^G
  - MLP branch: concatenate [p_u^M ; q_v^M], then a tower of stacked ReLU layers
- Concatenate the GMF and MLP last hidden layers, project through a single sigmoid output to produce s(u->v) in (0,1).
- The aggregation f that fuses directions into r(A,B) lives in evaluation, NOT here. This model produces directional scores only. No aggregation, no reciprocal score here.

Scope, strictly limited to models/neumf/:
1. Define the GMF, MLP, and fusion modules and the forward pass producing s(u->v). Use PyTorch (or justify an alternative). Layer sizes / tower depth come from config (extend config only via the existing hyperparameters mechanism, not by editing src/core types).
2. Train with binary cross-entropy (log loss) over the binary labels from ProcessedInteraction, using the train split only (assert trained_on_split == "train"); use val for early stopping/monitoring; never touch test. Use config negative_downsample_ratio for the negative set.
3. Optional: support pretraining GMF and MLP separately and using them to initialize NeuMF; if included, keep it behind a flag and test both paths.
4. directional_score(u, v) runs the forward pass for the (u, v) pair via the UserIndex.
5. save() writes a ModelArtifact: put p_u/q_u-style embeddings in source_embeddings/target_embeddings where they map cleanly, and place the remaining learned tensors (MLP weights, output layer, both branch embedding tables) in the artifact's generic extra field so the schema does not change. load() must fully reconstruct a working model; save/load round-trip must reproduce directional_score within float tolerance.

Constraints:
- Import everything shared from src/core/. Do not redefine shared types or interfaces. Do not change the ModelArtifact schema; use its extra field for model-specific tensors.
- Do not import eval/, experiments/, models/mf/, or data/ internals; receive interactions as list[ProcessedInteraction].
- No aggregation, reciprocal score, or metrics here.
- Apply the code-structure skill: the model class is the orchestration surface (fit/score/save/load); factor the branches, the training step, BCE loss, and artifact (de)serialization into explicit-input service functions.
- Type hints throughout. Deterministic given config.random_seed (seed torch, numpy, and Python RNGs).

Produce: the file list with one-line descriptions, the network/forward-pass design, exactly how the artifact's extra field stores the model so it round-trips, the test cases (forward pass output shape and range in (0,1), BCE loss decreases on the synthetic fixture, deterministic under a seed, save/load round-trip reproduces directional_score, optional pretraining path), and decisions to confirm (tower depth defaults, pretraining on/off).
```

## Why this chat exists

NeuMF is the non-linear counterpart in the central comparison, and it is structurally heavier than MF (multiple embedding tables, MLP weights, BCE training). Isolating it forces it to satisfy the exact same interface and artifact contract as MF without sharing implementation, which is precisely what proves the preference model is a true plug-in. A separate chat also keeps PyTorch and training complexity out of the lean MF and data sessions.
