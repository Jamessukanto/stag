# Chat name

Architecture

## Purpose

Defines the project scaffold, shared typed contracts, the serialization schema, the abstract interfaces, and the test infrastructure that every downstream module depends on. This chat owns no business logic whatsoever — only the contracts that keep modules decoupled.

## Dependencies

None. This is the first session. Its committed artifacts (`core/`, `conftest.py`, `pyproject.toml`) are the foundation every other chat builds on. No other chat may start until this one's contracts are committed.

## Initial Plan Mode prompt

```
You are scaffolding a Python project called reciprocal-rec. This is a Plan Mode session: produce a plan only, no implementation code.

Project summary:
A modular reciprocal recommendation system built on the Reciprocal Embedding Framework (REF). Each user u carries two directional embeddings: a source embedding p_u (how u selects others) and a target embedding q_u (how others select u). A plug-in preference model produces a directional score s(u->v) from these embeddings. An aggregation function f fuses the two directions into one reciprocal score r(A,B) = f(s(A->B), s(B->A)). Two preference models will be implemented later (matrix factorization and NeuMF) behind one shared interface, plus a data module and an evaluation module.

Your task is to define the shared foundation all modules depend on. Scope is strictly limited to contracts and scaffolding. Hard architectural invariants you must encode structurally (not by convention):
- The evaluation module must never import the model modules. Evaluation consumes model outputs only as serialized artifacts read from disk.
- The data, model, and evaluation modules must never import each other's internals. They communicate only through the shared contracts you define here and through on-disk artifacts.
- The preference model is a plug-in: swapping matrix factorization for NeuMF must not require any change to data, aggregation, or evaluation code.

Define exactly the following, and nothing more:

1. Directory layout
   - Top-level: core/, data/, models/, eval/, experiments/, tests/, artifacts/
   - Each package has its own __init__.py
   - A top-level conftest.py and pyproject.toml (configure pytest, mypy, and ruff)

2. Shared types in core/types.py
   Typed dataclasses or Pydantic v2 models for:
   - RawInteraction: user_id (str), target_id (str), rating (int, 1-10 scale)
   - ProcessedInteraction: user_id (str), target_id (str), label (int, 1 = like / rating>=7, 0 = explicit dislike), split ("train" | "val" | "test")
   - UserIndex: bidirectional mapping between user_id strings and contiguous integer indices
   - EvaluationResult: model_name, aggregation, k, recall_at_k (float), hr_at_k (float), ndcg_at_k (float), evaluated_at (ISO timestamp str)

3. Protocols in core/interfaces.py (use typing.Protocol, not ABC)
   - DataLoader: load() -> list[ProcessedInteraction]; get_negatives(user_id: str, strategy: str, n: int, seed: int) -> list[str]
   - PreferenceModel: fit(interactions: list[ProcessedInteraction]) -> None; directional_score(user_u: str, target_v: str) -> float; save(path: Path) -> None; load(path: Path) -> "PreferenceModel". This is the single plug-in interface both MF and NeuMF implement.
   - Aggregator: aggregate(s_ab: float, s_ba: float) -> float  (the reciprocal fusion function f)
   - Evaluator: evaluate(artifact_path: Path, aggregation: str, k: int) -> EvaluationResult

4. Serialization contract in core/serialization.py
   A ModelArtifact schema that every preference model emits, decoupling models from evaluation:
   - model_name: str ("mf" | "neumf")
   - sampling_strategy: str ("random" | "popularity_biased")
   - hyperparameters: dict[str, Any]
   - user_index: serialized UserIndex
   - source_embeddings: list[list[float]]  # p_u, shape [n_users, emb_dim]
   - target_embeddings: list[list[float]]  # q_u, shape [n_users, emb_dim]
   - extra: dict[str, Any]  # model-specific tensors NeuMF needs beyond p_u/q_u (e.g. MLP weights), kept generic so the schema does not leak model internals
   - trained_on_split: str  # must be "train"
   - created_at: ISO timestamp str
   Provide ModelArtifact.save(path: Path) and ModelArtifact.load(path: Path) -> ModelArtifact. Use JSON so artifacts are human-readable. Decide and document how directional_score is reconstructed at evaluation time from an artifact without importing model code (e.g. each artifact carries enough to recompute scores, or models expose a pure scoring function the artifact references by name).

5. Config schema in core/config.py
   One dataclass (or Pydantic BaseSettings) covering: data_path, artifact_dir, train_ratio, val_ratio, test_ratio, embedding_dim, learning_rate, epochs, k_values (list[int]), negative_downsample_ratio, random_seed.

6. Test infrastructure
   - A shared synthetic interactions fixture in the top-level conftest.py (a small deterministic set of RawInteraction/ProcessedInteraction records) so the data, model, and evaluation chats all inherit one consistent fixture and never invent divergent toy data.
   - pytest configured to discover tests/ and per-module test directories.

Constraints:
- No data loading, model training, aggregation, or evaluation logic here. core/ holds contracts only.
- All types and interfaces defined here are stable. Downstream chats are forbidden from redefining them; they import from core/.
- Apply the code-structure skill: core/ is pure contract; it owns no orchestration and no mechanics.
- Python 3.11+. Use typing.Protocol for interfaces. Use dataclasses or Pydantic v2 for data.

Produce: the file list with one-line descriptions, the key design decisions to confirm (especially the ModelArtifact JSON schema and how evaluation reconstructs directional scores without importing models), and anything that would break downstream chats if changed later.
```

## Build requirements

After you review the plan and click **Build**, implement under TDD. (`core/` is contracts, but contracts are still code: serialization round-trips, the config schema, and the synthetic fixture all need tests.)

```
Implement core/ and the test infrastructure strictly test-first. Do not write implementation code before its test exists and fails.

Workflow, repeat per artifact (types, interfaces, serialization, config, conftest fixture):
1. Write the pytest tests first. Cover: every shared type constructs and validates its fields (e.g. label in {0,1}, split in {train,val,test}, rating 1-10); UserIndex maps user_id <-> integer index bijectively; ModelArtifact.save then load round-trips losslessly and reconstructs directional scores via the documented artifact-scoring contract WITHOUT importing any model module; the config schema parses defaults and rejects invalid ratios; and the shared synthetic fixture in conftest.py yields the documented deterministic records.
2. Run pytest and show me the tests failing for the right reason before writing any implementation.
3. Implement the minimum code to pass: the dataclasses/Pydantic models, the typing.Protocol interfaces (these need import-time and structural-conformance tests, not behavior), ModelArtifact (de)serialization, the config schema, and the conftest fixture.
4. Run pytest again and show all tests passing.

Hard rules:
- Never modify a test to make a failing implementation pass. If a test is genuinely wrong, fix it deliberately and re-verify it fails for the right reason first.
- core/ holds contracts only: no data loading, training, aggregation, or evaluation logic. Protocols are interfaces, not implementations.
- Once committed, these types, Protocols, the ModelArtifact schema, and the fixture are FROZEN. Downstream chats import them and may not redefine them, so design carefully now.
- Type hints everywhere; keep mypy and ruff clean. Python 3.11+.

When done, commit core/, conftest.py, and pyproject.toml, and report the frozen public surface: every type and its fields, every Protocol signature, the ModelArtifact schema, how evaluation reconstructs directional_score from an artifact without importing models, and the synthetic fixture's shape — so all downstream chats can build against it in parallel.
```

## Why this chat exists

Every other session imports these contracts, so they must be designed once, deliberately, and frozen. Defining them inside a feature chat would couple the contract to that feature and let the boundaries drift. Isolating the shared foundation is what makes the data, model, and evaluation chats able to run in parallel without colliding.
