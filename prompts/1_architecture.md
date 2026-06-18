Purpose
Defines the project scaffold, shared type contracts, serialization schema, abstract interfaces, and test infrastructure. Everything downstream depends on what this chat produces — it owns no business logic whatsoever.

You are scaffolding a Python project called reciprocal-rec. This is a Plan Mode session. Produce a plan only — no implementation code.

Project summary:
A modular reciprocal recommendation system. It has three independently testable modules: data/, models/, and eval/. A hard architectural invariant governs them: eval/ must never import from models/. These modules communicate only through serialized artifacts on disk. This boundary must be enforced structurally, not by convention.

Your task is to define the shared foundation that all three modules depend on. Scope is strictly limited to:

1. Directory layout
   - Top-level: data/, models/, eval/, experiments/, tests/, artifacts/, src/core/
   - Each module has its own __init__.py
   - A top-level conftest.py and pytest.ini
   - A pyproject.toml with pytest, mypy, and ruff configured

2. Shared types in src/core/types.py
   Define typed dataclasses or Pydantic models for:
   - RawInteraction: user_id (str), target_id (str), rating (float), timestamp (int)
   - ProcessedInteraction: same fields plus split label ("train" | "val" | "test")
   - UserIndex: bidirectional mapping between user_id strings and integer indices
   - ModelArtifact: the serialization schema every model must output (see point 4)
   - EvaluationResult: model_name, sampling_strategy, k, reciprocal_precision_at_k (float), mutual_hit_rate (float), evaluated_at (str timestamp)

3. Protocols in src/core/interfaces.py
   Use typing.Protocol for:
   - DataLoader: methods load() -> list[ProcessedInteraction], get_negatives(user_id: str, strategy: str, n: int, seed: int) -> list[str]
   - ReciprocityModel: methods fit(interactions: list[ProcessedInteraction]) -> None, predict(user_a: str, user_b: str) -> float, save(path: Path) -> None, load(path: Path) -> None
   - Evaluator: method evaluate(artifact_path: Path, k: int) -> EvaluationResult

4. Serialization contract in src/core/serialization.py
   ModelArtifact must include:
   - model_name: str ("naive" | "ref")
   - sampling_strategy: str ("random" | "popularity_biased")
   - hyperparameters: dict[str, Any]
   - user_index: UserIndex (serialized form)
   - source_embeddings: list[list[float]]  # p_u, shape [n_users, emb_dim]
   - target_embeddings: list[list[float]]  # q_u, shape [n_users, emb_dim]
   - trained_on_split: str  # must be "train"
   - created_at: str  # ISO timestamp
   Provide ModelArtifact.save(path: Path) and ModelArtifact.load(path: Path) -> ModelArtifact as class methods. Use JSON as the serialization format so artifacts are human-readable.

5. Config schema in src/core/config.py
   A single dataclass (or Pydantic BaseSettings) covering:
   - data_path: Path
   - artifact_dir: Path
   - train_ratio: float, val_ratio: float, test_ratio: float
   - embedding_dim: int
   - learning_rate: float
   - epochs: int
   - k_values: list[int]
   - random_seed: int

Constraints:
- No model training logic here.
- No data loading logic here.
- No evaluation logic here.
- Do not place any domain logic in src/core/. These files are contracts, not implementations.
- All types defined here are stable. Downstream chats are forbidden from redefining them.
- Python 3.11+. Use dataclasses or Pydantic v2. Use typing.Protocol, not ABC, for interfaces.

Produce: a list of files with one-line descriptions, key design decisions to confirm (especially around the ModelArtifact JSON schema), and anything that would break downstream chats if changed later.