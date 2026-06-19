# Chat name

Data

## Purpose

Owns everything that turns the raw Libimseti ratings into model-ready supervision: loading, binarization, per-user stratified splitting, and negative sampling. Produces processed data that conforms to the shared contracts and nothing else.

## Dependencies

- Architecture (committed): imports `ProcessedInteraction`, `RawInteraction`, `UserIndex`, `DataLoader`, and the config from `src/core/`, plus the shared `conftest.py` fixture.
- May run in parallel with the MF, NeuMF, and Evaluation chats. Forbidden from importing `models/` or `eval/`.

## Initial Plan Mode prompt

```
You are implementing the data/ module of the reciprocal-rec project. This is a Plan Mode session: produce a plan only, no implementation code.

The shared contracts already exist in src/core/ and are frozen. You must import and conform to them; do not redefine or modify them. Relevant contracts: RawInteraction, ProcessedInteraction (fields: user_id, target_id, label, split), UserIndex, the DataLoader Protocol (load() -> list[ProcessedInteraction]; get_negatives(user_id, strategy, n, seed) -> list[str]), and the config dataclass.

Scope, strictly limited to the data/ module:

1. Loading
   - Read the Libimseti ratings (user_id, target_id, rating on a 1-10 integer scale) into RawInteraction records.
   - Build a UserIndex over all users that appear as either a rater or a target.

2. Binarization
   - rating >= 7 -> label 1 (like / positive)
   - rating < 7  -> label 0 (explicit dislike / negative). These are explicit negatives from observed ratings, not sampled.

3. Splitting
   - Per-user stratified random holdout into train/val/test using the config ratios. Libimseti has no timestamps, so no temporal split.
   - Every user must be represented across the splits where they have enough interactions; document the fallback for users with too few interactions.
   - Splitting must be deterministic given config.random_seed.

4. Negative sampling (for ablation and training efficiency, NOT the source of negative signal)
   - Implement get_negatives with two strategies: "random" (uniform over non-interacted targets) and "popularity_biased" (sampled proportional to how often a target is rated).
   - Negative downsampling: trim the explicit-negative set per positive to config.negative_downsample_ratio when enabled. Make clear that explicit negatives from binarization remain the training signal; downsampling only trims them.
   - All sampling is seeded and reproducible.

5. Implement the DataLoader Protocol as the module's public entry point. This is the only surface other modules use.

Constraints:
- Import everything shared from src/core/. Do not redefine shared types.
- Do not import from models/ or eval/. The data module knows nothing about how scores are computed or evaluated.
- Apply the code-structure skill: the DataLoader is the orchestration/action surface (load, split, expose negatives); factor reusable mechanics (rating parsing, the two sampling strategies, the stratified splitter) into small explicit-input service functions with structured returns.
- Enforce type hints throughout.
- Use the shared synthetic fixture from conftest.py for tests; do not invent a separate toy dataset.
- Assume the raw file may be large; note where streaming or chunked reads matter, but keep the public interface unchanged.

Produce: the file list with one-line descriptions, the public API of the DataLoader, the test cases you will write (binarization threshold, deterministic seeded splits, per-user stratification coverage, both negative-sampling strategies, downsampling ratio), and any decisions that need confirming (e.g. handling of users with too few ratings to populate all splits).
```

## Why this chat exists

Data preparation is shared, model-agnostic, and the single biggest source of subtle leakage and reproducibility bugs (split contamination, seeding, sampling bias). Isolating it guarantees that MF and NeuMF train on identical supervision, which is the precondition for the comparison to isolate the preference model. It must never know how scores are computed.
