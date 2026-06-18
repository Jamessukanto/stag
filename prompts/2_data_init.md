Purpose
Implements the data/ module: loading Libimseti, temporal splitting, and both negative sampling strategies. Produces ProcessedInteraction lists that models consume.

Dependencies
Architecture chat (src/core/types.py, src/core/interfaces.py, src/core/config.py).

You are implementing the data/ module of a reciprocal recommendation system. This is a Plan Mode session. Produce a plan only — no code.

Module responsibility (data/ exclusively):
- Load the Libimseti dataset: a tab-separated file of directional ratings with columns user_id, target_id, rating, timestamp.
- Parse into list[RawInteraction] (defined in src/core/types.py — do not redefine this type).
- Split into train/val/test using temporal cutoffs. The split must be timestamp-ordered: all train interactions precede all val interactions, which precede all test interactions. No shuffling. This prevents data leakage.
- Expose a NegativeSampler with two strategies:
    - random: sample user IDs uniformly from users the target user has not interacted with.
    - popularity_biased: sample proportionally to interaction count (popular users appear more often as hard negatives).
  Both strategies must share the same interface and be swappable at runtime without changing the DataLoader.
- Conform to the DataLoader Protocol in src/core/interfaces.py.

Constraints:
- All new files live under data/ or tests/. Do not create files elsewhere.
- Do not import from models/ or eval/ anywhere.
- Do not redefine RawInteraction, ProcessedInteraction, or UserIndex. Import them from src/core/types.py.
- Do not hardcode file paths. Consume them from src/core/config.py.
- Negative sampling must be deterministic given a seed.
- All type hints required. No untyped function signatures.

Plan must address:
1. Files under data/ and their responsibilities.
2. Exact method signatures on the DataLoader implementation class.
3. How the two sampling strategies are structured (strategy pattern? subclassing? callable injection?). Justify the choice.
4. How temporal splits are validated: what check prevents train/val/test timestamp ranges from overlapping?
5. How the module handles users present in test but absent from train (cold-start users). Decide: raise, skip, or flag — and state the reason.
6. Whether interaction data is cached after first load, and if so, how.

