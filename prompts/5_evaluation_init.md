Purpose
Implements the eval/ module: loading model artifacts from disk and computing Reciprocal Precision@K and Mutual Hit Rate. This module has zero knowledge of how models are trained.

Dependencies
Architecture chat (src/core/types.py, src/core/serialization.py, src/core/config.py). No dependency on Data, NaiveModel, or REFModel chats.

You are implementing the eval/ module of a reciprocal recommendation system. This is a Plan Mode session. Produce a plan only — no code.

Critical architectural constraint — read this first:
eval/ must never import from models/. This is a hard boundary enforced at code review and in CI. The evaluator does not know that NaiveReciprocal or REFModel exist. It knows only about ModelArtifact (src/core/serialization.py), and it loads artifacts from disk paths.

Module responsibility:
- Load a ModelArtifact from a file path using ModelArtifact.load() from src/core/serialization.py.
- From the artifact's stored embeddings and user index, reconstruct recommendation rankings for each user (top-K by score).
- Compute two metrics over the test-split users:
    1. Reciprocal Precision@K: for a given user u, take their top-K recommended users. For each recommended user v in that list, check whether u also appears in v's top-K. RP@K is the fraction of top-K recommendations satisfying this mutual condition. Average over all users.
    2. Mutual Hit Rate: binary — does at least one mutual match exist in any user's top-K list? Average over all users (i.e., fraction of users with at least one mutual hit).
- Return an EvaluationResult (src/core/types.py) with model_name, sampling_strategy, k, reciprocal_precision_at_k, mutual_hit_rate, evaluated_at.
- Support batch evaluation: given a directory of artifact files, evaluate all and return list[EvaluationResult].

Constraints:
- Lives in eval/. Do not create files outside eval/ or tests/.
- Zero imports from models/. Zero imports from data/ (the implementation module).
- Do not retrain or call any model training logic. Work only with the stored embeddings in the artifact.
- The evaluator must verify that the artifact's trained_on_split is "train" before evaluating. If not, raise a descriptive error.
- K must come from the function argument, not be hardcoded.
- All type hints required.

Plan must address:
1. Files under eval/ and their responsibilities.
2. How top-K rankings are reconstructed from stored embeddings (vectorised or looped? what complexity?).
3. Exact formal definitions for RP@K and MHR as you will implement them — spell out edge cases (K larger than the number of users, users with no interactions in test).
4. How the batch evaluator handles artifacts with different model_names (it should treat them uniformly).
5. How EvaluationResult is written to disk (JSON? CSV?).

