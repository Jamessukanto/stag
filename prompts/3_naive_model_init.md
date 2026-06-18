Purpose
Implements the Naive Reciprocal Aggregation model in models/naive.py: two independently trained one-sided embedding models whose scores are combined post-hoc via a configurable aggregation function.

Dependencies
Architecture chat (src/core/types.py, src/core/interfaces.py, src/core/serialization.py, src/core/config.py). The data interface (not the implementation) from the Architecture chat's DataLoader Protocol.

You are implementing the NaiveReciprocal model in models/naive.py. This is a Plan Mode session. Produce a plan only — no code.

Model description:
Each user has two embeddings:
  p_u — source embedding: how user u rates others
  q_u — target embedding: how others rate user u

Two one-sided models are trained independently:
  s(A→B) = p_A · q_B   (dot product: how much A is predicted to like B)
  s(B→A) = p_B · q_A   (dot product: how much B is predicted to like A)

Reciprocal score: r(A,B) = f(s(A→B), s(B→A))
where f is one of:
  - dot_product: s(A→B) * s(B→A)
  - weighted_sum: alpha * s(A→B) + (1 - alpha) * s(B→A), where alpha is configurable

This module is models/naive.py only.

Module responsibility:
- Implement class NaiveReciprocal conforming strictly to the ReciprocityModel Protocol in src/core/interfaces.py.
- On fit(), train both one-sided models using BPR (Bayesian Personalised Ranking) loss or matrix factorisation, consuming list[ProcessedInteraction].
- Expose predict(user_a, user_b) -> float returning r(A,B).
- On save(), write a ModelArtifact to disk via src/core/serialization.ModelArtifact. The artifact must include both embedding matrices, the aggregation function identifier, alpha (if applicable), and all hyperparameters.
- On load(), restore the model from a ModelArtifact so predict() can be called without retraining.

Constraints:
- Lives in models/naive.py only. Do not create files outside models/ or tests/.
- Do not import from eval/.
- Do not import from data/ (the implementation). Accept list[ProcessedInteraction] as input — the caller handles data loading.
- Do not redefine any type from src/core/types.py.
- The aggregation function f must be injected or selected via config, not hardcoded. Two runs with different f values must produce different reciprocal scores without modifying training logic.
- models/naive.py must not know about models/ref.py. They are siblings, not a hierarchy.

Plan must address:
1. The training loop: are the two one-sided models trained in separate passes or interleaved? Justify.
2. How the aggregation function is selected (enum? callable? string key?). Write out the interface.
3. What UserIndex maps in the context of this model.
4. What exactly is serialized and how it maps to ModelArtifact fields.
5. How predict() behaves for a user_id not seen during training (raise or return a sentinel — decide and justify).

