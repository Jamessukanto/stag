Purpose
Implements the Reciprocal Embedding Framework (REF, Wan et al. AAAI 2021) in models/ref.py: a jointly trained embedding model where the reciprocal structure is encoded during training, not assembled post-hoc.

Dependencies
Architecture chat. The REF paper at https://cdn.aaai.org/ojs/17807/17807-13-21301-1-2-20210518.pdf. Does not depend on the NaiveModel chat.

You are implementing the REF model in models/ref.py. This is a Plan Mode session. Produce a plan only — no code.

Reference paper: Wan et al., "Reciprocal Recommendation System for Online Dating", AAAI 2021.
URL: https://cdn.aaai.org/ojs/17807/17807-13-21301-1-2-20210518.pdf
Read this paper before planning. The implementation must match its formulation, not a generic matrix factorisation baseline.

Model description:
Each user has two embeddings:
  p_u — source embedding
  q_u — target embedding
Unlike naive aggregation, p_u and q_u are trained jointly over the same interaction graph. The reciprocal score is:
  r(A,B) = p_A · q_B
where both embeddings emerge from a single joint training objective.

The key distinction from naive.py: there is no post-hoc aggregation. Reciprocity is encoded in the embedding space during training.

Module responsibility:
- Implement class REFModel conforming to the same ReciprocityModel Protocol as models/naive.py (src/core/interfaces.py).
- On fit(), train using the joint objective from the paper over list[ProcessedInteraction].
- Expose predict(user_a, user_b) -> float returning r(A,B).
- On save(), write a ModelArtifact using src/core/serialization.ModelArtifact. The artifact schema is identical in structure to the naive model artifact, with model_name "ref".
- On load(), restore the model for inference without retraining.

Constraints:
- Lives in models/ref.py only. Do not create files outside models/ or tests/.
- Do not import from eval/ or from data/ (the module).
- Do not import from models/naive.py. The two models are siblings under a shared Protocol, not a hierarchy. No shared training utilities.
- The artifact JSON schema must be structurally identical to the naive model's (same required keys). The values will differ.
- Do not redefine types from src/core/types.py.

Plan must address:
1. The joint training objective: write out the loss function as it appears in or is derived from the paper.
2. How the interaction graph is batched during training (how are positive and negative pairs structured per step?).
3. Key structural differences from the naive training loop — specifically, what makes this "joint" and not two sequential passes.
4. What is stored in the artifact and how it maps to the ModelArtifact fields.
5. How predict() is called at inference time — any differences from naive.py at the interface level? (There should be none.)

