# Chat name

API

## Purpose

A thin read-only service that serves reciprocal recommendations from already-trained model artifacts, ranking candidate profiles by r(A,B). It adds a delivery layer on top of the core system without the core ever depending on it.

## Dependencies

- Architecture, the model artifacts produced by Experiments, and the Evaluation module's aggregation + scoring (reused for ranking). All committed and stable.
- Imports shared contracts from `src/core/` and reads `ModelArtifact`s from disk. It must not modify training, data, or evaluation code, and core modules must never import the API.

## Initial Plan Mode prompt

```
You are implementing the api/ module of the reciprocal-rec project. This is a Plan Mode session: produce a plan only, no implementation code.

All core modules are complete and frozen. The API is a leaf: nothing in src/core/, data/, models/, eval/, or experiments/ may import it. It loads trained ModelArtifacts from disk and serves recommendations; it performs no training.

Scope, strictly limited to api/:
1. A small HTTP service (FastAPI preferred) that on startup loads one or more ModelArtifacts from artifacts/ (selected by config: which model_name, which aggregation).
2. Endpoints:
   - GET /health -> liveness.
   - GET /recommend?user_id=...&k=...&aggregation=... -> top-K recommended target user_ids for the given user, ranked by the reciprocal score r(A,B) = f(s(A->B), s(B->A)). Reuse the aggregation and artifact-scoring logic from eval/ via its public interface; do not reimplement scoring, aggregation, or metrics.
   - Optionally GET /score?user_a=...&user_b=... -> the reciprocal score for a pair.
3. Request/response models are typed (Pydantic). Validate inputs (unknown user_id -> 404, bad k/aggregation -> 422). No write endpoints.

Constraints:
- Import shared types from src/core/ and reuse eval/'s public aggregation + scoring surface for ranking. Do not duplicate scoring/aggregation logic. Do not import data/ or models/ internals.
- Apply the code-structure skill: route handlers are the orchestration surface (parse request, authorize/validate, shape response); a thin service layer owns reusable mechanics (artifact loading/caching, candidate generation, ranking). Handlers stay slim.
- Type hints throughout. Keep it stateless beyond the loaded read-only artifacts.

Produce: the file list with one-line descriptions, the endpoint contracts (params, response schemas, error codes), how artifacts are loaded/cached at startup, the test cases (health, recommend returns K ranked ids, scoring matches eval's r(A,B) on a known artifact, validation/error paths via the test client), and decisions to confirm (framework choice, candidate set for /recommend).
```

## Why this chat exists

Serving concerns (HTTP, validation, artifact loading, caching) are entirely orthogonal to modeling and evaluation, and pulling them into a core chat would invert the dependency direction and tempt scoring logic to drift into the web layer. A separate leaf chat keeps the API replaceable and guarantees the research core stays runnable without any web stack.
