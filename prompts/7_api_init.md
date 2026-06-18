Purpose
Implements a read-only FastAPI layer over pre-computed artifacts, serving recommendation queries and evaluation summaries to the React frontend. Does not trigger training.

Dependencies
Architecture chat. Experiments chat (for the artifact directory layout convention).

You are planning the api/ module of a reciprocal recommendation system. This is a Plan Mode session. Produce a plan only — no code.

This chat is a future-facing session. The API makes pre-computed artifacts queryable over HTTP. It does not trigger model training. It is read-only.

Module responsibility:
- Load ModelArtifact files from the artifacts/ directory using src/core/serialization.ModelArtifact.load().
- Serve these endpoints:
    GET /artifacts — list available artifact identifiers (model, sampling strategy, created_at).
    GET /artifacts/{artifact_id}/recommendations/{user_id}?k=10 — return top-K reciprocal recommendations for a user.
    GET /artifacts/{artifact_id}/evaluation — return the EvaluationResult for this artifact.
    GET /health — returns 200 OK.
- Accept the artifacts base directory as an environment variable ARTIFACT_DIR (no hardcoded paths).

Constraints:
- Lives in api/. Do not create files outside api/ or tests/.
- No imports from models/ or eval/ (the modules). Load artifacts only via src/core/serialization.
- No side effects on artifact files. All endpoints are GET-only.
- Use FastAPI. Define Pydantic response models for all endpoints. Do not return raw ModelArtifact objects.
- Designed for a React frontend to consume. CORS must be configurable via environment variable.
- Authentication is a stub: include an API_KEY env var check that currently always passes if the var is unset.

Plan must address:
1. All endpoint paths, HTTP methods, request params, and response schemas.
2. How artifact_id is derived from artifact file paths (hash? directory name? timestamp?).
3. How top-K recommendations are reconstructed at request time from stored embeddings.
4. Error response format (must be consistent across all 404 and 422 cases).
5. CORS configuration.

