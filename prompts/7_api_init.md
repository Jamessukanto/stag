# Chat name

API

## Purpose

A thin read-only service that serves reciprocal recommendations from already-trained model artifacts, ranking candidate profiles by r(A,B). It adds a delivery layer on top of the core system without the core ever depending on it.

## Dependencies

- Architecture, the model artifacts produced by Experiments, and the Evaluation module's aggregation + scoring (reused for ranking). All committed and stable.
- Imports shared contracts from `core/` and reads `ModelArtifact`s from disk. It must not modify training, data, or evaluation code, and core modules must never import the API.

## Initial Plan Mode prompt

```
You are planning the api/ module of the reciprocal-rec project. This is a Plan Mode session: produce a plan only, no implementation code. After I review the plan and click Build, implement it under the Build requirements at the end of this prompt.

All core modules are complete and frozen. The API is a leaf: nothing in core/, data/, models/, eval/, or experiments/ may import it. It loads trained ModelArtifacts from disk and serves recommendations; it performs no training.

Scope, strictly limited to api/:
1. A small HTTP service (FastAPI preferred) that on startup loads one or more ModelArtifacts from artifacts/ (selected by config: which model_name, which aggregation).
2. Endpoints:
   - GET /health -> liveness.
   - GET /recommend?user_id=...&k=...&aggregation=... -> top-K recommended target user_ids for the given user, ranked by the reciprocal score r(A,B) = f(s(A->B), s(B->A)). Reuse the aggregation and artifact-scoring logic from eval/ via its public interface; do not reimplement scoring, aggregation, or metrics.
   - Optionally GET /score?user_a=...&user_b=... -> the reciprocal score for a pair.
3. Request/response models are typed (Pydantic). Validate inputs (unknown user_id -> 404, bad k/aggregation -> 422). No write endpoints.

Constraints:
- Import shared types from core/ and reuse eval/'s public aggregation + scoring surface for ranking. Do not duplicate scoring/aggregation logic. Do not import data/ or models/ internals.
- Apply the code-structure skill: route handlers are the orchestration surface (parse request, authorize/validate, shape response); a thin service layer owns reusable mechanics (artifact loading/caching, candidate generation, ranking). Handlers stay slim.
- Type hints throughout. Keep it stateless beyond the loaded read-only artifacts.

Produce: the file list with one-line descriptions, the endpoint contracts (params, response schemas, error codes), how artifacts are loaded/cached at startup, the test cases (health, recommend returns K ranked ids, scoring matches eval's r(A,B) on a known artifact, validation/error paths via the test client), and decisions to confirm (framework choice, candidate set for /recommend).

Build requirements (apply when implementing after approval):
Implement api/ strictly test-first. Do not write implementation code before its test exists and fails.
Workflow, repeat per capability (artifact loading/caching, ranking service, each endpoint, validation):
1. Write the pytest tests first using the framework's test client and a small synthetic ModelArtifact written to a temp artifacts dir. Cover: /health, /recommend returns exactly K ranked target ids for a known user, the served ranking matches eval's r(A,B) on that artifact, /score (if included) matches eval, and validation paths (unknown user_id -> 404, bad k/aggregation -> 422).
2. Run pytest and show the tests failing for the right reason before writing any implementation.
3. Implement the minimum code to pass, reusing eval/'s public aggregation + scoring surface and loading artifacts from disk.
4. Run pytest, then `python -m importlinter.cli`, mypy, and ruff; all must pass.
Hard rules:
- Never modify a test to make a failing implementation pass; if a test is genuinely wrong, fix it deliberately and re-verify it fails for the right reason first.
- Do not redefine or edit core/. Do not reimplement scoring/aggregation/metrics; reuse eval/. Do not import data/ or models/ internals. Nothing in the core may import api/.
- Keep route handlers slim; put artifact loading, candidate generation, and ranking in a thin service layer (code-structure skill).
- Type hints everywhere.
When done, commit api/ and report the endpoint contracts (paths, params, response schemas) so the frontend chat can build against them.
```

## Why this chat exists

Serving concerns (HTTP, validation, artifact loading, caching) are entirely orthogonal to modeling and evaluation, and pulling them into a core chat would invert the dependency direction and tempt scoring logic to drift into the web layer. A separate leaf chat keeps the API replaceable and guarantees the research core stays runnable without any web stack.
