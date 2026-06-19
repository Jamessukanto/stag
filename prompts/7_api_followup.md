# Follow-up implementation prompt — API

Paste this after you approve the plan, to move from planning to implementation under TDD.

```
The plan is approved. Implement api/ strictly test-first. Do not write implementation code before its test exists and fails.

Workflow, repeat per capability (artifact loading/caching, ranking service, each endpoint, validation):
1. Write the pytest tests first using the framework's test client and a small synthetic ModelArtifact written to a temp artifacts dir. Cover: /health, /recommend returns exactly K ranked target ids for a known user, the served ranking matches eval's r(A,B) on that artifact, /score (if included) matches eval, and validation paths (unknown user_id -> 404, bad k/aggregation -> 422).
2. Run pytest and show me the tests failing for the right reason before writing any implementation.
3. Implement the minimum code to pass, reusing eval/'s public aggregation + scoring surface and loading artifacts from disk.
4. Run pytest again and show all tests passing.

Hard rules:
- Never modify a test to make a failing implementation pass. If a test is genuinely wrong, fix it deliberately and re-verify it fails for the right reason first.
- Do not redefine or edit src/core/. Do not reimplement scoring/aggregation/metrics; reuse eval/. Do not import data/ or models/ internals. Nothing in the core may import api/.
- Keep route handlers slim; put artifact loading, candidate generation, and ranking in a thin service layer (code-structure skill).
- Type hints everywhere; keep mypy and ruff clean.

When done, commit api/ and report the endpoint contracts (paths, params, response schemas) so the frontend chat can build against them.
```
