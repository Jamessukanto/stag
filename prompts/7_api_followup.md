Plan approved. Implement api/.

Strict TDD rules:
1. Write tests first in tests/test_api.py using FastAPI TestClient.
2. Run pytest — confirm tests fail before implementation.
3. Implement to make tests pass.
4. Never modify tests to satisfy implementations.
5. Run pytest --tb=short after every change.

Use synthetic ModelArtifact fixtures — do not require trained models.

Required tests:
- test_health_returns_200.
- test_list_artifacts_empty_dir: returns empty list when artifact directory has no valid artifacts.
- test_list_artifacts_populated: returns correct identifiers for fixture artifacts.
- test_recommendations_valid_user: returns list of user IDs of length K.
- test_recommendations_unknown_user: returns 404 with structured error body.
- test_recommendations_unknown_artifact: returns 404.
- test_evaluation_returns_result: returns EvaluationResult fields.
- test_no_file_written: after any sequence of GET requests, assert no file in the artifact directory has been modified (compare mtime before and after).

Before closing:
  grep -r "from models" api/
  grep -r "from eval" api/
Both must return empty.