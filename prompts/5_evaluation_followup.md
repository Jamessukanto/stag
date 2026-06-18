Plan approved. Implement eval/.

Strict TDD rules:
1. Write all tests in tests/test_eval.py before any implementation.
2. Run pytest — confirm every test fails before writing any implementation.
3. Implement to make tests pass.
4. Never modify a test to satisfy an implementation.
5. Run pytest --tb=short after every file change.

Do not use trained models in tests. Construct ModelArtifact objects directly using synthetic embeddings.

Required tests:
- test_rp_at_k_perfect: construct embeddings where users A,B,C all mutually prefer each other in top-3. Assert RP@3 == 1.0.
- test_rp_at_k_zero: construct embeddings where no user appears in any other user's top-K. Assert RP@K == 0.0.
- test_rp_at_k_partial: construct a case where exactly half of top-K recommendations are mutual. Assert RP@K == 0.5 (within tolerance).
- test_mutual_hit_rate_hit: at least one mutual pair exists. Assert MHR > 0.0.
- test_mutual_hit_rate_miss: no mutual pairs. Assert MHR == 0.0.
- test_wrong_split_raises: artifact with trained_on_split != "train" must raise a descriptive ValueError before any metric is computed.
- test_batch_evaluation_uniform: given two artifacts (one "naive", one "ref"), batch evaluation returns two EvaluationResult objects with the correct model_name fields.
- test_evaluation_result_schema: returned EvaluationResult has all required fields from src/core/types.py.

After all tests pass, run:
  grep -r "from models" eval/
  grep -r "import models" eval/
  grep -r "from data" eval/
Paste all results. All must return empty output. This is a hard requirement.