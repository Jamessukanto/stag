Plan approved. Implement experiments/.

Strict TDD rules:
1. Write all tests in tests/test_experiments.py before any implementation.
2. Run pytest — confirm every test fails.
3. Implement to make tests pass.
4. Never modify a test to satisfy an implementation.
5. Run pytest --tb=short after every change.

Use Protocol-conformant stubs for DataLoader, ReciprocityModel, and Evaluator in tests. Do not run real training in unit tests.

Required unit tests:
- test_four_configs_generated: the experiment matrix produces exactly 4 configurations.
- test_artifact_serialized_before_eval: mock the Evaluator and assert it is called with a Path argument, never with a model object. (Use unittest.mock.call_args inspection.)
- test_idempotency_skips_training: if the artifact path already exists, the model's fit() method is not called. (Mock fit() and check call count is zero.)
- test_comparison_table_schema: the output CSV has one row per configuration and the expected column names.
- test_seed_propagates: the same seed is passed to both the DataLoader and each ReciprocityModel. (Assert via mock call inspection.)
- test_partial_failure_behaviour: if one model configuration raises during fit(), assert the documented behaviour (abort or continue-and-log) occurs.

Integration test (marked @pytest.mark.slow, excluded from default run):
- test_full_pipeline_integration: runs all 4 real configurations end-to-end on the Libimseti dataset. Asserts artifact files exist, comparison.csv is written, and all EvaluationResult fields are non-None. Run manually with: pytest -m slow

Add to pytest.ini:
  markers =
    slow: marks tests as requiring real data and significant compute