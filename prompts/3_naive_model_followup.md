Plan approved. Implement models/naive.py.

Strict TDD rules:
1. Write all tests in tests/test_naive_model.py before writing any implementation.
2. Run pytest — confirm every test fails before implementation begins.
3. Implement to make tests pass.
4. Never modify a test to satisfy an implementation.
5. Run pytest --tb=short after every change.

Use a synthetic fixture from tests/conftest.py (15 users, ~60 interactions). Do not require real data.

Required tests:
- test_protocol_conformance: NaiveReciprocal satisfies ReciprocityModel Protocol (runtime check).
- test_asymmetry: after fit(), s(A→B) != s(B→A) for at least one pair in the fixture (confirms the two sides are independent).
- test_aggregation_dot_product: r(A,B) equals s(A→B) * s(B→A) when dot_product aggregation is selected.
- test_aggregation_weighted_sum: r(A,B) equals alpha * s(A→B) + (1-alpha) * s(B→A) when weighted_sum is selected.
- test_artifact_round_trip: save() then load() produces identical predict() scores for all fixture pairs (tolerance 1e-6).
- test_artifact_schema: the saved JSON contains exactly the keys required by ModelArtifact (no extras, no missing).
- test_artifact_model_name: artifact model_name field equals "naive".
- test_unknown_user: calling predict() with an unseen user_id raises the documented exception type.

After tests pass, run:
  grep -r "from eval" models/
  grep -r "from data" models/
Paste results. models/ must not import from eval/ or from data/ (the module, not the type).