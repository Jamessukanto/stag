Plan approved. Implement models/ref.py.

Strict TDD rules:
1. Write all tests in tests/test_ref_model.py before writing any implementation.
2. Run pytest — confirm every test fails before implementation begins.
3. Implement to make tests pass.
4. Never modify a test to satisfy an implementation.
5. Run pytest --tb=short after every change.

Use the same synthetic fixture from tests/conftest.py as test_naive_model.py.

Required tests:
- test_protocol_conformance: REFModel satisfies ReciprocityModel Protocol.
- test_embeddings_differ_from_naive: train both NaiveReciprocal and REFModel on the same fixture; assert their p_u matrices are not identical (different objectives must produce different embeddings).
- test_reciprocal_score_symmetry_property: verify whether r(A,B) == r(B,A) for REF — this is a model property to document, not a correctness requirement per se.
- test_artifact_round_trip: save() then load() produces identical predict() scores (tolerance 1e-6).
- test_artifact_schema_identical_to_naive: the set of top-level keys in the REF artifact JSON equals the set in the naive artifact JSON exactly.
- test_artifact_model_name: artifact model_name equals "ref".
- test_no_naive_import: the file models/ref.py contains no import from models.naive (assert via ast.parse or grep).
- test_unknown_user: same behaviour as documented for naive — raise the same exception type.

After tests pass, run:
  grep -r "from eval" models/
  grep -r "from models.naive" models/ref.py
Paste results. Both must return empty.