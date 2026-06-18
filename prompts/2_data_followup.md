Plan approved. Implement data/.

Strict TDD rules:
1. Write all tests in tests/test_data.py before writing any implementation.
2. Run pytest — confirm every new test fails before any implementation file exists.
3. Implement to make tests pass. Do not write untested methods.
4. Never modify a test to satisfy an implementation.
5. Run pytest --tb=short after every file change.

All tests must use a synthetic fixture — do not require the real Libimseti file:
- Fixture: 15 users, ~60 directional interactions with monotonically increasing fake timestamps.
- Define this fixture in tests/conftest.py so it is available across sessions.

Required tests:
- test_temporal_split_no_overlap: assert max(train timestamps) < min(val timestamps) < min(test timestamps).
- test_temporal_split_ratios: assert split sizes are approximately correct (within 5% of config ratios).
- test_negatives_not_in_positives: for any user, no returned negative appears in that user's positive interaction set.
- test_random_sampling_reproducible: same seed → same negatives.
- test_popularity_biased_distribution: over 1000 samples, high-frequency users appear significantly more often than low-frequency users (use a chi-square or ratio check, not exact counts).
- test_cold_start_behavior: assert documented behavior (raise/skip/flag) occurs for a user absent from train.
- test_dataloader_protocol_conformance: assert the implementation satisfies the DataLoader Protocol (use isinstance with runtime_checkable or inspect its __protocol_attrs__).
- test_types_match: assert every returned ProcessedInteraction is an instance of the type from src/core/types.py.

Before closing, run:
  grep -r "from models" data/
  grep -r "from eval" data/
Both must return empty output. Paste the results as confirmation.