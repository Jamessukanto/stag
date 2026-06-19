# Follow-up implementation prompt — Evaluation

Paste this after you approve the plan, to move from planning to implementation under TDD.

```
The plan is approved. Implement eval/ strictly test-first. Do not write implementation code before its test exists and fails.

Workflow, repeat per capability (each aggregation function, ranking, mutual-match filtering, Recall@K, HR@K, NDCG@K, leave-one-out sampling):
1. Write the pytest tests first, using the shared synthetic fixture from conftest.py plus tiny hand-built rankings with known correct metric values. Cover: product/harmonic/weighted aggregation on hand values, Recall@K / HR@K / NDCG@K against rankings whose answers you computed by hand, mutual-match ground truth excludes one-sided pairs, deterministic negative sampling under a seed, and that the evaluator produces identical results for a synthetic "mf" artifact and a synthetic "neumf" artifact carrying the same embeddings.
2. Run pytest and show me the tests failing for the right reason before writing any implementation.
3. Implement the minimum code to pass, conforming to the Aggregator and Evaluator Protocols and reading models only as ModelArtifacts from disk.
4. Run pytest again and show all tests passing.

Hard rules:
- Never modify a test to make a failing implementation pass. If a test is genuinely wrong, fix it deliberately and re-verify it fails for the right reason first.
- Do not redefine or edit src/core/. NEVER import models/ (mf or neumf) or data/ internals; if scoring seems to require it, stop and report the contract gap.
- Keep the Evaluator as the orchestration surface; keep aggregation functions, the ranker, leave-one-out sampling, and each metric as separate service functions (code-structure skill).
- Type hints everywhere; keep mypy and ruff clean; seed all sampling.

When done, commit eval/ and report the Evaluator.evaluate signature and the EvaluationResult fields, so the experiments chat can drive it across models and aggregations.
```
