# Chat name

Evaluation

## Purpose

Owns the reciprocal side of the system and all scoring of model quality: the aggregation functions f that fuse directional scores into r(A,B), the mutual-match ground truth, and the ranking metrics (Recall@K, HR@K, NDCG@K). It consumes model outputs only as serialized artifacts and never imports model code.

## Dependencies

- Architecture (committed): imports `ProcessedInteraction`, `UserIndex`, `ModelArtifact`, `EvaluationDataset`, the `Aggregator` and `Evaluator` Protocols, `EvaluationResult`, and config from `core/`, plus the shared `conftest.py` fixture.
- May run in parallel with Data, MF, and NeuMF. Hard invariant: forbidden from importing `models/` (mf or neumf) or `data/` internals. It reads `ModelArtifact`s from disk and receives held-out interactions as an `EvaluationDataset`.

## Initial Plan Mode prompt

```
You are planning the eval/ module of the reciprocal-rec project. This is a Plan Mode session: produce a plan only, no implementation code. After I review the plan and click Build, implement it under the Build requirements at the end of this prompt.

The shared contracts in core/ are frozen. Import and conform to them; do not modify them. You implement the Aggregator Protocol (aggregate(s_ab, s_ba) -> float) and the Evaluator Protocol (evaluate(artifact_path, ground_truth, aggregation, k) -> EvaluationResult), returning EvaluationResult records.

HARD ARCHITECTURAL INVARIANT: eval/ must never import models/ (neither mf nor neumf) and must never import data/. You receive a model only as a ModelArtifact loaded from disk (source_embeddings, target_embeddings, user_index, model_name, and the generic extra field), and held-out ground truth only as a core.ground_truth.EvaluationDataset passed in by the caller. Reconstruct directional scores s(u->v) from the artifact alone via core.scoring.reconstruct_scorer (never branch on model_name). If you find you need to import a model or the data loader, stop and treat that as a contract gap to raise, not a reason to import.

Scope, strictly limited to eval/:
1. Aggregation functions f (the reciprocal fusion), each implementing the Aggregator Protocol:
   - product: s(A->B) * s(B->A)
   - harmonic mean of the two directional scores
   - weighted mean with a configurable weight
   Selectable by name ("product" | "harmonic" | "weighted"). The reciprocal score is r(A,B) = f(s(A->B), s(B->A)).
2. Ranking: for each evaluated user, score candidate targets by r(A,B) and rank them.
3. Mutual-match ground truth: derive relevance from the EvaluationDataset via core.ground_truth.mutual_match_partners — a held-out pair (A,B) is relevant only if BOTH rated each other >= 7 (label 1 both ways). A top-K hit counts only against this mutual-match set, so metrics measure mutual preference, not unilateral relevance.
4. Metrics over the ranking:
   - Recall@K (REF): fraction of a user's held-out mutual-match targets recovered in top-K.
   - HR@K and NDCG@K (NCF): leave-one-out evaluation with sampled negatives.
   Use config.k_values; report per the EvaluationResult schema (model_name, aggregation, k, recall_at_k, hr_at_k, ndcg_at_k, evaluated_at).
5. The evaluator must work identically for any artifact regardless of model_name ("mf" or "neumf"); it never branches on which model produced the artifact beyond using core.scoring.

Constraints:
- Import everything shared from core/. Do not redefine shared types.
- Do not import models/ or data/. Read artifacts from disk; receive the eval split as an EvaluationDataset argument.
- Apply the code-structure skill: the Evaluator is the orchestration surface (load artifact -> build candidates -> rank -> score metrics); factor reusable mechanics (each aggregation function, the ranker, leave-one-out sampling, each metric) into explicit-input service functions with structured returns.
- Type hints throughout. Metric computations and negative sampling must be deterministic under config.random_seed.

Produce: the file list with one-line descriptions, exactly how directional scores are reconstructed from a ModelArtifact without importing a model, the candidate-generation and leave-one-out protocol, the test cases (each aggregation function on hand values, Recall@K / HR@K / NDCG@K against tiny rankings with known answers, mutual-match filtering excludes one-sided pairs, identical behavior for an "mf" vs "neumf" artifact), and decisions to confirm (number of sampled negatives for HR/NDCG, weighted-mean default weight).

Build requirements (apply when implementing after approval):
Implement eval/ strictly test-first. Do not write implementation code before its test exists and fails.
Workflow, repeat per capability (each aggregation function, ranking, mutual-match filtering, Recall@K, HR@K, NDCG@K, leave-one-out sampling):
1. Write the pytest tests first, using the shared synthetic fixture from conftest.py plus tiny hand-built rankings with known correct metric values. Cover: product/harmonic/weighted aggregation on hand values, Recall@K / HR@K / NDCG@K against rankings whose answers you computed by hand, mutual-match ground truth excludes one-sided pairs, deterministic negative sampling under a seed, and that the evaluator produces identical results for a synthetic "mf" artifact and a synthetic "neumf" artifact carrying the same embeddings.
2. Run pytest and show the tests failing for the right reason before writing any implementation.
3. Implement the minimum code to pass, conforming to the Aggregator and Evaluator Protocols, reading models only as ModelArtifacts from disk and ground truth only as an EvaluationDataset.
4. Run pytest, then `python -m importlinter.cli`, mypy, and ruff; all must pass.
Hard rules:
- Never modify a test to make a failing implementation pass; if a test is genuinely wrong, fix it deliberately and re-verify it fails for the right reason first.
- Do not redefine or edit core/. NEVER import models/ (mf or neumf) or data/; if scoring seems to require it, stop and report the contract gap.
- Keep the Evaluator as the orchestration surface; keep aggregation functions, the ranker, leave-one-out sampling, and each metric as separate service functions (code-structure skill).
- Type hints everywhere; seed all sampling.
When done, commit eval/ and report the Evaluator.evaluate signature and the EvaluationResult fields, so the experiments chat can drive it across models and aggregations.
```

## Why this chat exists

The whole project hinges on the claim that one-sided metrics can still measure mutual preference when paired with mutual-match ground truth, and on the boundary that evaluation cannot peek into models. Both are easiest to get wrong and most damaging if wrong. A dedicated chat that is structurally barred from importing models is the strongest guarantee that the comparison is fair and that aggregation/metrics stay model-agnostic.
