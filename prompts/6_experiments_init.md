# Chat name

Experiments

## Purpose

The reproducible orchestration layer that wires the finished modules together end to end: load data, train MF and NeuMF, evaluate both across every aggregation function, and emit the comparison tables. It owns no algorithms — only the run configuration, sequencing, and result collection.

## Dependencies

- Architecture, Data, MF, NeuMF, and Evaluation, all committed and passing their full test suites with stable public interfaces.
- Imports modules only through their committed public surfaces (DataLoader, the two PreferenceModel implementations, the Evaluator) and the shared contracts. It does not reach into any module's internals.

## Initial Plan Mode prompt

```
You are planning the experiments/ module of the reciprocal-rec project. This is a Plan Mode session: produce a plan only, no implementation code. After I review the plan and click Build, implement it under the Build requirements at the end of this prompt.

All upstream modules are complete and frozen: data/, models/mf/, models/neumf/, eval/, and core/ contracts. You orchestrate them; you do not reimplement any of their logic and you import only their documented public surfaces.

Scope, strictly limited to experiments/:
1. A single reproducible pipeline that, from one config:
   - loads and splits the data via the DataLoader,
   - trains the MF model and the NeuMF model on the train split, writing each as a ModelArtifact to artifacts/,
   - builds a core.ground_truth.EvaluationDataset for the eval split (filtering the loaded interactions) — this is how eval gets ground truth without importing data/,
   - evaluates each artifact via the Evaluator across all aggregation functions ("product", "harmonic", "weighted") and all config.k_values,
   - collects EvaluationResult records into a comparison table (model x aggregation x k -> Recall@K, HR@K, NDCG@K).
2. Reproducibility: one config file / CLI entry point; a single random_seed controls data splitting, training, and evaluation sampling. Re-running must reproduce the same artifacts and metrics. Persist the resolved config alongside results.
3. Output: write results as a machine-readable file (CSV/JSON) in artifacts/ or experiments/results/, and render a readable comparison table. No plotting library is required; a printed/markdown table is fine.

Constraints:
- Import only public interfaces from data/, models/, eval/, and shared types from core/. Do not duplicate training, scoring, aggregation, or metric logic that already lives in a module. If something you need is missing from a public interface, note it as an upstream gap rather than reaching into internals or copying code.
- Apply the code-structure skill: experiments/ is pure orchestration (the "what to run, in what order, with what config"); the mechanics (fit, score, evaluate) stay in their modules. Keep the run as composable steps (load -> train each model -> build ground truth -> evaluate each across aggregations -> collect), not one monolithic function.
- Type hints throughout. The pipeline must be deterministic under the seed.

Produce: the file list with one-line descriptions, the CLI/config entry point design, the exact shape of the comparison table, the test cases (the pipeline runs end to end on the synthetic fixture and produces a fully populated table, results are reproducible under a fixed seed, every model x aggregation x k cell is filled), and any upstream interface gaps you discover.

Build requirements (apply when implementing after approval):
Implement experiments/ strictly test-first. Do not write implementation code before its test exists and fails.
Workflow, repeat per capability (config loading, the train step per model, ground-truth construction, the evaluate sweep, result collection, end-to-end pipeline):
1. Write the pytest tests first, using the shared synthetic fixture and small config so the full pipeline runs fast. Cover: the pipeline runs end to end and produces a comparison table with every model x aggregation x k cell populated, results are reproducible (byte-for-byte or within float tolerance) under a fixed seed, and the resolved config is persisted with the results.
2. Run pytest and show the tests failing for the right reason before writing any implementation.
3. Implement the minimum code to pass, importing only public interfaces from data/, models/, and eval/.
4. Run pytest, then `python -m importlinter.cli`, mypy, and ruff; all must pass.
Hard rules:
- Never modify a test to make a failing implementation pass; if a test is genuinely wrong, fix it deliberately and re-verify it fails for the right reason first.
- Do not redefine or edit core/. Do not import any module's internals (data.services, models.mf, models.neumf) and do not copy training, scoring, aggregation, or metric logic; if a public interface is missing something, stop and report it as an upstream gap.
- Keep the run as composable orchestration steps (code-structure skill), not a monolith.
- Type hints everywhere; one seed drives the whole run.
When done, commit experiments/ and report the entry-point command and the results file location/schema, so the API chat can serve from trained artifacts.
```

## Why this chat exists

Orchestration is where module boundaries are most tempting to violate — it is easy to "just import the internals" or copy a training loop to make a run work. Isolating it after the modules stabilize keeps it as a thin, reproducible driver and surfaces any missing public interface as an explicit upstream gap instead of a silent coupling. It also concentrates all reproducibility concerns (seeding, config persistence) in one place.
