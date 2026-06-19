# Chat name

Experiments

## Purpose

The reproducible orchestration layer that wires the finished modules together end to end: load data, train MF and NeuMF, evaluate both across every aggregation function, and emit the comparison tables. It owns no algorithms — only the run configuration, sequencing, and result collection.

## Dependencies

- Architecture, Data, MF, NeuMF, and Evaluation, all committed and passing their full test suites with stable public interfaces.
- Imports modules only through their committed public surfaces (DataLoader, the two PreferenceModel implementations, the Evaluator) and the shared contracts. It does not reach into any module's internals.

## Initial Plan Mode prompt

```
You are implementing the experiments/ module of the reciprocal-rec project. This is a Plan Mode session: produce a plan only, no implementation code.

All upstream modules are complete and frozen: data/, models/mf/, models/neumf/, eval/, and src/core/ contracts. You orchestrate them; you do not reimplement any of their logic and you import only their documented public surfaces.

Scope, strictly limited to experiments/:
1. A single reproducible pipeline that, from one config:
   - loads and splits the data via the DataLoader,
   - trains the MF model and the NeuMF model on the train split, writing each as a ModelArtifact to artifacts/,
   - evaluates each artifact via the Evaluator across all aggregation functions ("product", "harmonic", "weighted") and all config.k_values,
   - collects EvaluationResult records into a comparison table (model x aggregation x k -> Recall@K, HR@K, NDCG@K).
2. Reproducibility: one config file / CLI entry point; a single random_seed controls data splitting, training, and evaluation sampling. Re-running must reproduce the same artifacts and metrics. Persist the resolved config alongside results.
3. Output: write results as a machine-readable file (CSV/JSON) in artifacts/ or experiments/results/, and render a readable comparison table. No plotting library is required; a printed/markdown table is fine.

Constraints:
- Import only public interfaces from data/, models/, eval/, and shared types from src/core/. Do not duplicate training, scoring, aggregation, or metric logic that already lives in a module. If something you need is missing from a public interface, note it as an upstream gap rather than reaching into internals or copying code.
- Apply the code-structure skill: experiments/ is pure orchestration (the "what to run, in what order, with what config"); the mechanics (fit, score, evaluate) stay in their modules. Keep the run as composable steps (load -> train each model -> evaluate each across aggregations -> collect), not one monolithic function.
- Type hints throughout. The pipeline must be deterministic under the seed.

Produce: the file list with one-line descriptions, the CLI/config entry point design, the exact shape of the comparison table, the test cases (the pipeline runs end to end on the synthetic fixture and produces a fully populated table, results are reproducible under a fixed seed, every model x aggregation x k cell is filled), and any upstream interface gaps you discover.
```

## Why this chat exists

Orchestration is where module boundaries are most tempting to violate — it is easy to "just import the internals" or copy a training loop to make a run work. Isolating it after the modules stabilize keeps it as a thin, reproducible driver and surfaces any missing public interface as an explicit upstream gap instead of a silent coupling. It also concentrates all reproducibility concerns (seeding, config persistence) in one place.
