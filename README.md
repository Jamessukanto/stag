# Reciprocal Recommendation Engine

**Status:** `core/`, `data/`, `models/mf/`, `models/neumf/`, `eval/`, and `experiments/` are implemented with tests.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest && python -m importlinter.cli && mypy && ruff check .
```

**Prototype dataset:** `data/ratings.local.dat` is not checked in. It is a **closed user subgraph** of [Libimseti](https://networkrepository.com/libimseti.php): sample mutual-like pairs to pick a user cohort, then keep **every** rating (likes, dislikes, one-sided) where **both** endpoints are in that set. Capped at **120,000** directed rows by default. Includes explicit negatives so training has contrast; still smaller and faster than the full benchmark.

Generate once from a full download:

```bash
mkdir -p data
curl -L -o /tmp/libimseti.zip https://nrvis.com/download/data/misc/libimseti.zip
unzip -p /tmp/libimseti.zip libimseti.edges | grep -v '^%' > data/ratings.dat

python scripts/generate_ratings_local.py   # → data/ratings.local.dat
```

Optional flags: `--target-pairs` (seed cohort size), `--max-directed-edges` (default 120000; use `0` for no cap). Do not use `head` on the raw file — early lines are one-directional and produce zero reciprocal eval signal.

**Run the pipeline** (load → train MF + NeuMF → evaluate → comparison table). All settings live in `experiments/configs/prototype.json`:

```bash
python -m experiments.cli --config experiments/configs/prototype.json
```

Writes artifacts to `artifacts/prototype/` and results to `experiments/results/prototype/` (`comparison.json`, `comparison.csv`, `resolved_config.json`).

### Prototype vs benchmark

**Prototype numbers are not comparable to benchmark numbers.** They answer different questions:

| | Prototype (`prototype.json`) | Benchmark (`benchmark.json`) |
|--|------------------------------|------------------------------|
| **Purpose** | Fast end-to-end smoke test | Meaningful model comparison on full Libimseti |
| **Data** | `ratings.local.dat` (~120k-row subgraph) | `ratings.dat` (~17M ratings) |
| **Training** | dim 4, 5 epochs | dim 32, 20 epochs |
| **NCF pool** | 7 distractors (8 candidates) | 100 distractors (101 candidates) |
| **Metrics** | HR@5 can look high; Recall@K stays tiny | Recall@K and HR@K reflect full-scale ranking |

Do not cite prototype HR/Recall as Libimseti benchmark results. Use `experiments/configs/benchmark.json` on `data/ratings.dat` when you need scores for a write-up or comparison.

<details>
<summary><strong>Matrix Factorization (MF)</strong></summary>

Public entry point: `MatrixFactorizationModel(config, *, sampling_strategy="random", l2_weight=0.01)` — conforms to `core.PreferenceModel`.

| Method | Behavior |
|--------|----------|
| `fit(interactions)` | Trains on `split=="train"` rows only; builds `UserIndex` from all users in the input list |
| `directional_score(user_u, target_v)` | `p_u^T q_v` via learned embeddings |
| `save(path)` / `load(path)` | Writes/reads a `ModelArtifact` with `model_name="mf"` (no `score_program`; eval uses default dot product) |

**Hyperparameters:** shared run settings (`embedding_dim`, `learning_rate`, `epochs`, `negative_downsample_ratio`, `random_seed`) come from `core.Config`. MF-specific settings ride on the model constructor and are persisted in `ModelArtifact.hyperparameters` — notably `l2_weight` (default `0.01`) and `optimizer: "adam"`. `core.Config` intentionally excludes model-specific fields so the shared contract stays stable when swapping MF for NeuMF.

Train-negative downsampling is **not** applied in `fit()`; the caller passes the interaction list from `LibimsetiDataLoader(downsample=True).load()`. The model records `negative_downsample_ratio` in the artifact for provenance only.

Mechanics live in `models/mf/services/` (embeddings, loss, training, artifact). `models/mf/` imports only `core/`; it never imports `data/` or `eval/`.

</details>

<details>
<summary><strong>NeuMF</strong></summary>

Public entry point: `NeuMFModel(config, *, sampling_strategy="random")` — conforms to `core.PreferenceModel`.

| Method | Behavior |
|--------|----------|
| `fit(interactions)` | Trains on `split=="train"`; uses val for early stopping |
| `directional_score(user_u, target_v)` | GMF + MLP branches → sigmoid s(u→v) |
| `save(path)` / `load(path)` | Writes/reads a `ModelArtifact` with `model_name="neumf"` and a `score_program` in `extra` for eval |

NeuMF keeps separate GMF and MLP embedding tables per user (source/target roles), but like MF it produces **directional scores only**. Reciprocal fusion and metrics are unchanged — handled entirely by `eval/`.

Mechanics live in `models/neumf/services/` (network, loss, training, artifact). `models/neumf/` imports only `core/`; it never imports `data/`, `eval/`, or `models/mf/`.

</details>

<details>
<summary><strong>Experiments</strong></summary>

Public entry point: `python -m experiments.cli --config experiments/configs/<prototype|benchmark>.json` or `from experiments import run_pipeline, ExperimentRunConfig`.

| Step | What happens |
|------|----------------|
| Load | `LibimsetiDataLoader(config, downsample=True).load()` |
| Train | MF → `artifacts/mf.json`, NeuMF → `artifacts/neumf.json` |
| Eval | `EvaluationDataset.from_interactions(..., split="test")` + `ReciprocalEvaluator` sweep |
| Output | `experiments/results/comparison.json` (model × aggregation × k → Recall@K, HR@K, NDCG@K) |

`ExperimentRunConfig` wraps `core.Config` with orchestration-only fields: `eval_split`, `sampling_strategy`, `aggregations`, `ncf_distractors`, `weighted_alpha`, `results_dir`. Config JSON accepts either a nested `"base"` object or flat fields (core settings at the top level alongside experiments fields). One `random_seed` drives splitting, training, and eval distractor sampling.

</details>

<details>
<summary><strong>Evaluation</strong></summary>

Public entry point: `ReciprocalEvaluator(config, *, ncf_distractors=100, weighted_alpha=0.5)`.

| Method | Behavior |
|--------|----------|
| `evaluate(artifact_path, ground_truth, aggregation, k)` | Load artifact → rank by r(A,B) → return `EvaluationResult` |

Aggregators: `"product"`, `"harmonic"`, `"weighted"`. Reads `ModelArtifact` JSON only — never imports `models/`. Ground truth arrives as `EvaluationDataset` (built by the caller from the eval split); mutual-match partners come from `core.ground_truth.mutual_match_partners`.

Mechanics live in `eval/services/` (aggregators, ranking, candidate sampling, metrics). `eval/` imports only `core/`.

</details>

<details>
<summary><strong>Architecture</strong></summary>

### Layout

```
core/         shared types, Protocols, ModelArtifact, scoring interpreter, config
data/         Libimseti loading, binarization, splits, negative sampling
models/mf/    matrix-factorization preference model (plug-in)
models/neumf/ NeuMF preference model (plug-in)
eval/         aggregation, ranking, metrics — never imports models/
experiments/  orchestrates load → train → evaluate
artifacts/    serialized ModelArtifact JSON (+ optional EvaluationDataset JSON)
```

Feature modules (`data/`, `models/`, `eval/`) sit beside `core/` at the repo root. Only `core/` is imported by everyone; siblings never import each other's internals.

### How eval scores models (without importing them)

`eval/` reads **`ModelArtifact`** JSON only — never `models/`. Each file holds learned weights plus an optional `score_program` (step list for non-linear models). `core.scoring.reconstruct_scorer` runs it; same path for every model.

| Model | Directional score s(u→v) | Reciprocal score r(A,B) |
|-------|--------------------------|-------------------------|
| MF | Dot product `p_u · q_v` (default when no program) | `eval/` only: f(s(A→B), s(B→A)) |
| NeuMF | `score_program` in `extra` (lookup → concat → dense → sigmoid) | same |

Golden tests (`verify_scorer_matches_directional`) keep the interpreter aligned with each model's `directional_score`. Aggregation is identical for both artifacts.

### Data flow

```
ratings.local.dat → data/ → ProcessedInteraction[]
                         ↓ train split
                    models/ → ModelArtifact JSON
                         ↓
              experiments/ pairs artifact + EvaluationDataset (test split)
                         ↓
                    eval/ → EvaluationResult
```

`eval/` never calls `data/` — `experiments/` (or any caller) filters the loaded interactions and passes ground truth in.

</details>
