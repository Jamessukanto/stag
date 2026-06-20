# Reciprocal Preference Pipeline

Offline train + eval pipeline for reciprocal preference scorer on [Libimseti dataset](https://networkrepository.com/libimseti.php). Models: Matrix Factorization, MF, from [REF (Ramanathan et al., AAAI 2021)](https://cdn.aaai.org/ojs/17807/17807-13-21301-1-2-20210518.pdf) and Neural Collaborative Filtering, NCF, [He et al., 2017](https://arxiv.org/pdf/1708.05031)

## Layout

```
core/           types, ModelArtifact, scoring interpreter, config
data/           load, binarize, split, downsample
models/mf/      matrix factorization (REF)
models/neumf/   NCF / NCF (He et al.)
eval/           aggregation, metrics, policy comparison — never imports models/
experiments/    CLI + orchestration
prompts/        Cursor Plan/Build prompts (assessment artifact)
```

**Boundaries:** modules communicate via `core/` contracts and on-disk artifacts only. `eval/` scores from `ModelArtifact` JSON via `core.scoring.reconstruct_scorer` — no PyTorch imports. Golden tests enforce save/load score parity.

## Quick start

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest && python -m importlinter.cli && ruff check .
```

Download [Libimseti dataset](https://nrvis.com/download/data/misc/libimseti.zip). Sample and store locally `data/ratings.local.dat` for a smoke test run, then:

```bash
python -m experiments.cli --config experiments/configs/prototype.json
python -m experiments.cli --config experiments/configs/prototype.json --policy-analysis
```

**Outputs:** `artifacts/prototype/{mf,neumf}.json`, `experiments/results/prototype/{comparison.json,comparison.csv,resolved_config.json}`; with `--policy-analysis`, also `policy_tradeoff.json`.

## Data

**Binarization:** rating ≥ 7 → like (`1`), else dislike (`0`). Per-user stratified train/val/test; train negatives downsampled when `downsample=True`.

**Prototype slice:** Preserves mutual likes, one-sided likes, and dislikes. Do not use `head` on the raw file — early rows lack reciprocal signal.


## Pipeline

Load → train MF + NCF → eval metric sweep → optional policy analysis.

| Step | What happens |
|------|----------------|
| Load | `LibimsetiDataLoader(config, downsample=True).load()` |
| Train | MF + NCF → `ModelArtifact` JSON |
| Eval | Test-split `EvaluationDataset`; Recall@K, HR@K, NDCG@K per model × aggregation |
| Policy | `--policy-analysis`: rank by `s(u→v)` vs `r(u,v)`; write `policy_tradeoff.json` |

Programmatic: `from experiments import run_pipeline, ExperimentRunConfig`.

Policy analysis (`--policy-analysis`): compare engagement vs mutual ranking on held-out mutual matches → `policy_tradeoff.json`. See [SUBMISSION.md §3.3](SUBMISSION.md).

## Models

Architecture follows [REF (Ramanathan et al., 2021)](https://cdn.aaai.org/ojs/17807/17807-13-21301-1-2-20210518.pdf). Each user *u* plays two roles on a directed edge *u→v*:

| Symbol | Meaning |
|--------|---------|
| **p_u** | Source embedding — *u* as rater |
| **q_v** | Target embedding — *v* as candidate |
| **s(u→v)** | Directional score — predicted strength of *u* liking *v* |
| **s(v→u)** | Reverse direction |
| **r(u,v)** | Reciprocal score — both directions fused (eval only) |

Both plug-in models implement **s(u→v)** only. **r(u,v) = f(s(u→v), s(v→u))** is computed in `eval/` ([`eval/services/aggregators.py`](eval/services/aggregators.py)):

| `f` | Formula |
|-----|---------|
| `product` | r = s(u→v) · s(v→u) |
| `harmonic` | r = 2·s(u→v)·s(v→u) / (s(u→v) + s(v→u)) |
| `weighted` | r = α·s(u→v) + (1−α)·s(v→u) |

Engagement ranking sorts by **s(u→v)**; mutual ranking sorts by **r(u,v)**.

### MF — linear (REF)

[`models/mf/`](models/mf/) — REF directional dot product:

```
s(u→v) = p_u · q_v     (inner product of source/target embeddings)
```

- **Train:** squared loss `(y − s)²` on binary labels, L2 on p_u and q_v, Adam
- **Score range:** unbounded ℝ
- **Artifact:** `source_embeddings` (p), `target_embeddings` (q)

REF defines reciprocal ranking via **f**; this repo keeps **f** in eval so MF is a pure directional plug-in.

### NCF — non-linear (He et al.)

[`models/neumf/`](models/neumf/) — [NCF](https://arxiv.org/pdf/1708.05031) GMF + MLP on user–user edges:

| Branch | Computation |
|--------|-------------|
| **GMF** | element-wise product p_u^G ⊙ q_v^G |
| **MLP** | concat [p_u^M ; q_v^M] → ReLU tower |
| **Fusion** | concat(GMF, MLP hidden) → linear → sigmoid → s(u→v) ∈ (0,1) |

- **Train:** binary cross-entropy; val-split early stopping
- **Artifact:** branch embeddings + MLP weights in `extra`; `score_program` replays forward pass in eval without PyTorch

### MF vs NCF

| | MF (REF) | NCF (NCF) |
|--|----------|-------------|
| **s(u→v)** | Linear dot product | GMF + MLP + sigmoid |
| Loss | Squared error | BCE |
| Score range | ℝ | (0, 1) |
| Reciprocal **r** | Fused in eval, not trained on matches | same |

Both implement `PreferenceModel` (`fit`, `directional_score`, `save`/`load`). Golden tests verify eval matches training scores after serialization.
