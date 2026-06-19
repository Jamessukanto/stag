# Project Brief: Modular Reciprocal Recommendation Engine

## 1. Problem Framing

### 1.1 Challenge

Matching platforms (e.g. dating and social discovery apps) optimize for engagement - swipes, likes, messages. But engagement is one-sided. A profile recommendation creates value only when both users mutually prefer each other. This mismatch produces profile cycling and interactions that go nowhere.

The challenge is not ranking relevant profiles, but identifying pairs with a high probability of **mutual** preference.

### 1.2 Why this is intersting to me

Most recommender systems (e.g. Netflix) are user–item problems where only one side expresses preference. I initially framed it as “Netflix for dating”, but it breaks down as social matching is inherently a negotiation - there's no clear producer–consumer split. This creates design tensions absent in one-sided systems:
1. Attraction is often asymmetric, with one side typically having more influence.
2. Preferences are malleable: users trade convenience (e.g. distance, effort) for perceived quality.
3. Standard metrics (NDCG, Precision@K) mislead when the goal is mutual preference, not unilateral relevance

---

## 2. Solution Design

### 2.1 How I Use AI

#### 2.1.1 Design decisions, literature review with Claude and Manus

- Summarize research papers
- Adversarial framing surfaces  constraints a summary misses. Argue both sides of a specific tradeoff:
    - Example prompt "Given sparse interaction data, is naive aggregation or REF preferable — and what's the strongest case against your own recommendation?" 
- Challenge the problem framing itself. 
    - Example prompt: "When is one-sided optimisation actually better than reciprocal?" 

#### 2.1.2 Code implementation with Cursor Agent

A meta prompt (`prompts/0_initial_meta_prompt.md`) plans each chat session; its outputs in `prompts/` are pasted into separate Plan Mode chats (`*_init.md` → approve → `*_followup.md`). One session per module; modules communicate only through committed artifacts and `src/core/` contracts. 

```
Chat session execution order:

Architecture → [Data, MF, NeuMF, Evaluation] (parallel) → Experiments → API → Frontend
```

| Mechanism | Details |
|------------|---------|
| Project rules | `pytest`; type hints; never modify tests to fit implementations |
| TDD | Write tests → verify failures → implement → pass |
| Code-structure skill | Module boundaries via the [code-structure skill](https://github.com/michaelshimeles/skills/blob/main/code-structure/SKILL.md) |


### 2.2 Scope

A modular reciprocal recommendation system and configurable evaluation pipeline built on the [Reciprocal Embedding Framework (REF; Ramanathan et al., AAAI 2021)](https://cdn.aaai.org/ojs/17807/17807-13-21301-1-2-20210518.pdf). REF learns each user's two directional preference embeddings and fuses them into one mutual-preference score via an aggregation function. The preference model that produces directional scores is a plug-in choice.

- **Matrix Factorization** - linear, dot-product interaction (REF, Eq. 1–2)
- **Neural Network** — [NeuMF (He et al., NCF, 2017)](https://arxiv.org/pdf/1708.05031) - non-linear interaction 

Both consume the same binarized Libimseti interactions and the same aggregation function, so the comparison isolates the effect of the preference model.

**Excluded**: chemistry, logistics, transient intent — difficult to observe or evaluate reliably.

### 2.3 Dataset

Libimseti (Czech dating site; ~135k users, millions of directional ratings on a **1–10 scale**). One of the few public datasets with explicit bidirectional interaction data suited to reciprocal recommendation research.

**Binarization**: rating ≥ 7 → positive (like); rating < 7 → **explicit** negative (dislike). We train on full binary supervision (see 2.4), and use negative sampling for ablation.

**Split**: Libimseti carries no timestamps so temporal cutoffs are not possible. Random per-user holdout into train/validation/test. Stratified per user to keep every user represented across splits.

### 2.4 Models

Both models sit inside the REF framework. Each user u carries two embeddings:

- p_u — source: how user u selects others
- q_u — target: how others select u

A preference model produces a **directional** score s(u→v) from these embeddings. REF then fuses the two directions into a single reciprocal score:

r(A,B) = f(s(A→B), s(B→A))

where f is an aggregation function (e.g. product, harmonic mean, or weighted mean, per the REF paper). The two models below differ only in how s(u→v) is computed.

**Training signal**: full binary supervision from binarized Libimseti — positives (rating ≥ 7) and explicit negatives (rating < 7). Negative downsampling (default on) trims the negative set per positive for training efficiency and ablation; it is not the source of negative signal.

<details>
<summary><strong>2.4.1 Matrix Factorization (REF, Eq. 1–2)</strong></summary>

Directional score is the inner product of the source and target embeddings:

- s(u→v) = p_u^T · q_v

Trained as weighted matrix factorization, minimizing a regularized squared loss with L2 regularization over the binary labels, via SGD/Adam. Linear and scalable, but the dot product is a fixed interaction function.

</details>

<details>
<summary><strong>2.4.2 Neural Network — NeuMF (He et al., NCF, 2017)</strong></summary>

Replaces the dot product with a learned, non-linear interaction function. NeuMF fuses two branches over separate embeddings:

- **GMF**: element-wise product p_u^G ⊙ q_v^G (a generalized matrix factorization)
- **MLP**: concatenate [p_u^M ; q_v^M], then stacked ReLU layers in a tower structure

Their last hidden layers are concatenated and projected through a sigmoid output to produce s(u→v), trained with binary cross-entropy (log loss). GMF and MLP can be pretrained separately and used to initialize NeuMF. This captures non-linear interactions the dot product cannot, and the input layer can absorb side features for cold-start users.

</details>


### 2.5 Evaluation Metrics

Candidates are ranked by the reciprocal score r(A,B), then scored with the metrics from the source papers:

- **Recall@K** (REF): fraction of held-out positive targets recovered in a user's top-K.
- **HR@K** and **NDCG@K** (NCF): hit rate and rank-discounted gain over the top-K, using leave-one-out evaluation with sampled negatives.

The standard concern (1.2) is that these one-sided metrics reward unilateral relevance. We address it at the ground-truth level rather than by inventing new metrics: the relevant set is **mutual matches** (both users rated each other ≥ 7), so a hit counts only when preference is reciprocated. Ranking by r(A,B) against mutual-match ground truth keeps familiar, comparable metrics while still measuring mutual preference.

All metrics are computed across both models (MF, NN) and aggregation functions.

---

## 3. Prototype

**Status:** Architecture session complete (`56324f2`). Frozen contracts in `src/core/`; shared synthetic fixture in `conftest.py`.

```
src/core/     types, Protocols, ModelArtifact, scoring interpreter, config
data/         (Data chat)
models/mf/    (MF chat)
models/neumf/ (NeuMF chat)
eval/         (Evaluation chat — never imports models/)
artifacts/    serialized ModelArtifact JSON
```

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest && mypy && ruff check .
```

**Key invariant:** `eval/` scores models from on-disk `ModelArtifact` files via `core.scoring.reconstruct_scorer` — no model imports, no branching on `model_name`.

---

## 4. Impact Analysis

### 4.1 On using AI


| Dimension | Without AI | With AI | Tradeoff / Limitation |
|---|---|---|---|
| Literature review | Slow manual reading | Faster targeted exploration | Summaries still required verification |
| Implementation scope | One model, minimal pipeline | Multiple models and sampling strategies | More complexity to validate |
| Debugging | Manual iteration | Test-driven iteration with Cursor | Passing tests ≠ correctness |

What I'd need to learn next:

- Graph neural networks, to capture higher-order interaction structure.
- Cold-start methods, combining profile features with interaction data.














