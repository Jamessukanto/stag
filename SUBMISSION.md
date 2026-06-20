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

A meta prompt (`prompts/0_initial_meta_prompt.md`) plans each chat session; its outputs in `prompts/` are pasted into separate Plan Mode chats (`*_init.md` → review plan → **Build**). Each prompt carries its own Build requirements (TDD loop + boundaries), and project-wide rules live in `.cursor/rules/`. One session per module; modules communicate only through committed artifacts and `core/` contracts. 

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

A modular reciprocal recommendation system and configurable evaluation pipeline built on the . We compare 2 models:

- **Matrix Factorization** — [Reciprocal Embedding Framework (REF; Ramanathan et al., AAAI 2021)](https://cdn.aaai.org/ojs/17807/17807-13-21301-1-2-20210518.pdf) linear, dot-product directional score (REF, Eq. 1–2)
- **NeuMF** — [NeuMF (He et al., NCF, 2017)](https://arxiv.org/pdf/1708.05031) non-linear directional score (GMF + MLP)


**Excluded**: chemistry, logistics, transient intent — difficult to observe or evaluate reliably.

### 2.3 Dataset

Libimseti (Czech dating site; ~135k users, millions of directional ratings on a **1–10 scale**). One of the few public datasets with explicit bidirectional interaction data suited to reciprocal recommendation research.

**Binarization**: rating ≥ 7 → positive (like); rating < 7 → **explicit** negative (dislike). We train on full binary supervision (see 2.4), and use negative sampling for ablation.

**Split**: Libimseti carries no timestamps so temporal cutoffs are not possible. Random per-user holdout into train/validation/test. Stratified per user to keep every user represented across splits.

### 2.4 Models

Both models sit inside the REF framework. Each user u carries two embeddings:

- p_u — source: how user u selects others
- q_u — target: how others select u

Each preference model produces a **directional** score s(u→v) using u's source-side and v's target-side representations (e.g. s(u→v) = p_u^T q_v for MF). The reciprocal score is assembled only at eval/ranking time:

r(A,B) = f(s(A→B), s(B→A))

where **f** is an aggregation function (product, harmonic mean, or weighted mean). 

**Training signal (directional, not mutual-match):** Loss compares s(u→v) to that single-direction label — models never train on r(A,B) or mutual-match labels. Train-negative downsampling (default on) thins explicit dislikes in the **train** split only.

**Eval ground truth (mutual-match):** a pair (A,B) is relevant only when **both** A→B and B→A are likes in the held-out split. This is derived at eval time via `core.ground_truth.mutual_match_partners`, not stored as a separate training label.

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

Their last hidden layers are concatenated and projected through a sigmoid output to produce s(u→v), trained with binary cross-entropy (log loss). Like MF, NeuMF uses **u’s source-side and v’s target-side** embeddings (not p_u · p_v). Aggregation into r(A,B) lives in `eval/`, not here.

</details>


### 2.5 Evaluation Metrics

`eval/` loads a `ModelArtifact`, reconstructs s(u→v), applies **f** to get r(A,B), ranks candidates, and reports metrics. **REF** and **NCF** here name **metric protocols**, not which model runs them — MF and NeuMF both get all three metrics.

| Metric | Protocol | Candidate pool | What it measures |
|--------|----------|----------------|------------------|
| **Recall@K** | REF | All other users in the artifact | Fraction of a user's mutual-match partners in top-K |
| **HR@K** | NCF | 1 held-out partner + 100 uninteracted distractors | Hit rate (binary) per leave-one-out trial |
| **NDCG@K** | NCF | Same as HR@K | Rank-discounted gain per trial (one relevant item) |

HR@K and NDCG@K hold out **one** mutual-match partner per trial and average over all trials. Recall@K ranks **all** of a user's mutual-match partners together against the full user pool.

---

## 3. Prototype

The prototype implements the design above: `core/`, `data/`, `models/mf/`, `models/neumf/`, `eval/`, and `experiments/` with tests. It trains MF and NeuMF on a local Libimseti subgraph and runs the full evaluation pipeline.

Setup, dataset generation, CLI usage, and module reference: **[README.md](README.md)**.

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
