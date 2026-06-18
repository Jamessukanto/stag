Purpose
Implements the orchestration layer in experiments/: runs all four configurations ({NaiveReciprocal, REF} × {random, popularity_biased}), persists artifacts, invokes the evaluator, and assembles comparison results. This is the only layer that integrates all three modules.

Dependencies
All previous chats: Architecture, Data, NaiveModel, REFModel, Evaluation.

You are implementing the experiments/ orchestration layer. This is a Plan Mode session. Produce a plan only — no code.

Module responsibility:
- Define the 4-configuration experiment matrix:
    {NaiveReciprocal, REFModel} × {random_sampling, popularity_biased_sampling}
- For each configuration:
    1. Load data using the DataLoader from data/.
    2. Prepare training interactions with the specified negative sampling strategy.
    3. Instantiate and fit the specified model.
    4. Serialize the artifact to artifacts/<model_name>/<sampling_strategy>/<timestamp>/artifact.json
    5. Call the Evaluator from eval/ with the artifact path. The Evaluator receives only the path, never an in-memory model object.
    6. Collect the EvaluationResult.
- Write a comparison table (CSV and JSON) to artifacts/results/comparison.csv and comparison.json.
- Accept a global random seed from config that is passed through to both negative sampling and model training.
- Be idempotent: if an artifact already exists for a given (model, sampling, hyperparameter hash), skip training and go straight to evaluation.

This is the one place where data/, models/, and eval/ are imported together. That is intentional and acceptable only here.

Critical re-enforcement: even in this orchestration layer, eval/ must receive only artifact file paths. Never pass in-memory model objects to any eval/ function. Serialize first. Always.

Constraints:
- Lives in experiments/. Do not create files outside experiments/ or tests/.
- Do not re-implement any logic from data/, models/, or eval/. Glue code only.
- Do not define new types; use src/core/types.py.
- The entry point should be runnable as: python -m experiments.run --config config.yaml
- Config file path is the only CLI argument. Everything else comes from the config file.

Plan must address:
1. The experiment configuration schema: how each of the 4 runs is parameterised as a data structure.
2. How artifact paths are named to be unique per (model, sampling, hyperparameter hash).
3. How idempotency is checked (what constitutes a "matching" prior run?).
4. The comparison table schema: what columns, one row per run.
5. How failures in one configuration are handled (abort all? log and continue?).

