Your task is NOT to implement the project.

Your task is NOT to design the software architecture directly.

Your task is to design a Cursor prompting strategy: a suite of chat sessions and, for each one, a ready-to-paste initial Plan Mode prompt that, executed in order, builds the prototype described in the project brief below.

Think of Cursor as a team of specialized engineers. Each engineer owns exactly one chat session and one subsystem. They never share scratch context — they communicate only through committed artifacts on disk and stable public interfaces. The goal is to preserve architecture and prevent the repository from collapsing into monolithic AI-generated code.

# Assumptions

* I primarily use Python.
* I use Cursor Agent.
* Each new Cursor chat begins in Plan Mode.
* Implementation follows TDD.
* I prefer separate chats for separate subsystems.
* Long-term maintainability matters more than minimizing the number of chats.
* An API and a lightweight React frontend may be added later.
* I want clean, structurally enforced boundaries between modules.

# How to decompose the work

Do NOT use a fixed, pre-supplied list of sessions. Derive the set of chat sessions yourself by reading the brief, guided by these principles:

* The REF framework is the architectural core. Every user carries two directional embeddings (source `p_u`, target `q_v`). A preference model turns these into a directional score `s(u->v)`; an aggregation function `f` fuses both directions into a reciprocal score `r(A,B) = f(s(A->B), s(B->A))`. Treat the preference model as a plug-in behind one common interface.
* Decompose so that swapping the preference model (Matrix Factorization vs NeuMF) never requires touching data, aggregation, or evaluation code. The comparison must isolate the effect of the preference model alone.
* Keep shared, model-agnostic concerns separated and independently testable: dataset binarization/splitting, negative sampling, the aggregation function, and the evaluation metrics. Each preference model is its own independently testable unit behind the shared interface.
* Enforce boundaries structurally, not by convention. A module must never import another module's internals; cross-module communication happens via serialized artifacts on disk plus shared typed contracts and interfaces. (For example, evaluation should consume model artifacts, not import model code.)
* The API and frontend come last. Core modules must never depend on them.
* One cohesive subsystem per chat. When in doubt, split for maintainability rather than merging for brevity.

# Rules every generated prompt must embed

Fold these into the prompts themselves — do not emit them as separate documents:

* pytest is the test runner; enforce type hints throughout.
* Never modify tests to make a failing implementation pass.
* TDD loop: write tests first -> run them and verify they fail for the right reason -> implement -> run pytest -> confirm they pass.
* Preserve public interfaces and architectural boundaries; downstream chats may not redefine shared contracts.
* Apply the code-structure skill: actions/orchestration own domain rules ("why/when"); a service layer owns reusable operational mechanics ("how") with a composable, explicit-input, structured-output API. Extract to the service layer only when logic is shared by 2+ callers.

# For each chat session, produce ONE self-contained file. Name it like number_chatname_init.md e.g. 1_architecture.md so I know the order to complete one session before starting another. The intended workflow is: paste the file into a fresh Plan Mode chat, review the plan, then click Build to implement — there is no separate follow-up paste. Each file includes:

## Chat name

A short, descriptive name (e.g. Architecture, Data, a preference-model session, Evaluation, Experiments, API, Frontend).

## Purpose

One or two sentences describing exactly what this chat owns.

## Dependencies

Which prior chats or committed artifacts this chat relies on, and which it must not touch.

## Initial Plan Mode prompt

This is the most important output. Write the exact prompt I should paste into a fresh Cursor chat with Plan Mode enabled. It must:

* constrain scope tightly;
* define responsibilities;
* specify allowed dependencies;
* specify forbidden dependencies;
* preserve separation of concerns;
* name the REF pieces this session owns (e.g. which of: directional score, aggregation, embeddings, artifact schema, metrics);
* state testing expectations;
* avoid implementation details unless they are load-bearing for the boundary.

Write it as ready-to-paste text that instructs the agent to produce a plan only, not code — and note that after I review the plan and click Build, it implements under the Build requirements below.

## Build requirements (inside the same prompt)

End the Plan Mode prompt with a "Build requirements" block that takes effect when I click Build. It must enforce the TDD loop above: generate tests first, verify failures, implement, run pytest (plus import-linter, mypy, ruff), never edit tests to fit the implementation, and preserve interfaces and boundaries. Do not emit a separate follow-up file.

## Why this chat exists

Briefly justify why this subsystem deserves its own chat rather than sharing context with another.

# Global ordering

Recommend the order in which the chats should run. Indicate explicitly which chats can run in parallel, and state the committed artifact each chat must produce before any downstream chat starts.

# Output constraints

* The output should be almost entirely the prompts themselves.
* Do NOT produce standalone architecture, milestone, risk, or global-rule documents. Incorporate architectural constraints, risks, maintainability concerns, and testing philosophy into the prompts.
* Optimize for: modularity, architecture preservation, reproducibility, minimal cross-chat coupling, future extensibility, and preventing AI agents from turning the repository into a monolith.

# Last not least, add a prompts/README.md to explain what we're doing and the order of execution. Be intuitive, illuminating but not verbose - the fewer words the better. 