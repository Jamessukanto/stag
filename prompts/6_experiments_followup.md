# Follow-up implementation prompt — Experiments

Paste this after you approve the plan, to move from planning to implementation under TDD.

```
The plan is approved. Implement experiments/ strictly test-first. Do not write implementation code before its test exists and fails.

Workflow, repeat per capability (config loading, the train step per model, the evaluate sweep, result collection, end-to-end pipeline):
1. Write the pytest tests first, using the shared synthetic fixture and small config so the full pipeline runs fast. Cover: the pipeline runs end to end and produces a comparison table with every model x aggregation x k cell populated, results are reproducible byte-for-byte (or within float tolerance) under a fixed seed, and the resolved config is persisted with the results.
2. Run pytest and show me the tests failing for the right reason before writing any implementation.
3. Implement the minimum code to pass, importing only public interfaces from data/, models/, and eval/.
4. Run pytest again and show all tests passing.

Hard rules:
- Never modify a test to make a failing implementation pass. If a test is genuinely wrong, fix it deliberately and re-verify it fails for the right reason first.
- Do not redefine or edit src/core/. Do not import any module's internals and do not copy training, scoring, aggregation, or metric logic; if a public interface is missing something, stop and report it as an upstream gap.
- Keep the run as composable orchestration steps (code-structure skill), not a monolith.
- Type hints everywhere; keep mypy and ruff clean; one seed drives the whole run.

When done, commit experiments/ and report the entry-point command and the results file location/schema, so the API chat can serve from trained artifacts.
```
