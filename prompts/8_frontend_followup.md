# Follow-up implementation prompt — Frontend

Paste this after you approve the plan, to move from planning to implementation under TDD.

```
The plan is approved. Implement frontend/ test-first using the project's JS/TS test runner (e.g. Vitest + Testing Library). Do not write a component before a test that exercises it exists and fails.

Workflow, repeat per capability (API client, recommendations view, loading/error states):
1. Write the tests first against a mocked API: the recommendations list renders the ranked ids returned by /recommend, a loading state shows while the request is in flight, and 404 (unknown user) and 422 (bad input) responses surface as clean user-facing messages.
2. Run the tests and show me them failing for the right reason before writing any implementation.
3. Implement the minimum component and client code to pass, calling the backend only over HTTP via the typed client.
4. Run the tests again and show all passing.

Hard rules:
- Never modify a test to make a failing implementation pass. If a test is genuinely wrong, fix it deliberately and re-verify it fails for the right reason first.
- No recommendation/scoring/ranking logic in the frontend; it only calls the API. Keep components thin and API access in a typed client module.
- Type everything; keep the linter clean; keep dependencies minimal.

When done, commit frontend/ and report how to run it against the API (env/base URL and the dev command).
```
