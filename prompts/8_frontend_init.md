# Chat name

Frontend

## Purpose

A lightweight React client that lets a user enter an id and view the top-K reciprocal recommendations served by the API. It is purely a presentation layer over the API contract and contains no recommendation logic.

## Dependencies

- API (committed and stable): consumes its HTTP endpoints only. It has no knowledge of models, data, or evaluation, and no Python module may import it.

## Initial Plan Mode prompt

```
You are implementing a lightweight React frontend for the reciprocal-rec project, in a separate frontend/ directory. This is a Plan Mode session: produce a plan only, no implementation code.

The backend API is complete and frozen. The frontend talks to it exclusively over HTTP using the API's documented endpoint contracts (/health, /recommend, optionally /score). It contains no recommendation, scoring, or ranking logic; all of that stays in the backend.

Scope, strictly limited to frontend/:
1. A minimal single-page app: an input for user_id (and optional controls for k and aggregation), a button to fetch recommendations, and a list rendering the returned ranked target ids (and reciprocal score if available).
2. A typed API client wrapping the backend endpoints, with loading and error states (handle 404 unknown user and 422 validation cleanly).
3. Keep it small and modern: a current React + TypeScript setup (e.g. Vite). No heavyweight state management; local component state is enough. Clean, readable UI.

Constraints:
- The frontend is a leaf: it only calls the API. Do not embed any modeling/scoring logic. Do not import or reach into the Python backend; communicate only via HTTP and a documented base URL from config/env.
- Apply the code-structure skill's spirit: keep components (presentation) thin and put API access + response shaping in a small typed client/service module, not inside components.
- Type everything (TypeScript). Keep dependencies minimal.

Produce: the file/component layout, the typed API client surface, the component tree and state, the test approach (component/client tests with a mocked API: renders recommendations, shows loading, surfaces 404/422 errors), and decisions to confirm (tooling choice, whether to expose aggregation/k controls).
```

## Why this chat exists

The frontend uses a different language and toolchain (TypeScript/React) and a different testing style than the Python core, and it depends only on the stable HTTP contract. Isolating it keeps JS/TS tooling out of the Python sessions and ensures the UI can be rebuilt or replaced without touching — or even understanding — the research core.
