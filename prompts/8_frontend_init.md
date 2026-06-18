Purpose
A lightweight React + TypeScript application that visualises the evaluation comparison table and lets users browse recommendations per model configuration. Communicates exclusively with the API layer.

Dependencies
API chat (endpoint contracts). Architecture chat is not a dependency — the frontend has no Python dependency.

You are planning the frontend/ module of a reciprocal recommendation system. This is a Plan Mode session. Produce a plan only — no code.

This is a lightweight React + TypeScript frontend. It visualises reciprocal recommendation results served by the api/ module. It has no direct access to Python modules, artifact files, or the file system.

Module responsibility:
- Display a comparison table: one row per experiment configuration (model × sampling strategy), columns for RP@K and Mutual Hit Rate.
- Allow a user ID to be entered; display top-K recommendations from each model side by side.
- Fetch all data from the API module via HTTP. API base URL is controlled by the environment variable VITE_API_BASE_URL.

Constraints:
- Lives in frontend/. Completely isolated from all Python code.
- React + TypeScript. Vite as build tool.
- No direct file system access, no Python imports.
- All API calls are async with loading and error states handled visibly in the UI.
- No hard-coded user IDs or artifact IDs.

Plan must address:
1. Component tree (names and responsibilities).
2. Which API endpoints are consumed and how their response types are typed in TypeScript.
3. State management approach (React state is sufficient for this scope — justify if you want something heavier).
4. How API errors (404, network failure) surface to the user without crashing the UI.
5. Build and dev server configuration (Vite, proxy for local API).

