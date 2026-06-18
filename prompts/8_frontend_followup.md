Plan approved. Implement frontend/.

Testing rules:
1. Write component tests first using Vitest + React Testing Library.
2. Run npm test — confirm tests fail before implementation.
3. Implement to make tests pass.
4. Never modify tests to satisfy implementations.
5. Run npm test after every component change.

All API calls must be mocked in tests using MSW (Mock Service Worker) or vi.fn(). No real HTTP requests in tests.

Required tests:
- test_comparison_table_renders: given mocked /artifacts and /artifacts/:id/evaluation responses, the table renders one row per configuration.
- test_user_search_triggers_api_call: entering a user ID and submitting calls GET /artifacts/:id/recommendations/:userId.
- test_loading_state_shown: while a fetch is in flight, a loading indicator is visible.
- test_api_error_shown: when the API returns 404, a human-readable error message is shown — no uncaught exceptions.
- test_empty_artifacts: when /artifacts returns an empty list, a descriptive empty-state message is shown rather than a broken table.

TypeScript strict mode must be enabled (strict: true in tsconfig.json). No any types except where unavoidable, and those must be commented.