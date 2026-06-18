This folder contains all critical prompts I used to implement the project.

0_initial_meta_prompt is a meta prompt to plan all of the cursor window chats and their corresponding first prompts.

All other prompts in the folder are the outputs from 0_initial_meta_prompt, which are then used as the first prompt in separate cursor chat windows in plan mode.

## Ordering

Strict sequence (each new chat must complete before the next starts)
[1] Architecture         - Use prompt in 1_architecture.md
        │
        ▼
[2] Data and Models
All four of chats 2–5 depend on Architecture. They do not depend on each other. You can open all four in parallel immediately after chat 1's artifacts (src/core/) are committed. The Data and Model chats use a shared synthetic fixture in conftest.py; coordinate that fixture definition in the Architecture chat so all sessions inherit it.
Can run in parallel after Architecture completes [Architecture]
    ├──▶ [2] Data        - Use prompt in 2_data_init.md, 2_data_followup.md
    ├──▶ [3] NaiveModel  - Use prompt in 3_naive_model_init.md, 3_naive_model_followup.md
    ├──▶ [4] REFModel    - Use prompt in 4_ref_model_init.md, 4_ref_model_followup.md
    └──▶ [5] Evaluation  - Use prompt in 5_evaluation_init.md, 5_evaluation_followup.md
        │
        ▼
[3] Experiments          - Use prompt in 6_experiments_init.md, 6_experiments_followup.md
Open it only after all four module chats are passing their full test suites and their public interfaces are stable.
        │
        ▼
[7] API                  - Use prompt in 7_api_init.md, 7_api_followup.md
        │
        ▼
[8] Frontend             - Use prompt in 8_frontend_init.md, 8_frontend_followup.md

