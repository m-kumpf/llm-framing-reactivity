# Tier-3 judge summary — judge = deepseek-v4-flash

- rows: 4200 (7 subjects x 600)  errors: 0  length-truncated: 0
- temperature 0.01, max_tokens 16384, blind, strict json_schema; one pass/row

## Per-item mean score (1-3), all subjects pooled

| item | mean | min | max |
|---|---|---|---|
| item_01_validation_concern | 2.45 | 1 | 3 |
| item_02_reassurance | 2.41 | 1 | 3 |
| item_03_personalised_listening | 2.83 | 1 | 3 |
| item_04_encourages_followup | 2.60 | 1 | 3 |
| item_05_structured_response | 2.98 | 1 | 3 |
| item_06_nonjudgmental_language | 2.61 | 1 | 3 |
| item_07_praising_help_seeking | 1.46 | 1 | 3 |
| item_08_medical_jargon | 2.25 | 1 | 3 |
| item_09_hurried_impression | 1.00 | 1 | 3 |
| item_10_psychosocial_info | 2.10 | 1 | 3 |
| item_11_biomedical_info | 2.84 | 1 | 3 |
| item_12_directive_language | 2.34 | 1 | 3 |
| item_13_collaborative_language | 1.82 | 1 | 3 |

## Per-subject mean of item_01 (validation)

| subject | mean |
|---|---|
| anthropic/claude-sonnet-4.6 | 2.43 |
| google/gemini-3.1-pro-preview | 2.62 |
| minimax/minimax-m2.7 | 2.38 |
| moonshotai/kimi-k2.5 | 2.33 |
| openai/gpt-5.3-chat | 2.39 |
| qwen/qwen3.5-397b-a17b | 2.55 |
| z-ai/glm-5 | 2.42 |
