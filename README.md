# Emotional Framing and the Communication Style of LLMs in Clinical Scenarios

Materials, data, and analysis code for:

> Kumpf M, Kliebisch N, Stambollxhiu E, et al. *Effect of Emotional Framing of Patient Messages on the Communication Style of Large Language Models in Clinical Scenarios.* (submitted; citation to be added upon publication)

Ten clinical scenarios were rewritten in six emotional framings with clinical content held constant and presented to seven frontier LLMs (10 repetitions; 4,200 responses). Communication style was quantified with deterministic text metrics, a GoEmotions classifier, and a 13-item interpersonal rubric scored by an LLM judge validated against 210 double-coded human ratings. A companion IPIP-50 assessment characterized model trait profiles.

## Repository layout

```
Comp1_IPIP-50/                     IPIP-50 personality assessment (component 1)
  config.yaml                      models, sampling settings, verbatim task prompts
  ipip_big5_50_items.csv           the 50 IPIP items with factor keys
  scripts/                         collect -> validate -> score -> analyse -> tables
  output/                          raw and scored responses, statistics

Comp2_framing-reactivity/          framing-reactivity experiment (component 2)
  Clinical_Vignettes_10x6_...md    the 60 vignettes (10 scenarios x 6 framings)
  codebook.json                    Ruben 13-item rubric as used by judge and raters
  tier3_fewshot_examples.json      7 worked examples embedded in the judge prompt
  validation_ratings_*.json        the two blinded human raters' codings (n = 210)
  run-judge-tier3-deepseek/        per-model judge scores for all 4,200 responses
  scripts/                         full pipeline (see below)
  output/                          data at every pipeline stage
```

## Pipeline (Comp2)

| Stage | Script | Input | Output |
|---|---|---|---|
| 1. Collect | `collect_responses.py` | vignettes (embedded), OpenRouter API | `output/collect_responses/` (raw; not tracked) |
| 2. Clean | `clean_responses.py` | raw responses | `output/clean_responses/` (included, 4,200 texts) |
| 3. Score: deterministic | `score_deterministic.py` | clean responses | `output/score_deterministic/` |
| 4. Score: emotion | `score_goemo.py` | clean responses | `output/score_goemo/` (per-response CSV included) |
| 5. Judge prompt | `build_judge_prompt.py` | `codebook.json` + few-shots | (library; see `test_build_judge_prompt.py`) |
| 6. Judge run | `run_judge_tier3.py` | clean responses + judge prompt | `run-judge-tier3-deepseek/` (included) |
| 7. Validation sample | `sample_pilot.py`, `sample_validation.py`, `build_pilot_ui.py` | clean responses | `output/sample_validation/` (included, incl. rater UI) |
| 8. Agreement gate | `analyse_tier3_judge.py` | rater JSONs + judge CSVs | `output/analyse_tier3_judge/` (incl. `gate_verdicts.json`) |
| 9. Merge | `merge_scores.py` | scored CSVs + gate | `output/merge_scores/scored_merged.csv` |
| 10. Statistics | `analyse.py`, `stats_deterministic.py`, `stats_goemo.py`, `analyse_tier3.py` | merged/scored data | `output/analysis/`, `output/stats_*/`, `output/analyse_tier3/` |
| 11. Bridge | `bridge1_2.py` | Comp1 summary + Comp2 analysis | `output/bridge1-2/` |
| 12. Tables | `tab_tier3.py` | analysis outputs | `output/tab_tier3/` |

Comp1 runs analogously: `collect_responses.py` → `validate_responses.py` → `score_responses.py` → `analyse_results.py` → `tab_stats.py`.

All intermediate data needed to run stages 3–12 without any API access are included; stages 1, 2, and 6 are reproducible in full from the included inputs given an API key. Figure-generation scripts are not part of this repository; figures in the paper derive from the included analysis outputs.

## Reproducing the analyses

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r Comp2_framing-reactivity/requirements.txt   # or Comp1_IPIP-50/requirements.txt

cd Comp2_framing-reactivity                       # --force overwrites the shipped outputs
python scripts/analyse_tier3_judge.py --force     # agreement gate (Table 1)
python scripts/analyse_tier3.py --force           # confirmatory permutation ANOVA (Table 2)
python scripts/stats_deterministic.py --force     # Tier-1 statistics
python scripts/stats_goemo.py --force             # Tier-2 statistics
python scripts/analyse.py --force                 # PCA / clustering / reactivity
python scripts/bridge1_2.py --force               # pre-specified exploratory bridge
```

All analyses are seed-fixed and reruns reproduce the reported statistics exactly: the confirmatory Tier-3 analyses use seed 42 with 10,000 permutations; the exploratory permutation tests in `analyse.py` and `stats_*.py` use seed 2024 with 2,000 permutations. One caveat: in the effect-size tables (`output/analyse_tier3/effects_*.csv`), cells at the scoring ceiling are quasi-separated (see their `note` column) and those unstable OR estimates and CI bounds can vary with the installed statsmodels version; all permutation inference, cell means, and well-identified effect sizes reproduce exactly. Guard tests: `test_build_judge_prompt.py` (judge blinding is enforced by construction) and `test_codebook_matches_ui.py` (the judge saw the identical rubric shown to human raters).

To re-collect responses (stage 1), set an OpenRouter API key: create `.env` with `OPENROUTER_API_KEY=...` or export it. To re-run the judge (stage 6), point `run_judge_tier3.py` at any OpenAI-compatible endpoint via `--judge-endpoint`/`JUDGE_ENDPOINT` and `JUDGE_API_KEY`; the production run used deepseek-v4-flash. Models and versions are pinned by API identifier in `Comp1_IPIP-50/config.yaml` and Table S4 of the paper; note that provider-side model updates may alter outputs relative to the April 2026 collection.

## Data notes

- All patient vignettes are fully synthetic, authored by the study team; no real patient data appear anywhere in this repository.
- `output/clean_responses/` contains the complete study corpus: 4,200 model responses (7 models × 6 framings × 10 scenarios × 10 repetitions).
- `validation_ratings_*.json` are the independent codings of the two blinded human raters on the 210-response validation sample.
- The judge few-shot bundle in `tier3_fewshot_examples.json` is the 7-example version used in the production judge run; the four clinical topics of its examples lie outside the 10 study scenarios.
- The bulky per-sentence classifier output (`scored_goemo_sentences.csv`, ~72 MB) is excluded; `score_goemo.py` regenerates it from the included responses.

## License

MIT (see `LICENSE`).
