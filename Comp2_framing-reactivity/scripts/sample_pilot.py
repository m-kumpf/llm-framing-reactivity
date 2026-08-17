#!/usr/bin/env python3
"""
Draw a stratified pilot sample for Tier 3 LLM-as-judge calibration.

Samples 21 items from the cleaned response dataset (7 models × 6 framings ×
10 scenarios × 10 reps = 4,200 rows) using a Latin-square-style allocation:
each model appears in exactly 3 framings, each framing appears in 3 or 4
models, with model-framing assignments randomised under those constraints.
Within each selected (model, framing) cell, one (scenario, run_number) is
drawn at random, preferring globally under-used scenarios where ties allow.

Both human raters score this sample independently on all 13 Ruben items,
then reconcile in a structured calibration meeting. The reconciled scores
feed the power analysis and supply the few-shot anchors for the LLM judge.

Both raters work from the same CSV in the same presentation order; rater
identity is captured at the UI level (via a rater_id input field) and
embedded in each rater's exported JSON. Keeping a single order across
raters simplifies the reconciliation meeting (item N is the same item
for both).

Input:  output/clean_responses/responses_*.csv
Output: output/sample_pilot/pilot_sample.csv
        output/sample_pilot/run_metadata.json

Usage:
    python sample_pilot.py
    python sample_pilot.py --seed 42
    python sample_pilot.py --dir /custom/path/to/clean_responses
"""

import argparse
import csv
import glob
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "output" / "clean_responses"
OUT_DIR = PROJECT_ROOT / "output" / "sample_pilot"

# Model display names (for internal metadata, hidden from raters).
_MODEL_NAMES = {
    "claude": "Claude Sonnet 4.6",
    "gpt": "GPT-5.3",
    "gemini": "Gemini 3.1 Pro",
    "kimi": "Kimi K2.5",
    "qwen": "Qwen 3.5",
    "minimax": "MiniMax M2.7",
    "glm": "GLM 5",
}

# Ruben (2026) 13-item rubric — column slugs in canonical order.
RUBEN_ITEMS = [
    "item_01_validation_concern",
    "item_02_reassurance",
    "item_03_personalised_listening",
    "item_04_encourages_followup",
    "item_05_structured_response",
    "item_06_nonjudgmental_language",
    "item_07_praising_help_seeking",
    "item_08_medical_jargon",
    "item_09_hurried_impression",
    "item_10_psychosocial_info",
    "item_11_biomedical_info",
    "item_12_directive_language",
    "item_13_collaborative_language",
]

RATER_COLUMNS = [
    "sample_id",
    "vignette_id",
    "scenario_id",
    "scenario_label",
    "scenario_domain",
    "framing_id",
    "framing_label",
    "prompt",
    "response_text",
]

META_COLUMNS = [
    "_model",
    "_model_short",
    "_run_number",
    "_set",  # always "pilot"
]

ALL_COLUMNS = RATER_COLUMNS + RUBEN_ITEMS + META_COLUMNS

N_MODELS_EXPECTED = 7
N_FRAMINGS_EXPECTED = 6
N_PER_MODEL = 3  # row sum in the Latin-square allocation

log = logging.getLogger("sample_pilot")


def model_short_name(raw: str) -> str:
    raw_lower = raw.lower()
    for key, name in _MODEL_NAMES.items():
        if key in raw_lower:
            return name
    return raw


def find_response_csvs(target_dir: Path) -> list[Path]:
    pattern_flat = str(target_dir / "responses_*.csv")
    pattern_nested = str(target_dir / "*" / "responses_*.csv")
    return sorted({Path(p) for p in glob.glob(pattern_flat) + glob.glob(pattern_nested)})


def load_responses(csv_files: list[Path]) -> list[dict]:
    rows = []
    for f in csv_files:
        with open(f, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("response_text", "").strip().upper().startswith("ERROR"):
                    continue
                rows.append(row)
    return rows


def build_cell_index(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """Index rows by (model_short, framing_label)."""
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        model = model_short_name(row.get("model", ""))
        framing = row.get("framing_label", "")
        cells[(model, framing)].append(row)
    return cells


def latin_square_assignments(
    models: list[str], framings: list[str], rng: random.Random
) -> list[tuple[str, str]]:
    """Build the 21 (model, framing) pairs via a randomised sliding window.

    With 7 models × 6 framings, model i (in shuffled order) is assigned the
    three framings at positions {i, i+1, i+2} mod 6 (in shuffled framing
    order). This guarantees row sums = 3 for each model and column sums
    = (4, 4, 4, 3, 3, 3) across framings, totalling 21 pairs.
    """
    if len(models) != N_MODELS_EXPECTED or len(framings) != N_FRAMINGS_EXPECTED:
        raise ValueError(
            f"Latin-square assignment requires "
            f"{N_MODELS_EXPECTED} models × {N_FRAMINGS_EXPECTED} framings; "
            f"got {len(models)} × {len(framings)}"
        )
    shuffled_models = rng.sample(models, len(models))
    shuffled_framings = rng.sample(framings, len(framings))
    pairs: list[tuple[str, str]] = []
    for i, model in enumerate(shuffled_models):
        for offset in range(N_PER_MODEL):
            framing = shuffled_framings[(i + offset) % len(shuffled_framings)]
            pairs.append((model, framing))
    return pairs


def draw_sample(
    cells: dict[tuple[str, str], list[dict]],
    pairs: list[tuple[str, str]],
    rng: random.Random,
) -> list[dict]:
    """For each (model, framing) pair, draw one rep, preferring scenarios
    with the lowest global usage so the 10 scenarios spread evenly across
    the 21 items."""
    scenario_counts: Counter = Counter()
    pair_order = list(pairs)
    rng.shuffle(pair_order)  # decorrelate scenario-balance bias from model order
    sampled: list[dict] = []
    for model, framing in pair_order:
        pool = cells.get((model, framing), [])
        if not pool:
            raise RuntimeError(f"Empty cell: {model} × {framing}")
        scenarios_in_pool = {row["scenario_id"] for row in pool}
        min_count = min(scenario_counts[s] for s in scenarios_in_pool)
        candidates = sorted(
            s for s in scenarios_in_pool if scenario_counts[s] == min_count
        )
        chosen_scenario = rng.choice(candidates)
        reps = [row for row in pool if row["scenario_id"] == chosen_scenario]
        sampled.append(rng.choice(reps))
        scenario_counts[chosen_scenario] += 1
    return sampled


def format_one(row: dict, sample_id: int) -> dict:
    out = {
        "sample_id": sample_id,
        "vignette_id": row.get("vignette_id", ""),
        "scenario_id": row.get("scenario_id", ""),
        "scenario_label": row.get("scenario_label", ""),
        "scenario_domain": row.get("scenario_domain", ""),
        "framing_id": row.get("framing_id", ""),
        "framing_label": row.get("framing_label", ""),
        "prompt": row.get("prompt", ""),
        "response_text": row.get("response_text", ""),
        "_model": row.get("model", ""),
        "_model_short": model_short_name(row.get("model", "")),
        "_run_number": row.get("run_number", ""),
        "_set": "pilot",
    }
    for item in RUBEN_ITEMS:
        out[item] = ""
    return out


def verify_marginals(rows: list[dict]) -> dict:
    """Confirm marginal constraints; raise on violation."""
    model_counts = Counter(r["_model_short"] for r in rows)
    framing_counts = Counter(r["framing_label"] for r in rows)
    scenario_counts = Counter(r["scenario_id"] for r in rows)

    bad_models = {m: c for m, c in model_counts.items() if c != N_PER_MODEL}
    if bad_models:
        raise RuntimeError(
            f"Model row-sum violation (expected {N_PER_MODEL} each): {bad_models}"
        )
    bad_framings = {f: c for f, c in framing_counts.items() if c not in (3, 4)}
    if bad_framings:
        raise RuntimeError(
            f"Framing column-sum violation (expected 3 or 4 each): {bad_framings}"
        )
    if sum(framing_counts.values()) != N_MODELS_EXPECTED * N_PER_MODEL:
        raise RuntimeError(f"Total rows ≠ 21: {sum(framing_counts.values())}")

    return {
        "per_model": dict(sorted(model_counts.items())),
        "per_framing": dict(sorted(framing_counts.items())),
        "per_scenario": dict(sorted(scenario_counts.items())),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Draw the Tier 3 pilot sample (21 items, Latin-square allocation)."
    )
    parser.add_argument(
        "--dir", type=Path, default=DATA_DIR,
        help=f"Directory containing responses_*.csv (default: {DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUT_DIR,
        help=f"Output directory (default: {OUT_DIR})",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    rng = random.Random(args.seed)

    csv_files = find_response_csvs(args.dir)
    if not csv_files:
        log.error("No responses_*.csv found in %s", args.dir)
        return 1
    log.info("Found %d response file(s) in %s", len(csv_files), args.dir)

    rows = load_responses(csv_files)
    log.info("Loaded %d valid responses (ERROR rows skipped)", len(rows))

    cells = build_cell_index(rows)
    models = sorted({m for m, _ in cells})
    framings = sorted({f for _, f in cells})
    log.info(
        "Design: %d models × %d framings = %d cells",
        len(models), len(framings), len(cells),
    )
    if len(models) != N_MODELS_EXPECTED or len(framings) != N_FRAMINGS_EXPECTED:
        log.error(
            "Expected %d × %d; got %d × %d. Aborting.",
            N_MODELS_EXPECTED, N_FRAMINGS_EXPECTED, len(models), len(framings),
        )
        return 1

    pairs = latin_square_assignments(models, framings, rng)
    log.info("Generated %d (model × framing) pairs via Latin-square allocation", len(pairs))

    sampled = draw_sample(cells, pairs, rng)
    log.info("Drew %d rows (one per pair)", len(sampled))

    # Master ordering: shuffle once, assign sample_id 1..N. The sample_id is a
    # stable join key shared across master + per-rater files.
    master_order = list(range(len(sampled)))
    rng.shuffle(master_order)
    master_rows = [format_one(sampled[src], sid) for sid, src in enumerate(master_order, 1)]

    marginals = verify_marginals(master_rows)
    log.info("Marginals OK — models: %s", marginals["per_model"])
    log.info("Marginals OK — framings: %s", marginals["per_framing"])
    log.info("Scenario spread: %s", marginals["per_scenario"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.output_dir / "pilot_sample.csv"
    write_csv(master_rows, sample_path)
    log.info("Wrote pilot sample: %s", sample_path)

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "n_total": len(master_rows),
        "n_per_model": N_PER_MODEL,
        "rubric_items": RUBEN_ITEMS,
        "source_dir": str(args.dir),
        "source_files": [f.name for f in csv_files],
        "design": {
            "n_models": len(models),
            "n_framings": len(framings),
            "models": models,
            "framings": framings,
        },
        "marginals": marginals,
        "outputs": {
            "sample": sample_path.name,
        },
    }
    meta_path = args.output_dir / "run_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    log.info("Wrote metadata: %s", meta_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
