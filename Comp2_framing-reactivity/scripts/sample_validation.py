#!/usr/bin/env python3
"""
Draw the Tier 3 *validation* sample (210 items) for human-vs-LLM-judge reliability.

This expands the existing 21-row pilot into a balanced, fully-factorial 210-row
validation set, anchored on the pilot so prior in-progress ratings carry over.

Design
------
1. Rows 1-21 are the existing pilot rows, copied VERBATIM from
   ``output/sample_pilot/pilot_sample.csv`` in their original order (NOT redrawn
   from RNG). The only column changed is ``_set`` (``pilot`` -> ``validation``);
   all other columns — patient/response text and model/run/scenario/framing
   metadata — are byte-identical to the pilot.
2. Balanced full factorial over the 42 (model x framing) cells, 5 rows per cell
   = 210. Each cell uses 5 DISTINCT scenarios, one run each, so no
   (model, framing, scenario) triple is ever reused and no patient prompt repeats
   within a cell.
3. The 21 pilot cells already hold 1 scenario each, so they get 4 more distinct
   scenarios (84 rows); the 21 empty cells get 5 distinct scenarios each
   (105 rows). 84 + 105 = 189 new rows + 21 pilot = 210. Final marginals:
   per-cell = 5, per-model = 30, per-framing = 35.
4. Scenario balance target: exactly 21 rows per scenario across the full 210
   (10 scenarios x 21 = 210). The pilot pre-seeds the scenario counts; the 189
   new slots fill the remaining deficits, drawn preferring the scenarios with
   the largest remaining deficit so demand is never stranded. The script fails
   loudly if the constraints cannot be satisfied.
5. Any (model, framing, scenario, run_number) already present in the pilot is
   excluded from the new draw. Only the 189 new rows are shuffled (into a
   randomised order, sample_id 22-210); rows 1-21 are never reordered.

Reproducibility
---------------
All randomness comes from a single ``random.Random(seed)`` (default 42, CLI
``--seed``), drawn in a fixed order: (a) cell-order shuffle inside the scenario
assignment, (b) per-slot scenario picks, (c) per-slot run picks, (d) the final
shuffle of the 189 new rows. Re-running with the same seed is bit-identical.

The scenario assignment uses a seeded largest-deficit-first greedy with a bounded
retry loop; if that ever dead-ends on a satisfiable instance, a deterministic
max-flow fallback completes it, so the script never spuriously fails.

Input:  output/clean_responses/responses_*.csv
        output/sample_pilot/pilot_sample.csv
Output: output/sample_validation/validation_sample.csv
        output/sample_validation/run_metadata.json

Usage:
    python sample_validation.py
    python sample_validation.py --seed 42
    python sample_validation.py --pilot-csv path/to/pilot_sample.csv
"""

import argparse
import csv
import glob
import json
import os
import logging
import random
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "output" / "clean_responses"
PILOT_CSV = PROJECT_ROOT / "output" / "sample_pilot" / "pilot_sample.csv"
OUT_DIR = PROJECT_ROOT / "output" / "sample_validation"


def _rel(p):
    """Path as recorded in metadata: relative to the component root."""
    return os.path.relpath(p, PROJECT_ROOT)

# Model display names (for internal metadata, hidden from raters).
# Mirrors sample_pilot.py so _model_short matches the pilot exactly.
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
    "_set",  # "validation" for the whole file (rows 1-21 flipped from "pilot")
]

ALL_COLUMNS = RATER_COLUMNS + RUBEN_ITEMS + META_COLUMNS

N_MODELS_EXPECTED = 7
N_FRAMINGS_EXPECTED = 6
N_SCENARIOS_EXPECTED = 10
N_PILOT_EXPECTED = 21
N_PER_CELL = 5          # distinct scenarios per (model x framing) cell
PER_SCENARIO_TARGET = 21
N_TOTAL = 210

SET_NAME = "validation"

log = logging.getLogger("sample_validation")


# ── Helpers (mirrored from sample_pilot.py so this script stays stdlib-only) ──
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


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)


# ── Pilot ingest ────────────────────────────────────────────────────────────
def load_pilot_rows(pilot_csv: Path) -> list[dict]:
    """Load the 21 pilot rows verbatim, preserving order and schema."""
    with open(pilot_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ALL_COLUMNS:
            raise RuntimeError(
                "Pilot CSV schema does not match expected ALL_COLUMNS.\n"
                f"    expected: {ALL_COLUMNS}\n"
                f"    got:      {reader.fieldnames}"
            )
        rows = list(reader)
    if len(rows) != N_PILOT_EXPECTED:
        raise RuntimeError(f"Expected {N_PILOT_EXPECTED} pilot rows, got {len(rows)}")
    ids = [int(r["sample_id"]) for r in rows]
    if ids != list(range(1, N_PILOT_EXPECTED + 1)):
        raise RuntimeError(f"Pilot sample_ids are not 1..{N_PILOT_EXPECTED} in order: {ids}")
    return rows


# ── Source index ────────────────────────────────────────────────────────────
def build_triple_index(rows: list[dict]) -> dict:
    """(model_short, framing_label, scenario_id) -> {run_number: source_row}."""
    index: dict = defaultdict(dict)
    for row in rows:
        key = (
            model_short_name(row.get("model", "")),
            row.get("framing_label", ""),
            row.get("scenario_id", ""),
        )
        index[key][row.get("run_number", "")] = row
    return index


# ── Scenario assignment: greedy (primary) + max-flow (fallback) ─────────────
def assign_scenarios_greedy(
    cells: list[tuple[str, str]],
    seeded: set[tuple[str, str]],
    pilot_scenario: dict[tuple[str, str], str],
    scenarios: list[str],
    deficits: dict[str, int],
    rng: random.Random,
    max_attempts: int = 200,
) -> dict[tuple[str, str], set[str]] | None:
    """Largest-deficit-first greedy with bounded retries. Returns None on failure."""
    for _ in range(max_attempts):
        deficit = dict(deficits)
        assigned: dict[tuple[str, str], set[str]] = {c: set() for c in cells}
        order = list(cells)
        rng.shuffle(order)
        ok = True
        for cell in order:
            need = N_PER_CELL - (1 if cell in seeded else 0)
            blocked = {pilot_scenario[cell]} if cell in seeded else set()
            for _ in range(need):
                cands = [
                    s for s in scenarios
                    if s not in blocked and s not in assigned[cell] and deficit[s] > 0
                ]
                if not cands:
                    ok = False
                    break
                max_def = max(deficit[s] for s in cands)
                pool = sorted(s for s in cands if deficit[s] == max_def)
                chosen = rng.choice(pool)
                assigned[cell].add(chosen)
                deficit[chosen] -= 1
            if not ok:
                break
        if ok and all(v == 0 for v in deficit.values()):
            return assigned
    return None


def assign_scenarios_maxflow(
    cells: list[tuple[str, str]],
    seeded: set[tuple[str, str]],
    pilot_scenario: dict[tuple[str, str], str],
    scenarios: list[str],
    deficits: dict[str, int],
) -> dict[tuple[str, str], set[str]]:
    """Deterministic exact assignment via Edmonds-Karp max-flow.

    source -> scenario (cap = deficit) -> cell (cap 1, allowed pairs only)
    -> sink (cap = cell need). Guaranteed to realise the design if feasible.
    """
    # Node ids: 0 = source, 1..S = scenarios, S+1..S+C = cells, last = sink.
    s_index = {s: i + 1 for i, s in enumerate(scenarios)}
    c_index = {c: len(scenarios) + 1 + i for i, c in enumerate(cells)}
    source, sink = 0, len(scenarios) + len(cells) + 1
    n_nodes = sink + 1

    cap = [defaultdict(int) for _ in range(n_nodes)]

    def add_edge(u: int, v: int, c: int) -> None:
        cap[u][v] += c

    for s in scenarios:
        add_edge(source, s_index[s], deficits[s])
    for cell in cells:
        need = N_PER_CELL - (1 if cell in seeded else 0)
        add_edge(c_index[cell], sink, need)
        blocked = pilot_scenario.get(cell) if cell in seeded else None
        for s in scenarios:
            if s != blocked:
                add_edge(s_index[s], c_index[cell], 1)

    target = sum(deficits.values())
    flow = 0
    while True:
        parent = [-1] * n_nodes
        parent[source] = source
        q = deque([source])
        while q:
            u = q.popleft()
            for v, c in cap[u].items():
                if c > 0 and parent[v] == -1:
                    parent[v] = u
                    if v == sink:
                        q.clear()
                        break
                    q.append(v)
        if parent[sink] == -1:
            break
        # augment by 1 (all relevant capacities are unit on the bottleneck path)
        v = sink
        bottleneck = float("inf")
        while v != source:
            u = parent[v]
            bottleneck = min(bottleneck, cap[u][v])
            v = u
        v = sink
        while v != source:
            u = parent[v]
            cap[u][v] -= bottleneck
            cap[v][u] += bottleneck
            v = u
        flow += bottleneck

    if flow != target:
        raise RuntimeError(
            "Scenario design is INFEASIBLE: max-flow assigned "
            f"{flow}/{target} new slots. Check the per-scenario targets and "
            "per-cell distinctness constraints."
        )

    assigned: dict[tuple[str, str], set[str]] = {c: set() for c in cells}
    for cell in cells:
        for s in scenarios:
            # a saturated scenario->cell edge shows up as residual back-capacity
            if cap[c_index[cell]][s_index[s]] > 0:
                assigned[cell].add(s)
    return assigned


# ── Verification ────────────────────────────────────────────────────────────
def verify(all_rows: list[dict], pilot_rows: list[dict], exclusion: set) -> dict:
    """Assert every design constraint; raise RuntimeError on any violation.
    Returns the marginal tables."""
    if len(all_rows) != N_TOTAL:
        raise RuntimeError(f"Expected {N_TOTAL} rows, got {len(all_rows)}")

    ids = [int(r["sample_id"]) for r in all_rows]
    if sorted(ids) != list(range(1, N_TOTAL + 1)):
        raise RuntimeError("sample_ids are not a permutation of 1..210")

    # Rows 1-21 identical to pilot on every column EXCEPT _set.
    for i in range(N_PILOT_EXPECTED):
        v, p = all_rows[i], pilot_rows[i]
        if int(v["sample_id"]) != i + 1:
            raise RuntimeError(f"Row {i} is not pilot sample_id {i + 1}")
        for col in ALL_COLUMNS:
            if col == "_set":
                continue
            if v[col] != p[col]:
                raise RuntimeError(
                    f"Row {i + 1} differs from pilot on column {col!r}:\n"
                    f"    validation: {v[col]!r}\n    pilot:      {p[col]!r}"
                )
        if v["_set"] != SET_NAME:
            raise RuntimeError(f"Row {i + 1} _set is {v['_set']!r}, expected {SET_NAME!r}")

    # Per-cell = 5 distinct scenarios, no prompt repeats within a cell.
    by_cell: dict = defaultdict(list)
    for r in all_rows:
        by_cell[(r["_model_short"], r["framing_label"])].append(r)
    if len(by_cell) != N_MODELS_EXPECTED * N_FRAMINGS_EXPECTED:
        raise RuntimeError(f"Expected 42 cells, got {len(by_cell)}")
    for cell, rs in by_cell.items():
        if len(rs) != N_PER_CELL:
            raise RuntimeError(f"Cell {cell} has {len(rs)} rows, expected {N_PER_CELL}")
        scen = [r["scenario_id"] for r in rs]
        if len(set(scen)) != N_PER_CELL:
            raise RuntimeError(f"Cell {cell} has non-distinct scenarios: {scen}")
        prompts = [r["prompt"] for r in rs]
        if len(set(prompts)) != len(prompts):
            raise RuntimeError(f"Cell {cell} has a repeated prompt")

    # Marginals.
    per_model = Counter(r["_model_short"] for r in all_rows)
    per_framing = Counter(r["framing_label"] for r in all_rows)
    per_scenario = Counter(r["scenario_id"] for r in all_rows)
    bad = {m: c for m, c in per_model.items() if c != 30}
    if bad:
        raise RuntimeError(f"Per-model != 30: {bad}")
    bad = {f: c for f, c in per_framing.items() if c != 35}
    if bad:
        raise RuntimeError(f"Per-framing != 35: {bad}")
    bad = {s: c for s, c in per_scenario.items() if c != PER_SCENARIO_TARGET}
    if bad:
        raise RuntimeError(f"Per-scenario != {PER_SCENARIO_TARGET}: {bad}")

    # Uniqueness of (model, framing, scenario) and (.. , run).
    triples = [(r["_model_short"], r["framing_label"], r["scenario_id"]) for r in all_rows]
    if len(set(triples)) != N_TOTAL:
        raise RuntimeError("Duplicate (model, framing, scenario) triple found")
    quads = [
        (r["_model_short"], r["framing_label"], r["scenario_id"], r["_run_number"])
        for r in all_rows
    ]
    if len(set(quads)) != N_TOTAL:
        raise RuntimeError("Duplicate (model, framing, scenario, run) combo found")

    # No NEW row collides with the pilot exclusion set.
    for r in all_rows[N_PILOT_EXPECTED:]:
        key = (r["_model_short"], r["framing_label"], r["scenario_id"], r["_run_number"])
        if key in exclusion:
            raise RuntimeError(f"New row reuses a pilot (model,framing,scenario,run): {key}")
        if r["_set"] != SET_NAME:
            raise RuntimeError(f"New row _set is {r['_set']!r}, expected {SET_NAME!r}")
        if any(r[item] != "" for item in RUBEN_ITEMS):
            raise RuntimeError(f"New row {r['sample_id']} has a non-empty Ruben item column")

    per_cell = {f"{m} | {f}": len(rs) for (m, f), rs in sorted(by_cell.items())}
    return {
        "per_cell": per_cell,
        "per_model": dict(sorted(per_model.items())),
        "per_framing": dict(sorted(per_framing.items())),
        "per_scenario": dict(sorted(per_scenario.items())),
    }


def format_new_row(row: dict, sample_id: int) -> dict:
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
        "_set": SET_NAME,
    }
    for item in RUBEN_ITEMS:
        out[item] = ""
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Draw the Tier 3 validation sample (210 items, anchored on the pilot)."
    )
    parser.add_argument(
        "--dir", type=Path, default=DATA_DIR,
        help=f"Directory containing responses_*.csv (default: {DATA_DIR})",
    )
    parser.add_argument(
        "--pilot-csv", type=Path, default=PILOT_CSV,
        help=f"Existing pilot_sample.csv to anchor on (default: {PILOT_CSV})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUT_DIR,
        help=f"Output directory (default: {OUT_DIR})",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the new draw (default: 42)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    rng = random.Random(args.seed)

    # 1. Load sources.
    csv_files = find_response_csvs(args.dir)
    if not csv_files:
        log.error("No responses_*.csv found in %s", args.dir)
        return 1
    log.info("Found %d response file(s) in %s", len(csv_files), args.dir)
    responses = load_responses(csv_files)
    log.info("Loaded %d valid responses (ERROR rows skipped)", len(responses))

    if not args.pilot_csv.exists():
        log.error("Pilot CSV not found: %s", args.pilot_csv)
        return 1
    pilot_rows = load_pilot_rows(args.pilot_csv)
    log.info("Loaded %d pilot rows from %s", len(pilot_rows), args.pilot_csv)

    # 2. Design skeleton.
    triple_index = build_triple_index(responses)
    models = sorted({model_short_name(r.get("model", "")) for r in responses})
    framings = sorted({r.get("framing_label", "") for r in responses})
    scenarios = sorted({r.get("scenario_id", "") for r in responses})
    if (len(models), len(framings), len(scenarios)) != (
        N_MODELS_EXPECTED, N_FRAMINGS_EXPECTED, N_SCENARIOS_EXPECTED,
    ):
        log.error(
            "Expected %dx%dx%d; got %dx%dx%d models/framings/scenarios. Aborting.",
            N_MODELS_EXPECTED, N_FRAMINGS_EXPECTED, N_SCENARIOS_EXPECTED,
            len(models), len(framings), len(scenarios),
        )
        return 1

    cells = [(m, f) for m in models for f in framings]
    pilot_scenario: dict[tuple[str, str], str] = {}
    exclusion: set = set()
    for r in pilot_rows:
        cell = (r["_model_short"], r["framing_label"])
        if cell in pilot_scenario:
            raise RuntimeError(f"Pilot has two rows in the same cell {cell}")
        pilot_scenario[cell] = r["scenario_id"]
        exclusion.add(
            (r["_model_short"], r["framing_label"], r["scenario_id"], r["_run_number"])
        )
    seeded = set(pilot_scenario)
    if len(seeded) != N_PILOT_EXPECTED:
        raise RuntimeError(f"Expected {N_PILOT_EXPECTED} seeded cells, got {len(seeded)}")

    pilot_scen_counts = Counter(r["scenario_id"] for r in pilot_rows)
    deficits = {s: PER_SCENARIO_TARGET - pilot_scen_counts.get(s, 0) for s in scenarios}
    if any(v < 0 for v in deficits.values()):
        raise RuntimeError(f"Negative scenario deficit (pilot over-uses a scenario): {deficits}")
    n_new_expected = N_TOTAL - N_PILOT_EXPECTED
    if sum(deficits.values()) != n_new_expected:
        raise RuntimeError(
            f"Scenario deficits sum to {sum(deficits.values())}, expected {n_new_expected}"
        )
    log.info("New-slot scenario deficits: %s (sum=%d)", dict(sorted(deficits.items())),
             sum(deficits.values()))

    # 3. Assign 5 distinct scenarios per cell (greedy, then deterministic fallback).
    assigned = assign_scenarios_greedy(
        cells, seeded, pilot_scenario, scenarios, deficits, rng,
    )
    if assigned is None:
        log.warning("Greedy assignment dead-ended; using deterministic max-flow fallback.")
        assigned = assign_scenarios_maxflow(
            cells, seeded, pilot_scenario, scenarios, deficits,
        )
    else:
        log.info("Greedy scenario assignment succeeded.")

    # 4. Pick a run per (cell, scenario) and materialise the 189 new rows.
    new_rows: list[dict] = []
    for cell in cells:
        m, f = cell
        for s in sorted(assigned[cell]):
            available = sorted(triple_index.get((m, f, s), {}).keys())
            pool = [
                run for run in available
                if (m, f, s, run) not in exclusion
            ]
            if not pool:
                raise RuntimeError(
                    f"No available (non-excluded) run for cell {cell}, scenario {s}"
                )
            chosen_run = rng.choice(pool)
            src = triple_index[(m, f, s)][chosen_run]
            new_rows.append(format_new_row(src, sample_id=-1))  # sample_id set after shuffle

    if len(new_rows) != n_new_expected:
        raise RuntimeError(f"Built {len(new_rows)} new rows, expected {n_new_expected}")

    # 5. Shuffle ONLY the new rows; assign sample_id 22..210. Pilot keeps 1..21.
    rng.shuffle(new_rows)
    for offset, row in enumerate(new_rows):
        row["sample_id"] = N_PILOT_EXPECTED + 1 + offset

    pilot_out = []
    for r in pilot_rows:
        rr = dict(r)
        rr["_set"] = SET_NAME  # flip pilot -> validation; everything else verbatim
        pilot_out.append(rr)

    all_rows = pilot_out + new_rows

    # 6. Verify all constraints.
    marginals = verify(all_rows, pilot_rows, exclusion)
    log.info("Verification passed: 210 rows, per-cell=5, per-model=30, per-framing=35, "
             "per-scenario=21, all triples unique, no pilot collisions.")

    # 7. Write outputs.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.output_dir / "validation_sample.csv"
    write_csv(all_rows, sample_path)
    log.info("Wrote validation sample: %s", sample_path)

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "n_total": N_TOTAL,
        "n_pilot_anchor": N_PILOT_EXPECTED,
        "n_new": n_new_expected,
        "set": SET_NAME,
        "rubric_items": RUBEN_ITEMS,
        "source_dir": _rel(args.dir),
        "source_files": [f.name for f in csv_files],
        "pilot_csv": _rel(args.pilot_csv),
        "design": {
            "n_models": len(models),
            "n_framings": len(framings),
            "n_scenarios": len(scenarios),
            "per_cell": N_PER_CELL,
            "models": models,
            "framings": framings,
            "scenarios": scenarios,
        },
        "scenario_deficits_filled": dict(sorted(deficits.items())),
        "marginals": marginals,
        "outputs": {"sample": sample_path.name},
        "pilot_rows_note": (
            "Rows 1-21 are identical to output/sample_pilot/pilot_sample.csv on all "
            "columns EXCEPT _set, which is flipped from 'pilot' to 'validation' so the "
            "whole file carries one clean per-set label."
        ),
    }
    meta_path = args.output_dir / "run_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    log.info("Wrote metadata: %s", meta_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
