#!/usr/bin/env python3
"""
Merge Tier 1 (deterministic), Tier 2 (GoEmotions) and Tier 3 (DeepSeek-V4-Flash
judge) scores into a single per-response file plus a cell-mean file.

Analysis-plan revision 2026-07: the merged cell-mean table (420 cells =
7 models x 6 framings x 10 scenarios) is the single unit of analysis for all
cross-tier work — convergent-validity correlations, the model x metric
reactivity matrix (PCA + Ward in analyse.py), and the composite reactivity
index that feeds the Comp1 trait bridge.

Tier 3 carries the 13 Ruben items plus the relationship-oriented composite
(mean of items 01, 02, 03, 06, 07, 10). Which Tier-3 columns count as
confirmatory is decided by the agreement gate (gate_verdicts.json), not here;
this script merges everything and records the gate verdicts in run_metadata.

Join key: (model, vignette_id, run_number). Expects 4,200 rows per tier.

Input:
    output/score_deterministic/scored_deterministic.csv
    output/score_goemo/scored_goemo.csv
    run-judge-tier3-deepseek/judge_*.csv
    output/analyse_tier3_judge/gate_verdicts.json       (optional, for metadata)

Output (output/merge_scores/):
    scored_merged.csv        — per response (4,200 rows)
    scored_merged_cells.csv  — per (model x framing x scenario) cell mean (420 rows)
    run_metadata.json

Usage:
    python merge_scores.py
    python merge_scores.py --deterministic path --goemo path --judge-dir dir --output-dir dir
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MERGE_KEY = ["model", "vignette_id", "run_number"]
ID_COLS = ["vignette_id", "scenario_id", "scenario_label",
           "framing_id", "framing_label", "model", "run_number", "model_short"]

TIER1_METRICS = ["word_count", "format_density", "questions", "avg_sent_len",
                 "coleman_liau", "first_person", "second_person", "hedging",
                 "fk_grade"]  # fk_grade supplementary
TIER2_METRICS = ["pos_emotion", "neg_emotion"]
TIER3_ITEMS = [
    "item_01_validation_concern", "item_02_reassurance",
    "item_03_personalised_listening", "item_04_encourages_followup",
    "item_05_structured_response", "item_06_nonjudgmental_language",
    "item_07_praising_help_seeking", "item_08_medical_jargon",
    "item_09_hurried_impression", "item_10_psychosocial_info",
    "item_11_biomedical_info", "item_12_directive_language",
    "item_13_collaborative_language",
]
# Ruben, Blanch-Hartigan & Hall (2026, JGIM) chatbot composites. The
# relationship-oriented component [1,2,3,6,7,10] is the validated PRIMARY
# composite — item 6 is a genuine member of the construct and stays in.
# The 5-item variant (item 6 removed) is an application-specific
# sensitivity check only (item 6 had poor human-human reliability with our
# raters). Item 9 is reverse-scored inside the conscientious composite.
RELATIONSHIP_ITEMS = ["item_01_validation_concern", "item_02_reassurance",
                      "item_03_personalised_listening", "item_06_nonjudgmental_language",
                      "item_07_praising_help_seeking", "item_10_psychosocial_info"]
TIER3_COMPOSITE = "relationship_oriented"
RELATIONSHIP_ITEMS_5 = [k for k in RELATIONSHIP_ITEMS
                        if k != "item_06_nonjudgmental_language"]
TIER3_COMPOSITE_5 = "relationship_oriented_5item"
RUBEN_OTHER = {
    "conscientious": (["item_09_hurried_impression", "item_13_collaborative_language"],
                      ["item_09_hurried_impression"]),           # (items, reversed)
    "guiding":       (["item_05_structured_response", "item_12_directive_language"], []),
    "technical":     (["item_08_medical_jargon", "item_11_biomedical_info"], []),
}
TIER3_COMPOSITES = [TIER3_COMPOSITE, TIER3_COMPOSITE_5] + \
    [f"composite_{name}" for name in RUBEN_OTHER]


def load_judge(judge_dir: Path) -> pd.DataFrame:
    files = sorted(judge_dir.glob("judge_*.csv"))
    if not files:
        sys.exit(f"ERROR: no judge_*.csv files in {judge_dir}")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df = df[df["status"] == "ok"].copy()
        frames.append(df)
        print(f"  {f.name}: {len(df)} ok rows")
    j = pd.concat(frames, ignore_index=True)
    for k in TIER3_ITEMS:
        j[k] = pd.to_numeric(j[k], errors="coerce")
    j[TIER3_COMPOSITE] = j[RELATIONSHIP_ITEMS].mean(axis=1)
    j[TIER3_COMPOSITE_5] = j[RELATIONSHIP_ITEMS_5].mean(axis=1)
    for name, (items, reversed_items) in RUBEN_OTHER.items():
        vals = j[items].copy()
        for rk in reversed_items:
            vals[rk] = 4 - vals[rk]  # 1<->3 on the 1-3 ordinal scale
        j[f"composite_{name}"] = vals.mean(axis=1)
    keep = MERGE_KEY + TIER3_ITEMS + TIER3_COMPOSITES
    return j[keep]


def main():
    ap = argparse.ArgumentParser(description="Merge Tier 1+2+3 scores.")
    ap.add_argument("--deterministic", type=Path,
                    default=PROJECT_ROOT / "output" / "score_deterministic"
                    / "scored_deterministic.csv")
    ap.add_argument("--goemo", type=Path,
                    default=PROJECT_ROOT / "output" / "score_goemo" / "scored_goemo.csv")
    ap.add_argument("--judge-dir", type=Path,
                    default=PROJECT_ROOT / "run-judge-tier3-deepseek")
    ap.add_argument("--gate", type=Path,
                    default=PROJECT_ROOT / "output" / "analyse_tier3_judge"
                    / "gate_verdicts.json")
    ap.add_argument("--output-dir", type=Path,
                    default=PROJECT_ROOT / "output" / "merge_scores")
    args = ap.parse_args()

    for p in (args.deterministic, args.goemo):
        if not p.exists():
            sys.exit(f"ERROR: input not found: {p}")

    print(f"Tier 1: {args.deterministic}")
    det = pd.read_csv(args.deterministic)
    print(f"  {len(det)} rows")

    print(f"Tier 2: {args.goemo}")
    goe = pd.read_csv(args.goemo)[MERGE_KEY + TIER2_METRICS]
    print(f"  {len(goe)} rows")

    print(f"Tier 3: {args.judge_dir}")
    judge = load_judge(args.judge_dir)
    print(f"  {len(judge)} rows total")

    merged = det.merge(goe, on=MERGE_KEY, how="outer", indicator="_m12") \
                .merge(judge, on=MERGE_KEY, how="outer", indicator="_m3")
    n_full = int(((merged["_m12"] == "both") & (merged["_m3"] == "both")).sum())
    print(f"\nMerge coverage: {n_full}/{len(merged)} rows have all three tiers")
    if n_full != len(merged):
        bad = merged[(merged["_m12"] != "both") | (merged["_m3"] != "both")]
        print(f"  WARNING: {len(bad)} incomplete rows (see run_metadata.json)")
    merged = merged.drop(columns=["_m12", "_m3"])

    metrics = TIER1_METRICS + TIER2_METRICS + TIER3_ITEMS + TIER3_COMPOSITES

    # Cell means: the 420-cell unit of analysis for all cross-tier work.
    cells = (merged.groupby(["model", "model_short", "scenario_id", "framing_id",
                             "framing_label"], as_index=False)[metrics]
             .mean())
    cells["n_runs"] = merged.groupby(
        ["model", "scenario_id", "framing_id"]).size().values

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_resp = args.output_dir / "scored_merged.csv"
    out_cell = args.output_dir / "scored_merged_cells.csv"
    merged.to_csv(out_resp, index=False)
    cells.to_csv(out_cell, index=False)
    print(f"\nWrote {out_resp} ({len(merged)} rows)")
    print(f"Wrote {out_cell} ({len(cells)} rows)")

    gate_status = None
    if args.gate.exists():
        gp = json.loads(args.gate.read_text())
        gate_status = {k: v.get("status") for k, v in gp.get("items", {}).items()}
        comp = gp.get("composite") or {}
        if comp:
            gate_status[TIER3_COMPOSITE] = comp.get("status")

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"tier1": str(args.deterministic), "tier2": str(args.goemo),
                   "tier3_dir": str(args.judge_dir)},
        "merge_key": MERGE_KEY,
        "rows_per_response": len(merged),
        "rows_cells": len(cells),
        "rows_all_three_tiers": n_full,
        "tier3_composite_items": RELATIONSHIP_ITEMS,
        "tier3_composites": {
            TIER3_COMPOSITE: {"items": RELATIONSHIP_ITEMS, "role":
                              "primary (Ruben 2026 relationship-oriented)"},
            TIER3_COMPOSITE_5: {"items": RELATIONSHIP_ITEMS_5, "role":
                                "sensitivity (item 6 removed; low human-human "
                                "reliability in this application)"},
            **{f"composite_{n}": {"items": items, "reversed": rev,
                                  "role": "Ruben 2026 secondary composite"}
               for n, (items, rev) in RUBEN_OTHER.items()},
        },
        "tier3_gate_status": gate_status or "agreement_gate.json not found at merge time",
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {args.output_dir / 'run_metadata.json'}")


if __name__ == "__main__":
    main()
