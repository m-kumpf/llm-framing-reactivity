#!/usr/bin/env python3
"""
Component-1 -> Component-2 bridge (exploratory, hypothesis-generating).

Pre-specified relation: Comp1 Emotional Stability
(IPIP-50 factor score) vs. the confirmatory composite reactivity index
(mean standardized |delta| across the 8 confirmatory individual Tier-3
items, composite excluded so no item is counted twice; from analyse.py). n = 7 models, Spearman, exploratory — no formal inference is
possible at n = 7; p-values are exact permutation values (all 5,040
orderings) and reported for transparency only.

Supplementary: the full 5 factors x 3 reactivity indices Spearman grid
(confirmatory bridge index, interpersonal primary, deterministic-only
sensitivity), all labelled exploratory.

Input:  ../Comp1_IPIP-50/output/analyse-results/summary_statistics.csv
        output/analysis/analysis_results.json      (from analyse.py)
Output: output/bridge1-2/
            bridge_results.json, bridge_correlations.csv
            analysis_summary.txt, run_metadata.json

Usage:
    python bridge1_2.py
    python bridge1_2.py --force
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

BASE = Path(__file__).resolve().parents[1]                      # Comp2_framing-reactivity/
COMP1_STATS = BASE.parent / "Comp1_IPIP-50" / "output" / "analyse-results" / "summary_statistics.csv"
ANALYSIS_JSON = BASE / "output" / "analysis" / "analysis_results.json"
DEFAULT_OUTDIR = BASE / "output" / "bridge1-2"


def _rel(p):
    """Path as recorded in metadata: relative to the component root."""
    return os.path.relpath(p, BASE)

MODEL_SHORT = {
    "anthropic/claude-sonnet-4.6": "Claude Sonnet 4.6",
    "openai/gpt-5.3-chat": "GPT-5.3",
    "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "moonshotai/kimi-k2.5": "Kimi K2.5",
    "qwen/qwen3.5-397b-a17b": "Qwen 3.5",
    "minimax/minimax-m2.7": "MiniMax M2.7",
    "z-ai/glm-5": "GLM 5",
}

# fixed model order for outputs (matches analyse_tier3.py)
MODEL_ORDER = [
    "Claude Sonnet 4.6", "GPT-5.3", "Gemini 3.1 Pro", "Kimi K2.5",
    "Qwen 3.5", "MiniMax M2.7", "GLM 5",
]

PRIMARY_FACTOR = "Emotional_Stability"
PRIMARY_INDEX = "reactivity_confirmatory_bridge_index"

FACTORS = ["Extraversion", "Agreeableness", "Conscientiousness",
           "Emotional_Stability", "Intellect_Openness"]
INDICES = {
    "reactivity_confirmatory_bridge_index": "Confirmatory bridge index (8 confirmatory Tier-3 items, non-overlapping)",
    "reactivity_primary": "Interpersonal primary (10 metrics)",
    "reactivity_sensitivity": "Deterministic-only sensitivity (7 metrics)",
}


def spearman_exact(x, y):
    """Spearman rho with exact two-sided permutation p (n <= 8)."""
    rho, _ = sp_stats.spearmanr(x, y)
    n = len(x)
    count = 0
    total = 0
    y = np.asarray(y)
    for perm in permutations(range(n)):
        r, _ = sp_stats.spearmanr(x, y[list(perm)])
        if abs(r) >= abs(rho) - 1e-12:
            count += 1
        total += 1
    return float(rho), count / total


def load_traits():
    traits = {}
    with open(COMP1_STATS, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            m = MODEL_SHORT.get(row["model"])
            if m is None:
                sys.exit(f"ERROR: unmapped Comp1 model id: {row['model']}")
            traits.setdefault(m, {})[row["factor"]] = float(row["mean"])
    assert set(traits) == set(MODEL_ORDER), "Comp1/Comp2 model sets differ"
    for m, f in traits.items():
        assert set(f) == set(FACTORS), f"{m}: missing factors"
    return traits


def load_reactivity():
    with open(ANALYSIS_JSON, encoding="utf-8") as fh:
        data = json.load(fh)
    out = {}
    for key in INDICES:
        idx = data.get(key)
        if idx is None:
            sys.exit(f"ERROR: {key} not in {ANALYSIS_JSON} — rerun analyse.py")
        assert set(idx) == set(MODEL_ORDER)
        out[key] = idx
    return out


def main():
    ap = argparse.ArgumentParser(description="Comp1 -> Comp2 exploratory bridge")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    outdir = args.output_dir
    if outdir.exists() and any(outdir.iterdir()) and not args.force:
        sys.exit(f"Output dir {outdir} is not empty — use --force to overwrite.")
    outdir.mkdir(parents=True, exist_ok=True)

    traits = load_traits()
    reactivity = load_reactivity()

    # Full exploratory grid
    rows = []
    for factor in FACTORS:
        x = [traits[m][factor] for m in MODEL_ORDER]
        for key, label in INDICES.items():
            y = [reactivity[key][m] for m in MODEL_ORDER]
            rho, p = spearman_exact(x, y)
            r_pear = float(np.corrcoef(x, y)[0, 1])
            rows.append({"factor": factor, "reactivity_index": key,
                         "index_label": label, "n": len(MODEL_ORDER),
                         "spearman_rho": round(rho, 4),
                         "p_exact_permutation": round(p, 4),
                         "pearson_r_descriptive": round(r_pear, 4),
                         "primary": factor == PRIMARY_FACTOR and key == PRIMARY_INDEX})

    primary = next(r for r in rows if r["primary"])

    with open(outdir / "bridge_correlations.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    results = {
        "generated_utc": now,
        "prespecified": {
            "x": f"Comp1 {PRIMARY_FACTOR} (IPIP-50 factor mean)",
            "y": PRIMARY_INDEX,
            "spearman_rho": primary["spearman_rho"],
            "p_exact_permutation": primary["p_exact_permutation"],
            "pearson_r_descriptive": primary["pearson_r_descriptive"],
            "n_models": 7,
            "status": "exploratory / hypothesis-generating",
        },
        "data": {m: {"traits": traits[m],
                     **{k: reactivity[k][m] for k in INDICES}}
                 for m in MODEL_ORDER},
        "exploratory_grid": rows,
    }
    with open(outdir / "bridge_results.json", "w") as fh:
        json.dump(results, fh, indent=2)

    lines = ["Comp1 -> Comp2 bridge — exploratory (n = 7 models)",
             "=" * 64,
             f"Generated (UTC): {now}",
             "",
             f"PRE-SPECIFIED: {PRIMARY_FACTOR} vs confirmatory bridge index",
             f"  Spearman rho = {primary['spearman_rho']:+.3f}   "
             f"exact permutation p = {primary['p_exact_permutation']:.4f}   "
             f"(Pearson r = {primary['pearson_r_descriptive']:+.3f}, descriptive)",
             "",
             f"{'factor':<22}{'index':<42}{'rho':>7}{'p_exact':>9}"]
    lines.append("-" * 82)
    for r in rows:
        mark = "  <- pre-specified" if r["primary"] else ""
        lines.append(f"{r['factor']:<22}{r['index_label']:<42}"
                     f"{r['spearman_rho']:>+7.3f}{r['p_exact_permutation']:>9.4f}{mark}")
    lines += ["", "All correlations exploratory; exact two-sided permutation "
              "p over all 5,040 orderings; no correction applied, no "
              "confirmatory claim intended."]
    (outdir / "analysis_summary.txt").write_text("\n".join(lines) + "\n")

    meta = {"script": "bridge1_2.py", "timestamp_utc": now,
            "sources": {"comp1": _rel(COMP1_STATS), "comp2": _rel(ANALYSIS_JSON)},
            "n_models": 7,
            "outputs": ["bridge_results.json", "bridge_correlations.csv",
                        "analysis_summary.txt"]}
    with open(outdir / "run_metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    print("\n".join(lines))
    print(f"\nOutputs written to {outdir}")


if __name__ == "__main__":
    main()
