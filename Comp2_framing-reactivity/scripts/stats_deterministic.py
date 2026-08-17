#!/usr/bin/env python3
"""
Statistical analysis — Tier 1 deterministic metrics.

Reads scored_deterministic.csv and produces descriptive statistics,
per-model mixed-model ANOVAs (framing × scenario), permutation
sensitivity checks, and within-framing model comparisons.

Deferred to later (requires full 13-metric dataset):
  - Reactivity composites
  - PCA + Ward clustering

Input:  output/score_deterministic/scored_deterministic.csv
Output: output/stats_deterministic/
            stats_results.json       — full statistical output
            descriptive_results.json — aggregated means, SDs, deltas
            analysis_summary.txt     — human-readable log

Usage:
    python scripts/stats_deterministic.py
    python scripts/stats_deterministic.py --input path/to/scored.csv
    python scripts/stats_deterministic.py --skip-permutation   # faster
    python scripts/stats_deterministic.py --force               # overwrite
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

METRIC_KEYS = [
    "word_count", "format_density", "questions", "avg_sent_len",
    "coleman_liau", "first_person", "second_person", "hedging",
]

SUPPLEMENTARY_KEYS = ["fk_grade"]

FRAMING_SHORT = {
    "Angry / Frustrated": "Angry",
    "Anxious / Catastrophising": "Anxious",
    "Humor / Irony": "Humor",
    "Hyper-rational / Information-seeking": "Hyper-rational",
    "Overwhelmed / Defeated": "Overwhelmed",
    "Stoic / Minimal": "Stoic",
}


def short_model(m: str) -> str:
    """Shorten model name for table display."""
    for long, short in [
        ("Claude Sonnet 4.6", "Claude"),
        ("GPT-5.3", "GPT-5.3"),
        ("Gemini 3.1 Pro", "Gemini"),
        ("Kimi K2.5", "Kimi"),
        ("Qwen 3.5", "Qwen"),
        ("MiniMax M2.7", "MiniMax"),
        ("GLM-5", "GLM-5"),
    ]:
        if long in m:
            return short
    return m[:12]


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

_log_lines: list[str] = []


def log(msg: str):
    print(msg)
    _log_lines.append(msg)


def _sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_scored(csv_path: str) -> list[dict]:
    """Load scored_deterministic.csv, convert metric columns to float."""
    all_metrics = METRIC_KEYS + SUPPLEMENTARY_KEYS
    records = []
    with open(csv_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rec = {
                "model": row["model_short"],
                "framing": row["framing_label"],
                "scenario": row["scenario_label"],
            }
            skip = False
            for m in all_metrics:
                val = row.get(m, "")
                if val == "":
                    skip = True
                    break
                rec[m] = float(val)
            if skip:
                continue
            records.append(rec)
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# HOLM CORRECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _holm_correct(pairs: list[dict]):
    """In-place Holm step-down correction on dicts with 'p_raw' key."""
    n = len(pairs)
    sorted_idx = sorted(range(n), key=lambda k: pairs[k]["p_raw"])
    # Step 1: multiply
    for rank, idx in enumerate(sorted_idx):
        pairs[idx]["p_holm"] = min(float(pairs[idx]["p_raw"] * (n - rank)), 1.0)
    # Step 2: enforce monotonicity
    running_max = 0.0
    for idx in sorted_idx:
        running_max = max(running_max, pairs[idx]["p_holm"])
        pairs[idx]["p_holm"] = min(running_max, 1.0)
    # Round and annotate
    for p in pairs:
        p["p_raw"] = round(p["p_raw"], 6)
        p["p_holm"] = round(p["p_holm"], 6)
        p["sig"] = _sig_stars(p["p_holm"])


# ═══════════════════════════════════════════════════════════════════════════════
# MIXED-MODEL ANOVA
# ═══════════════════════════════════════════════════════════════════════════════

def mixed_anova(y_values, framing_labels, scenario_labels):
    """Two-way mixed ANOVA: framing (fixed) × scenario (random).

    F_framing = MS_framing / MS_interaction, df = (a-1, (a-1)(b-1)).
    Returns dict with omnibus F, variance components, EMMs, pairwise contrasts.
    """
    y_values = np.asarray(y_values, dtype=np.float64)
    framings = sorted(set(framing_labels))
    scenarios = sorted(set(scenario_labels))
    a, b = len(framings), len(scenarios)

    cells = defaultdict(list)
    for y, f, s in zip(y_values, framing_labels, scenario_labels):
        fi, si = framings.index(f), scenarios.index(s)
        cells[(fi, si)].append(y)

    N = len(y_values)
    grand_mean = np.mean(y_values)

    cell_means = {}
    cell_n = {}
    for (fi, si), vals in cells.items():
        cell_means[(fi, si)] = np.mean(vals)
        cell_n[(fi, si)] = len(vals)

    framing_means = np.zeros(a)
    framing_ns = np.zeros(a)
    for fi in range(a):
        vals = []
        for si in range(b):
            vals.extend(cells.get((fi, si), []))
        framing_means[fi] = np.mean(vals) if vals else 0
        framing_ns[fi] = len(vals)

    scenario_means = np.zeros(b)
    for si in range(b):
        vals = []
        for fi in range(a):
            vals.extend(cells.get((fi, si), []))
        scenario_means[si] = np.mean(vals) if vals else 0

    SS_A = sum(
        sum(cell_n.get((fi, si), 0) for si in range(b))
        * (framing_means[fi] - grand_mean) ** 2
        for fi in range(a)
    )
    SS_B = sum(
        sum(cell_n.get((fi, si), 0) for fi in range(a))
        * (scenario_means[si] - grand_mean) ** 2
        for si in range(b)
    )
    SS_AB = sum(
        cell_n.get((fi, si), 0)
        * (cell_means[(fi, si)] - framing_means[fi] - scenario_means[si] + grand_mean) ** 2
        for fi in range(a)
        for si in range(b)
        if cell_n.get((fi, si), 0) > 0
    )
    SS_E = sum(
        sum((v - cell_means[(fi, si)]) ** 2 for v in vals)
        for (fi, si), vals in cells.items()
    )

    df_A = a - 1
    df_B = b - 1
    df_AB = (a - 1) * (b - 1)
    df_E = N - a * b

    MS_A = SS_A / df_A if df_A > 0 else 0
    MS_B = SS_B / df_B if df_B > 0 else 0
    MS_AB = SS_AB / df_AB if df_AB > 0 else 0
    MS_E = SS_E / df_E if df_E > 0 else 0

    F_framing = MS_A / MS_AB if MS_AB > 1e-15 else 0.0
    p_framing = float(sp_stats.f.sf(F_framing, df_A, df_AB))

    F_scenario = MS_B / MS_AB if MS_AB > 1e-15 else 0.0
    p_scenario = float(sp_stats.f.sf(F_scenario, df_B, df_AB))

    # Variance components
    n_bar = len(cells) / sum(1.0 / n for n in cell_n.values()) if cell_n else 5.0
    var_e = max(MS_E, 0.0)
    var_ab = max((MS_AB - MS_E) / n_bar, 0.0)
    var_b = max((MS_B - MS_AB) / (a * n_bar), 0.0)
    total_var = var_b + var_ab + var_e
    icc = var_b / total_var if total_var > 0 else 0.0

    # Effect sizes
    partial_eta2 = SS_A / (SS_A + SS_AB) if (SS_A + SS_AB) > 0 else 0.0
    var_fixed = np.var(framing_means)
    denom = var_fixed + var_b + var_ab + var_e
    marginal_r2 = var_fixed / denom if denom > 0 else 0.0
    conditional_r2 = (var_fixed + var_b) / denom if denom > 0 else 0.0

    # Estimated marginal means
    emm = {}
    se_emm = np.sqrt(MS_AB / (b * n_bar)) if (b * n_bar) > 0 else 0
    for fi in range(a):
        emm[framings[fi]] = {
            "mean": round(float(framing_means[fi]), 4),
            "se": round(float(se_emm), 4),
            "n": int(framing_ns[fi]),
        }

    # Pairwise contrasts (Holm-corrected)
    se_diff = np.sqrt(2.0 * MS_AB / (b * n_bar)) if (b * n_bar) > 0 else 1e-10
    pairs = []
    for (i, fi), (j, fj) in combinations(enumerate(framings), 2):
        diff = framing_means[i] - framing_means[j]
        t_val = diff / se_diff if se_diff > 1e-15 else 0.0
        p_raw = float(2.0 * sp_stats.t.sf(abs(t_val), df_AB))
        d = diff / np.sqrt(MS_E) if MS_E > 1e-15 else 0.0
        pairs.append({
            "a": fi,
            "b": fj,
            "diff": round(float(diff), 4),
            "se": round(float(se_diff), 4),
            "t": round(float(t_val), 3),
            "df": int(df_AB),
            "p_raw": float(p_raw),
            "cohens_d": round(float(d), 3),
        })
    _holm_correct(pairs)

    return {
        "omnibus": {
            "F": round(float(F_framing), 3),
            "df1": int(df_A),
            "df2": int(df_AB),
            "p_value": float(p_framing),
            "MS_framing": round(float(MS_A), 4),
            "MS_interaction": round(float(MS_AB), 4),
            "MS_error": round(float(MS_E), 4),
        },
        "scenario_effect": {
            "F": round(float(F_scenario), 3),
            "p_value": float(p_scenario),
        },
        "variance_components": {
            "scenario": round(float(var_b), 4),
            "framing_x_scenario": round(float(var_ab), 4),
            "residual": round(float(var_e), 4),
            "icc_scenario": round(float(icc), 4),
        },
        "effect_sizes": {
            "partial_eta2": round(float(partial_eta2), 4),
            "marginal_r2": round(float(marginal_r2), 4),
            "conditional_r2": round(float(conditional_r2), 4),
        },
        "estimated_marginal_means": emm,
        "pairwise": [
            {
                "a": p["a"],
                "b": p["b"],
                "diff": p["diff"],
                "se": p["se"],
                "t": p["t"],
                "df": p["df"],
                "p_raw": p["p_raw"],
                "p_holm": p["p_holm"],
                "cohens_d": p["cohens_d"],
                "sig": p["sig"],
            }
            for p in pairs
        ],
        "n_significant_pairs": sum(1 for p in pairs if p["p_holm"] < 0.05),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FAST MIXED F (speed-critical for permutation inner loop)
# ═══════════════════════════════════════════════════════════════════════════════

def fast_mixed_F(y, fi_arr, bi_arr, a, b):
    """Compute F = MS_A / MS_AB for permutation inner loop."""
    grand_mean = y.mean()

    factor_means = np.array([y[fi_arr == i].mean() for i in range(a)])
    block_means = np.array([y[bi_arr == i].mean() for i in range(b)])
    factor_ns = np.array([(fi_arr == i).sum() for i in range(a)])

    SS_A = np.sum(factor_ns * (factor_means - grand_mean) ** 2)

    SS_AB = 0.0
    for i in range(a):
        for j in range(b):
            mask = (fi_arr == i) & (bi_arr == j)
            n_ij = mask.sum()
            if n_ij > 0:
                SS_AB += n_ij * (
                    y[mask].mean() - factor_means[i] - block_means[j] + grand_mean
                ) ** 2

    df_A = a - 1
    df_AB = (a - 1) * (b - 1)
    MS_A = SS_A / df_A if df_A > 0 else 0
    MS_AB = SS_AB / df_AB if df_AB > 0 else 0
    return MS_A / MS_AB if MS_AB > 1e-15 else 0.0


def pairwise_from_mixed(y_arr, factor_labels, block_labels):
    """Pairwise contrasts with Holm correction from a mixed ANOVA."""
    factors = sorted(set(factor_labels))
    blocks = sorted(set(block_labels))
    a, b = len(factors), len(blocks)
    fi_map = {f: i for i, f in enumerate(factors)}
    bi_map = {s: i for i, s in enumerate(blocks)}
    fi_arr = np.array([fi_map[f] for f in factor_labels])
    bi_arr = np.array([bi_map[s] for s in block_labels])

    y_arr = np.asarray(y_arr, dtype=np.float64)
    grand_mean = y_arr.mean()
    factor_means = np.array([y_arr[fi_arr == i].mean() for i in range(a)])
    block_means = np.array([y_arr[bi_arr == i].mean() for i in range(b)])

    # MS_AB
    SS_AB = 0.0
    cell_counts = np.zeros((a, b))
    for i in range(a):
        for j in range(b):
            mask = (fi_arr == i) & (bi_arr == j)
            n_ij = mask.sum()
            cell_counts[i, j] = n_ij
            if n_ij > 0:
                SS_AB += n_ij * (
                    y_arr[mask].mean() - factor_means[i] - block_means[j] + grand_mean
                ) ** 2

    df_AB = (a - 1) * (b - 1)
    MS_AB = SS_AB / df_AB if df_AB > 0 else 0

    # MS_E for Cohen's d
    SS_E = 0.0
    for i in range(a):
        for j in range(b):
            mask = (fi_arr == i) & (bi_arr == j)
            if mask.sum() > 0:
                SS_E += np.sum((y_arr[mask] - y_arr[mask].mean()) ** 2)
    df_E = len(y_arr) - a * b
    MS_E = SS_E / df_E if df_E > 0 else 1.0

    n_cells = np.sum(cell_counts > 0)
    n_bar = n_cells / np.sum(1.0 / np.maximum(cell_counts[cell_counts > 0], 1))
    se_diff = np.sqrt(2.0 * MS_AB / (b * n_bar)) if (b * n_bar) > 0 else 1e-10

    pairs = []
    for (i, fi), (j, fj) in combinations(enumerate(factors), 2):
        diff = factor_means[i] - factor_means[j]
        t_val = diff / se_diff if se_diff > 1e-15 else 0.0
        p_raw = float(2.0 * sp_stats.t.sf(abs(t_val), df_AB))
        d = diff / np.sqrt(MS_E) if MS_E > 1e-15 else 0.0
        pairs.append({
            "a": fi,
            "b": fj,
            "diff": round(float(diff), 4),
            "se": round(float(se_diff), 4),
            "t": round(float(t_val), 3),
            "df": int(df_AB),
            "p_raw": float(p_raw),
            "cohens_d": round(float(d), 3),
        })

    _holm_correct(pairs)
    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Statistical analysis of Tier 1 deterministic metrics"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to scored_deterministic.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/stats_deterministic"),
        help="Output directory (default: output/stats_deterministic)",
    )
    parser.add_argument(
        "--skip-permutation",
        action="store_true",
        help="Skip permutation tests (faster)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files",
    )
    args = parser.parse_args()

    input_path = args.input or "output/score_deterministic/scored_deterministic.csv"
    out_dir = args.output_dir

    # Pre-check outputs
    output_files = [
        out_dir / "stats_results.json",
        out_dir / "descriptive_results.json",
        out_dir / "analysis_summary.txt",
    ]
    if not args.force:
        for p in output_files:
            if p.exists():
                print(f"ERROR: {p} exists. Use --force to overwrite.", file=sys.stderr)
                sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    # 1. LOAD DATA
    # ══════════════════════════════════════════════════════════════════════════

    log(f"Loading scored data from {input_path}...")
    records = load_scored(input_path)
    if not records:
        print(f"ERROR: No valid records from {input_path}", file=sys.stderr)
        sys.exit(1)

    MODELS = sorted(set(r["model"] for r in records))
    FRAMINGS = sorted(set(r["framing"] for r in records))
    SCENARIOS = sorted(set(r["scenario"] for r in records))

    log(
        f"  {len(records)} responses  |  {len(MODELS)} models  |  "
        f"{len(FRAMINGS)} framings  |  {len(SCENARIOS)} scenarios\n"
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 2. DESCRIPTIVE STATISTICS
    # ══════════════════════════════════════════════════════════════════════════

    log(f"{'=' * 76}")
    log("PART 1: DESCRIPTIVE STATISTICS")
    log(f"{'=' * 76}\n")

    agg = defaultdict(lambda: defaultdict(list))
    for r in records:
        key = (r["model"], r["framing"])
        for m in METRIC_KEYS + SUPPLEMENTARY_KEYS:
            agg[key][m].append(r[m])

    aggregated = []
    for (model, framing), metrics in sorted(agg.items()):
        entry = {
            "model": model,
            "framing": framing,
            "framing_short": FRAMING_SHORT.get(framing, framing),
            "n": len(metrics[METRIC_KEYS[0]]),
        }
        for m, vals in metrics.items():
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)
            entry[m + "_mean_raw"] = mean
            entry[m + "_mean"] = round(mean, 2)
            entry[m + "_sd"] = round(var ** 0.5, 2)
        aggregated.append(entry)

    # Baselines (grand mean per model across framings)
    baselines_raw = defaultdict(lambda: defaultdict(list))
    for e in aggregated:
        for k in e:
            if k.endswith("_mean_raw"):
                baselines_raw[e["model"]][k].append(e[k])

    baselines_unrounded = {}
    baselines_json = {}
    for model, metrics in baselines_raw.items():
        baselines_unrounded[model] = {k: sum(v) / len(v) for k, v in metrics.items()}
        baselines_json[model] = {
            k.replace("_mean_raw", "_mean"): round(sum(v) / len(v), 2)
            for k, v in metrics.items()
        }

    # Deltas from baseline
    for e in aggregated:
        bl = baselines_unrounded[e["model"]]
        for k in list(e.keys()):
            if k.endswith("_mean_raw"):
                base = k.replace("_mean_raw", "")
                e[base + "_delta"] = round(e[k] - bl[k], 2)

    # Print summary table
    for model in MODELS:
        log(f"  {model}")
        model_entries = [e for e in aggregated if e["model"] == model]
        header = f"    {'Framing':<15}"
        for m in METRIC_KEYS:
            header += f"  {m:>14}"
        log(header)
        log(f"    {'─' * (15 + 16 * len(METRIC_KEYS))}")
        for e in sorted(model_entries, key=lambda x: x["framing"]):
            row = f"    {e['framing_short']:<15}"
            for m in METRIC_KEYS:
                row += f"  {e[m + '_mean']:10.2f}±{e[m + '_sd']:<3.1f}"
            log(row)
        log("")

    # Clean aggregated for JSON (remove _mean_raw helper columns)
    for e in aggregated:
        for k in list(e.keys()):
            if k.endswith("_mean_raw"):
                del e[k]

    desc_output = {
        "aggregated": aggregated,
        "baselines": baselines_json,
    }

    # ══════════════════════════════════════════════════════════════════════════
    # 3. PER-MODEL MIXED-MODEL ANOVA
    # ══════════════════════════════════════════════════════════════════════════

    log(f"\n{'=' * 76}")
    log("PART 2: MIXED-MODEL ANOVA (per model)")
    log("  metric ~ framing + (1|scenario) + framing:scenario")
    log("  F = MS_framing / MS_interaction  (scenario = random)")
    log(f"{'=' * 76}\n")

    anova_results = {}

    for model in MODELS:
        log(f"  MODEL: {model}")
        log(f"  {'─' * 70}")
        mrecs = [r for r in records if r["model"] == model]
        fl = [r["framing"] for r in mrecs]
        sl = [r["scenario"] for r in mrecs]

        model_res = {}
        for metric in METRIC_KEYS:
            y = [r[metric] for r in mrecs]
            res = mixed_anova(y, fl, sl)
            model_res[metric] = res

            p = res["omnibus"]["p_value"]
            log(
                f"    {metric:18s}  "
                f"F({res['omnibus']['df1']},{res['omnibus']['df2']})="
                f"{res['omnibus']['F']:7.2f}  "
                f"p={p:.2e} {_sig_stars(p):3s}  "
                f"eta2p={res['effect_sizes']['partial_eta2']:.3f}  "
                f"ICC={res['variance_components']['icc_scenario']:.3f}  "
                f"sig.pairs={res['n_significant_pairs']:2d}/15"
            )

        anova_results[model] = model_res
        log("")

    # ── Cross-model summary tables ───────────────────────────────────────────

    header = f"{'Metric':18s}" + "".join(f"  {short_model(m):>10s}" for m in MODELS)
    sep = "─" * len(header)

    log(f"\n{'=' * 76}")
    log("SUMMARY: OMNIBUS p-VALUES")
    log(f"{'=' * 76}")
    log(header)
    log(sep)
    for metric in METRIC_KEYS:
        row = f"{metric:18s}"
        for model in MODELS:
            p = anova_results[model][metric]["omnibus"]["p_value"]
            if p < 0.001:
                row += f"  {'<.001***':>10s}"
            elif p < 0.01:
                row += f"  {f'{p:.3f} **':>10s}"
            elif p < 0.05:
                row += f"  {f'{p:.3f}  *':>10s}"
            else:
                row += f"  {f'{p:.3f} ns':>10s}"
        log(row)

    log(f"\n{'=' * 76}")
    log("SUMMARY: PARTIAL eta-squared")
    log(f"{'=' * 76}")
    log(header)
    log(sep)
    for metric in METRIC_KEYS:
        row = f"{metric:18s}"
        for model in MODELS:
            eta2 = anova_results[model][metric]["effect_sizes"]["partial_eta2"]
            row += f"  {eta2:10.3f}"
        log(row)

    log(f"\n{'=' * 76}")
    log("SUMMARY: ICC (scenario)")
    log(f"{'=' * 76}")
    log(header)
    log(sep)
    for metric in METRIC_KEYS:
        row = f"{metric:18s}"
        for model in MODELS:
            icc = anova_results[model][metric]["variance_components"]["icc_scenario"]
            row += f"  {icc:10.3f}"
        log(row)

    # ══════════════════════════════════════════════════════════════════════════
    # 4. PERMUTATION F-TESTS
    # ══════════════════════════════════════════════════════════════════════════

    N_PERM = 2000
    perm_results = {}

    if not args.skip_permutation:
        log(f"\n\n{'=' * 76}")
        log(f"PART 3: PERMUTATION F-TESTS ({N_PERM} iterations)")
        log("  Restricted: framing labels shuffled within scenario blocks")
        log(f"{'=' * 76}\n")

        rng = np.random.default_rng(seed=2024)

        for model in MODELS:
            log(f"  {model}")
            mrecs = [r for r in records if r["model"] == model]
            fl = [r["framing"] for r in mrecs]
            sl = [r["scenario"] for r in mrecs]

            framings_u = sorted(set(fl))
            scenarios_u = sorted(set(sl))
            a, b = len(framings_u), len(scenarios_u)
            fi_map = {f: i for i, f in enumerate(framings_u)}
            bi_map = {s: i for i, s in enumerate(scenarios_u)}
            fi_arr = np.array([fi_map[f] for f in fl], dtype=np.int32)
            bi_arr = np.array([bi_map[s] for s in sl], dtype=np.int32)
            block_idx = [np.where(bi_arr == j)[0] for j in range(b)]

            perm_results[model] = {}

            for metric in METRIC_KEYS:
                y = np.array([r[metric] for r in mrecs], dtype=np.float64)
                F_obs = fast_mixed_F(y, fi_arr, bi_arr, a, b)

                fi_perm = fi_arr.copy()
                n_exceed = 0
                for _ in range(N_PERM):
                    for idx in block_idx:
                        block = fi_perm[idx].copy()
                        rng.shuffle(block)
                        fi_perm[idx] = block
                    if fast_mixed_F(y, fi_perm, bi_arr, a, b) >= F_obs:
                        n_exceed += 1

                p_perm = (n_exceed + 1) / (N_PERM + 1)
                p_param = float(sp_stats.f.sf(F_obs, a - 1, (a - 1) * (b - 1)))
                agree = (p_perm < 0.05) == (p_param < 0.05)

                perm_results[model][metric] = {
                    "F_observed": round(float(F_obs), 3),
                    "p_parametric": float(p_param),
                    "p_permutation": round(float(p_perm), 4),
                    "agree_at_05": agree,
                }

                match = "Y" if agree else "N"
                log(
                    f"    {metric:18s}  F={F_obs:7.2f}  "
                    f"p_param={p_param:.2e}  p_perm={p_perm:.4f}  agree={match}"
                )

            n_agree = sum(
                1 for m in METRIC_KEYS if perm_results[model][m]["agree_at_05"]
            )
            log(f"    -> {n_agree}/{len(METRIC_KEYS)} agree\n")
    else:
        log(f"\n  [Permutation tests skipped (--skip-permutation)]")

    # ══════════════════════════════════════════════════════════════════════════
    # 5. WITHIN-FRAMING MODEL COMPARISONS
    # ══════════════════════════════════════════════════════════════════════════

    log(f"\n\n{'=' * 76}")
    log("PART 4: WITHIN-FRAMING MODEL COMPARISONS")
    log("  metric ~ model + (1|scenario)")
    log(f"{'=' * 76}\n")

    within_framing = {}

    for framing in FRAMINGS:
        fshort = FRAMING_SHORT.get(framing, framing)
        log(f"  Framing: {fshort}")
        frecs = [r for r in records if r["framing"] == framing]
        ml = [r["model"] for r in frecs]
        sl = [r["scenario"] for r in frecs]

        within_framing[framing] = {}

        for metric in METRIC_KEYS:
            y = np.array([r[metric] for r in frecs], dtype=np.float64)

            models_u = sorted(set(ml))
            scenarios_u = sorted(set(sl))
            a, b = len(models_u), len(scenarios_u)
            fi_map = {f: i for i, f in enumerate(models_u)}
            bi_map = {s: i for i, s in enumerate(scenarios_u)}
            fi_arr = np.array([fi_map[f] for f in ml], dtype=np.int32)
            bi_arr = np.array([bi_map[s] for s in sl], dtype=np.int32)

            F_val = fast_mixed_F(y, fi_arr, bi_arr, a, b)
            df1, df2 = a - 1, (a - 1) * (b - 1)
            p_val = float(sp_stats.f.sf(F_val, df1, df2))

            # Partial eta-squared (recompute properly)
            grand_mean = y.mean()
            factor_means = np.array([y[fi_arr == i].mean() for i in range(a)])
            factor_ns = np.array([(fi_arr == i).sum() for i in range(a)])
            block_means = np.array([y[bi_arr == i].mean() for i in range(b)])
            SS_A_real = np.sum(factor_ns * (factor_means - grand_mean) ** 2)
            SS_AB_real = 0.0
            for i in range(a):
                for j in range(b):
                    mask = (fi_arr == i) & (bi_arr == j)
                    n_ij = mask.sum()
                    if n_ij > 0:
                        SS_AB_real += n_ij * (
                            y[mask].mean()
                            - factor_means[i]
                            - block_means[j]
                            + grand_mean
                        ) ** 2
            eta2 = (
                SS_A_real / (SS_A_real + SS_AB_real)
                if (SS_A_real + SS_AB_real) > 0
                else 0
            )

            pairs = pairwise_from_mixed(y, ml, sl)
            n_sig = sum(1 for p in pairs if p["p_holm"] < 0.05)

            model_means = {}
            for m in MODELS:
                vals = [r[metric] for r in frecs if r["model"] == m]
                model_means[m] = round(float(np.mean(vals)), 4) if vals else 0

            n_model_pairs = len(list(combinations(MODELS, 2)))

            within_framing[framing][metric] = {
                "omnibus": {
                    "F": round(float(F_val), 3),
                    "df1": int(df1),
                    "df2": int(df2),
                    "p_value": float(p_val),
                    "partial_eta2": round(float(eta2), 4),
                },
                "model_means": model_means,
                "pairwise": pairs,
                "n_sig_pairs": n_sig,
            }

            log(
                f"    {metric:18s}  F({df1},{df2})={F_val:7.2f}  "
                f"p={p_val:.2e} {_sig_stars(p_val):3s}  "
                f"eta2p={eta2:.3f}  sig.pairs={n_sig:2d}/{n_model_pairs}"
            )
        log("")

    # ══════════════════════════════════════════════════════════════════════════
    # 6. SAVE
    # ══════════════════════════════════════════════════════════════════════════

    desc_path = out_dir / "descriptive_results.json"
    with open(desc_path, "w") as f:
        json.dump(desc_output, f, indent=2)

    stats_output = {
        "anova": anova_results,
        "permutation_tests": perm_results,
        "within_framing": {
            framing: {metric: within_framing[framing][metric] for metric in METRIC_KEYS}
            for framing in FRAMINGS
        },
    }
    stats_path = out_dir / "stats_results.json"
    with open(stats_path, "w") as f:
        json.dump(stats_output, f, indent=2)

    txt_path = out_dir / "analysis_summary.txt"
    with open(txt_path, "w") as f:
        f.write("\n".join(_log_lines))

    log(f"\n{'=' * 76}")
    log(f"Descriptive results -> {desc_path}")
    log(f"Statistical results -> {stats_path}")
    log(f"Summary text        -> {txt_path}")


if __name__ == "__main__":
    main()
