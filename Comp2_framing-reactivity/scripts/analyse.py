#!/usr/bin/env python3
"""
Integrated cross-tier analysis — descriptive stats, ANOVA, permutation tests,
within-framing model comparisons, convergent validity across tiers,
PCA + Ward clustering, and the composite reactivity index for the Comp1 bridge.

Tier-3 metric membership is resolved at runtime from the agreement gate
(output/analyse_tier3_judge/gate_verdicts.json): gate-confirmed items plus
the relationship-oriented composite (finalised 2026-07-15).

Input:  output/merge_scores/scored_merged.csv  (from merge_scores.py)
Output: output/analysis/analysis_results.json
        output/analysis/stats_results.json
        output/analysis/analysis_summary.txt

Usage:
    python analyse.py
    python analyse.py --input path/to/scored_merged.csv
    python analyse.py --skip-permutation
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np
from scipy import stats as sp_stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "output" / "analysis"


def _rel(p):
    """Path as recorded in metadata: relative to the component root."""
    return os.path.relpath(p, PROJECT_ROOT)


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

# Tier 1 (8 main deterministic) + Tier 2 (2 GoEmotions) + Tier 3 (gate-confirmed).
#
# Tier-3 membership is taken from the agreement-gate output, never hard-coded
# (decision 2026-07-15: gate-confirmed single items + relationship-oriented
# composite; saturated and exploratory items are excluded from the
# cross-tier metric set).

_GATE_PATH = PROJECT_ROOT / "output" / "analyse_tier3_judge" / "gate_verdicts.json"
if not _GATE_PATH.exists():
    sys.exit(f"ERROR: gate verdicts not found: {_GATE_PATH} — "
             "run analyse_tier3_judge.py first")
with open(_GATE_PATH, encoding='utf-8') as _fh:
    _GATE = json.load(_fh)

TIER3_COMPOSITE = 'relationship_oriented'  # column name from merge_scores.py
TIER3_COMPOSITE_5ITEM = 'relationship_oriented_5item'  # sensitivity (item 6 out)
TIER3_CONFIRMED_ITEMS = [k for k, v in _GATE['items'].items()
                         if v['status'] == 'confirmatory']
TIER3_METRICS = TIER3_CONFIRMED_ITEMS + (
    [TIER3_COMPOSITE] if _GATE['composite']['status'] == 'confirmatory' else [])

# Descriptives / per-model ANOVA / within-framing run over everything,
# including both composites (each analysed independently — no summation).
METRIC_KEYS = [
    'word_count', 'format_density', 'questions', 'avg_sent_len', 'coleman_liau',
    'first_person', 'second_person', 'hedging', 'pos_emotion', 'neg_emotion',
] + TIER3_METRICS + [TIER3_COMPOSITE_5ITEM]

# Feature matrix for PCA / Ward: composites EXCLUDED (double-counting fix
# 2026-07-15 — four composite items are already individual features).
PCA_METRICS = [k for k in METRIC_KEYS
               if k not in (TIER3_COMPOSITE, TIER3_COMPOSITE_5ITEM)]

# Interpersonal subset (dual PCA + primary reactivity profile): the four Tier-1
# interpersonal metrics, both emotion scores, and the confirmed relational
# Tier-3 items. The composite is excluded here too (it would double-count
# its own members inside a summed/combined set).
_RELATIONAL_ITEMS = [k for k in TIER3_CONFIRMED_ITEMS
                     if int(k.split('_')[1]) in
                     _GATE['config']['composite_items']]
INTERPERSONAL_METRICS = [
    'questions', 'first_person', 'second_person', 'hedging',
    'pos_emotion', 'neg_emotion',
] + _RELATIONAL_ITEMS

# Composite reactivity index for the Comp1 trait bridge: the confirmatory
# INDIVIDUAL Tier-3 items only (revision 2026-07-15b — the composite is
# dropped so no item is counted twice, consistent with the Fig 4 / PCA
# de-duplication).
REACTIVITY_CONFIRMATORY = TIER3_CONFIRMED_ITEMS

# Primary descriptive reactivity (interpersonal profile) + sensitivity variant
# with no LLM in the measurement loop (deterministic + classifier only).
REACTIVITY_PRIMARY = INTERPERSONAL_METRICS
REACTIVITY_SENSITIVITY = [
    'hedging', 'questions', 'first_person', 'second_person',
    'pos_emotion', 'neg_emotion', 'format_density',
]

# Convergent-validity pairs: Tier-3 relationship-oriented composite vs its
# no-LLM-in-the-loop shadows (computed on the 42 model x framing cells).
# The 5-item composite (item 6 removed) runs as a sensitivity target.
CONVERGENCE_TARGET = TIER3_COMPOSITE
CONVERGENCE_TARGETS = [
    (TIER3_COMPOSITE, 'Ruben 6-item composite (primary)'),
    (TIER3_COMPOSITE_5ITEM, '5-item sensitivity (item 6 removed)'),
]
CONVERGENCE_SHADOWS = ['pos_emotion', 'second_person', 'questions',
                       'first_person', 'neg_emotion']


# Model display names
_MODEL_NAMES = {
    'claude':  'Claude Sonnet 4.6',
    'gpt':     'GPT-5.3',
    'gemini':  'Gemini 3.1 Pro',
    'kimi':    'Kimi K2.5',
    'qwen':    'Qwen 3.5',
    'minimax': 'MiniMax M2.7',
    'glm':     'GLM 5',
}

FRAMING_SHORT = {
    "Angry / Frustrated": "Angry",
    "Anxious / Catastrophising": "Anxious",
    "Humor / Irony": "Humor",
    "Hyper-rational / Information-seeking": "Hyper-rational",
    "Overwhelmed / Defeated": "Overwhelmed",
    "Stoic / Minimal": "Stoic",
}

# ── Logging ──────────────────────────────────────────────────────────────────

_log_lines = []

def log(msg: str):
    print(msg)
    _log_lines.append(msg)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def model_short_name(raw: str) -> str:
    raw_lower = raw.lower()
    for key, name in _MODEL_NAMES.items():
        if key in raw_lower:
            return name
    return raw


def load_scored(csv_path: str) -> list[dict]:
    """Load scored_merged.csv, convert metric columns to float."""
    records = []
    with open(csv_path, encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            rec = {
                # Canonical display name from the OpenRouter id — the CSV's
                # model_short writes "GLM-5" where this script uses "GLM 5".
                'model': model_short_name(row.get('model', '')
                                          or row.get('model_short', '')),
                'framing': row.get('framing_label', ''),
                'scenario': row.get('scenario_label', ''),
            }
            # Parse metric values
            skip = False
            for m in METRIC_KEYS:
                val = row.get(m, '')
                if val == '':
                    skip = True
                    break
                rec[m] = float(val)
            if skip:
                continue
            records.append(rec)
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# MIXED-MODEL ANOVA (full version — variance components, EMMs, effect sizes)
# ═══════════════════════════════════════════════════════════════════════════════

def mixed_anova(y_values, framing_labels, scenario_labels):
    """Two-way mixed ANOVA: framing (fixed) x scenario (random).

    F_framing = MS_framing / MS_interaction.
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

    SS_A = sum(sum(cell_n.get((fi, si), 0) for si in range(b))
               * (framing_means[fi] - grand_mean)**2 for fi in range(a))
    SS_B = sum(sum(cell_n.get((fi, si), 0) for fi in range(a))
               * (scenario_means[si] - grand_mean)**2 for si in range(b))
    SS_AB = sum(cell_n.get((fi, si), 0)
                * (cell_means[(fi, si)] - framing_means[fi]
                   - scenario_means[si] + grand_mean)**2
                for fi in range(a) for si in range(b)
                if cell_n.get((fi, si), 0) > 0)
    SS_E = sum(sum((v - cell_means[(fi, si)])**2 for v in vals)
               for (fi, si), vals in cells.items())

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

    n_bar = len(cells) / sum(1.0 / n for n in cell_n.values()) if cell_n else 5.0
    var_e = max(MS_E, 0.0)
    var_ab = max((MS_AB - MS_E) / n_bar, 0.0)
    var_b = max((MS_B - MS_AB) / (a * n_bar), 0.0)
    total_var = var_b + var_ab + var_e
    icc = var_b / total_var if total_var > 0 else 0.0

    partial_eta2 = SS_A / (SS_A + SS_AB) if (SS_A + SS_AB) > 0 else 0.0
    var_fixed = np.var(framing_means)
    denom = var_fixed + var_b + var_ab + var_e
    marginal_r2 = var_fixed / denom if denom > 0 else 0.0
    conditional_r2 = (var_fixed + var_b) / denom if denom > 0 else 0.0

    # EMMs
    emm = {}
    se_emm = np.sqrt(MS_AB / (b * n_bar)) if (b * n_bar) > 0 else 0
    for fi in range(a):
        emm[framings[fi]] = {
            'mean': round(float(framing_means[fi]), 4),
            'se': round(float(se_emm), 4),
            'n': int(framing_ns[fi]),
        }

    # Pairwise contrasts
    se_diff = np.sqrt(2.0 * MS_AB / (b * n_bar)) if (b * n_bar) > 0 else 1e-10
    pairs = []
    for (i, fi), (j, fj) in combinations(enumerate(framings), 2):
        diff = framing_means[i] - framing_means[j]
        t_val = diff / se_diff if se_diff > 1e-15 else 0.0
        p_raw = float(2.0 * sp_stats.t.sf(abs(t_val), df_AB))
        d = diff / np.sqrt(MS_E) if MS_E > 1e-15 else 0.0
        pairs.append({
            'a': fi, 'b': fj,
            'diff': float(diff), 'se': float(se_diff),
            't': float(t_val), 'df': int(df_AB),
            'p_raw': float(p_raw), 'cohens_d': float(d),
        })

    # Holm correction
    _holm_correct(pairs)

    return {
        'omnibus': {
            'F': round(float(F_framing), 3), 'df1': int(df_A), 'df2': int(df_AB),
            'p_value': float(p_framing),
            'MS_framing': round(float(MS_A), 4),
            'MS_interaction': round(float(MS_AB), 4),
            'MS_error': round(float(MS_E), 4),
        },
        'scenario_effect': {
            'F': round(float(F_scenario), 3), 'p_value': float(p_scenario),
        },
        'variance_components': {
            'scenario': round(float(var_b), 4),
            'framing_x_scenario': round(float(var_ab), 4),
            'residual': round(float(var_e), 4),
            'icc_scenario': round(float(icc), 4),
        },
        'effect_sizes': {
            'partial_eta2': round(float(partial_eta2), 4),
            'marginal_r2': round(float(marginal_r2), 4),
            'conditional_r2': round(float(conditional_r2), 4),
        },
        'estimated_marginal_means': emm,
        'pairwise': [{
            'a': p['a'], 'b': p['b'],
            'diff': round(p['diff'], 4), 'se': round(p['se'], 4),
            't': round(p['t'], 3), 'df': p['df'],
            'p_raw': round(p['p_raw'], 6), 'p_holm': round(p['p_holm'], 6),
            'cohens_d': round(p['cohens_d'], 3),
            'sig': _sig_stars(p['p_holm']),
        } for p in pairs],
        'n_significant_pairs': sum(1 for p in pairs if p['p_holm'] < 0.05),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FAST MIXED F (for permutation tests — speed-critical)
# ═══════════════════════════════════════════════════════════════════════════════

def fast_mixed_F(y, fi_arr, bi_arr, a, b):
    """Compute F = MS_A / MS_AB for permutation inner loop."""
    N = len(y)
    grand_mean = y.mean()

    factor_means = np.array([y[fi_arr == i].mean() for i in range(a)])
    block_means = np.array([y[bi_arr == i].mean() for i in range(b)])
    factor_ns = np.array([(fi_arr == i).sum() for i in range(a)])

    SS_A = np.sum(factor_ns * (factor_means - grand_mean)**2)

    SS_AB = 0.0
    for i in range(a):
        for j in range(b):
            mask = (fi_arr == i) & (bi_arr == j)
            n_ij = mask.sum()
            if n_ij > 0:
                SS_AB += n_ij * (y[mask].mean() - factor_means[i]
                                 - block_means[j] + grand_mean)**2

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

    # Compute MS_AB
    grand_mean = y_arr.mean()
    factor_means = np.array([y_arr[fi_arr == i].mean() for i in range(a)])
    block_means = np.array([y_arr[bi_arr == i].mean() for i in range(b)])

    SS_AB = 0.0
    cell_counts = np.zeros((a, b))
    for i in range(a):
        for j in range(b):
            mask = (fi_arr == i) & (bi_arr == j)
            n_ij = mask.sum()
            cell_counts[i, j] = n_ij
            if n_ij > 0:
                SS_AB += n_ij * (y_arr[mask].mean() - factor_means[i]
                                 - block_means[j] + grand_mean)**2

    df_AB = (a - 1) * (b - 1)
    MS_AB = SS_AB / df_AB if df_AB > 0 else 0

    # Within-cell residual for Cohen's d
    SS_E = 0.0
    for i in range(a):
        for j in range(b):
            mask = (fi_arr == i) & (bi_arr == j)
            if mask.sum() > 0:
                SS_E += np.sum((y_arr[mask] - y_arr[mask].mean())**2)
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
            'a': fi, 'b': fj,
            'diff': round(float(diff), 4), 'se': round(float(se_diff), 4),
            't': round(float(t_val), 3), 'df': int(df_AB),
            'p_raw': float(p_raw), 'cohens_d': round(float(d), 3),
        })

    _holm_correct(pairs)
    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _sig_stars(p: float) -> str:
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''


def _holm_correct(pairs: list[dict]):
    """In-place Holm correction on a list of dicts with 'p_raw' key."""
    n = len(pairs)
    sorted_idx = sorted(range(n), key=lambda k: pairs[k]['p_raw'])
    for rank, idx in enumerate(sorted_idx):
        pairs[idx]['p_holm'] = min(float(pairs[idx]['p_raw'] * (n - rank)), 1.0)
    running_max = 0.0
    for idx in sorted_idx:
        running_max = max(running_max, pairs[idx]['p_holm'])
        pairs[idx]['p_holm'] = min(running_max, 1.0)
    for p in pairs:
        p['sig'] = _sig_stars(p['p_holm'])
        p['p_raw'] = round(p['p_raw'], 6)
        p['p_holm'] = round(p['p_holm'], 6)


def short_model(m: str) -> str:
    """Shorten model name for table display."""
    for long, short in [('Claude Sonnet 4.6', 'Claude'), ('GPT-5.3', 'GPT-5.3'),
                        ('Gemini 3.1 Pro', 'Gemini'), ('Kimi K2.5', 'Kimi'),
                        ('Qwen 3.5', 'Qwen'), ('MiniMax M2.7', 'MiniMax'),
                        ('GLM 5', 'GLM')]:
        if long in m:
            return short
    return m[:12]


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Full analysis pipeline")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to scored_merged.csv")
    parser.add_argument("--skip-permutation", action="store_true",
                        help="Skip permutation tests (faster)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing outputs")
    args = parser.parse_args()

    input_path = args.input or str(PROJECT_ROOT / "output" / "merge_scores"
                                   / "scored_merged.csv")
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()) and not args.force:
        sys.exit(f"Output dir {OUT_DIR} is not empty — use --force to overwrite.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    # 1. LOAD
    # ══════════════════════════════════════════════════════════════════════════

    log(f"Loading scored data from {_rel(input_path)}...")
    records = load_scored(input_path)
    if not records:
        print(f"ERROR: No valid records loaded from {input_path}", file=sys.stderr)
        sys.exit(1)

    MODELS = sorted(set(r['model'] for r in records))
    FRAMINGS = sorted(set(r['framing'] for r in records))
    SCENARIOS = sorted(set(r['scenario'] for r in records))

    log(f"  {len(records)} responses  |  {len(MODELS)} models  |  "
        f"{len(FRAMINGS)} framings  |  {len(SCENARIOS)} scenarios\n")

    # ══════════════════════════════════════════════════════════════════════════
    # 2. DESCRIPTIVE STATISTICS
    # ══════════════════════════════════════════════════════════════════════════

    log(f"{'='*76}")
    log("PART 1: DESCRIPTIVE STATISTICS")
    log(f"{'='*76}\n")

    agg = defaultdict(lambda: defaultdict(list))
    for r in records:
        key = (r['model'], r['framing'])
        for m in METRIC_KEYS:
            agg[key][m].append(r[m])

    aggregated = []
    for (model, framing), metrics in sorted(agg.items()):
        entry = {'model': model, 'framing': framing, 'n': len(metrics[METRIC_KEYS[0]])}
        for m, vals in metrics.items():
            mean = sum(vals) / len(vals)
            var = sum((v - mean)**2 for v in vals) / max(len(vals) - 1, 1)
            entry[m + '_mean_raw'] = mean
            entry[m + '_mean'] = round(mean, 2)
            entry[m + '_sd'] = round(var**0.5, 2)
        aggregated.append(entry)

    # Baselines (grand mean per model)
    baselines_raw = defaultdict(lambda: defaultdict(list))
    for e in aggregated:
        for k in e:
            if k.endswith('_mean_raw'):
                baselines_raw[e['model']][k].append(e[k])

    baselines_unrounded = {}
    baselines_json = {}
    for model, metrics in baselines_raw.items():
        baselines_unrounded[model] = {k: sum(v)/len(v) for k, v in metrics.items()}
        baselines_json[model] = {
            k.replace('_mean_raw', '_mean'): round(sum(v)/len(v), 2)
            for k, v in metrics.items()
        }

    # Deltas
    for e in aggregated:
        bl = baselines_unrounded[e['model']]
        for k in list(e.keys()):
            if k.endswith('_mean_raw'):
                base = k.replace('_mean_raw', '')
                e[base + '_delta'] = round(e[k] - bl[k], 2)

    # ── Reactivity ───────────────────────────────────────────────────────────
    # Standardized: each metric's |delta from own-model baseline| is divided by
    # the SD of that metric's 42 model x framing cell means, so metrics on
    # different scales contribute comparably; per-model score = mean over
    # framings of the mean standardized |delta| across the metric list.
    metric_sd = {}
    for m in METRIC_KEYS:
        vals = [e[m + '_mean_raw'] for e in aggregated]
        mu = sum(vals) / len(vals)
        sd = (sum((v - mu) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
        metric_sd[m] = sd if sd > 1e-10 else 1.0

    def compute_reactivity(metric_list, label):
        accum = defaultdict(list)
        for e in aggregated:
            bl = baselines_unrounded[e['model']]
            z = [abs(e[m + '_mean_raw'] - bl[m + '_mean_raw']) / metric_sd[m]
                 for m in metric_list]
            accum[e['model']].append(sum(z) / len(z))
        result = {m: round(sum(v)/len(v), 3) for m, v in accum.items()}

        log(f"\n  REACTIVITY — {label}  (mean standardized |delta|)")
        for model, score in sorted(result.items(), key=lambda x: -x[1]):
            bar = '|' * int(score * 40)
            log(f"    {model:25s}  {score:6.3f}  {bar}")
        return result

    reactivity_primary = compute_reactivity(
        REACTIVITY_PRIMARY, f"Primary ({len(REACTIVITY_PRIMARY)} interpersonal)")
    reactivity_sensitivity = compute_reactivity(
        REACTIVITY_SENSITIVITY,
        f"Sensitivity ({len(REACTIVITY_SENSITIVITY)} deterministic/classifier-only)")
    # Bridge index: confirmatory metrics only -> y-axis of the Comp1 trait bridge.
    reactivity_confirmatory = compute_reactivity(
        REACTIVITY_CONFIRMATORY,
        f"Confirmatory bridge index ({len(REACTIVITY_CONFIRMATORY)} "
        "confirmatory Tier-3 items, non-overlapping)")

    # ── Convergent validity across tiers ─────────────────────────────────────
    # Judge-artifact defence: if the Tier-3 metrics move in step
    # with deterministic/classifier shadows that have no LLM in the loop, the
    # judge is measuring text, not favouring it. Computed on the 42 cells,
    # both on raw cell means and on within-model deltas (framing movement net
    # of model identity). The 5-item composite is an application-specific
    # robustness check (item 6 had poor human-human reliability here).
    convergence_by_target = {}
    for target, target_label in CONVERGENCE_TARGETS:
        log(f"\n  CONVERGENT VALIDITY — {target} [{target_label}] vs "
            f"deterministic shadows (n={len(aggregated)} model x framing cells)")
        convergence = []
        t_raw = [e[target + '_mean_raw'] for e in aggregated]
        t_del = [e[target + '_mean_raw']
                 - baselines_unrounded[e['model']][target + '_mean_raw']
                 for e in aggregated]
        for shadow in CONVERGENCE_SHADOWS:
            s_raw = [e[shadow + '_mean_raw'] for e in aggregated]
            s_del = [e[shadow + '_mean_raw']
                     - baselines_unrounded[e['model']][shadow + '_mean_raw']
                     for e in aggregated]
            r_raw, p_raw = sp_stats.pearsonr(t_raw, s_raw)
            rho_raw = sp_stats.spearmanr(t_raw, s_raw)[0]
            r_del, p_del = sp_stats.pearsonr(t_del, s_del)
            rho_del = sp_stats.spearmanr(t_del, s_del)[0]
            convergence.append({
                'shadow': shadow, 'n_cells': len(aggregated),
                'pearson_cell_means': round(float(r_raw), 4),
                'spearman_cell_means': round(float(rho_raw), 4),
                'p_cell_means': round(float(p_raw), 6),
                'pearson_within_model_deltas': round(float(r_del), 4),
                'spearman_within_model_deltas': round(float(rho_del), 4),
                'p_within_model_deltas': round(float(p_del), 6),
            })
            log(f"    vs {shadow:16s}  cells: r={r_raw:+.3f} rho={rho_raw:+.3f}   "
                f"deltas: r={r_del:+.3f} rho={rho_del:+.3f}")
        convergence_by_target[target] = convergence
    convergence = convergence_by_target[CONVERGENCE_TARGET]

    # ── Per-model reactivity of the two composites (single-metric) ──────────
    reactivity_composite6 = compute_reactivity(
        [TIER3_COMPOSITE], 'Ruben 6-item composite alone (primary)')
    reactivity_composite5 = compute_reactivity(
        [TIER3_COMPOSITE_5ITEM],
        '5-item composite alone (sensitivity, item 6 removed)')

    # Clean aggregated for JSON
    for e in aggregated:
        for k in list(e.keys()):
            if k.endswith('_mean_raw'):
                del e[k]

    desc_output = {
        'aggregated': aggregated,
        'baselines': baselines_json,
        'reactivity_definition': 'mean standardized |delta from own-model '
                                 'baseline| across metric set (SD unit = SD of '
                                 'the 42 model x framing cell means per metric)',
        'reactivity_primary': reactivity_primary,
        'reactivity_sensitivity': reactivity_sensitivity,
        'reactivity_confirmatory_bridge_index': reactivity_confirmatory,
        'reactivity_confirmatory_note': (
            'RESOLVED 2026-07-15b: the bridge index now averages the 8 '
            'confirmatory INDIVIDUAL Tier-3 items (01,02,04,07,08,10,11,12); '
            'the relationship composite is excluded so no item is counted '
            'twice — consistent with the Fig 4 / PCA de-duplication. Same '
            'standardization (SD of the 42 model x framing cell means per '
            'metric).'),
        'reactivity_composite_6item': reactivity_composite6,
        'reactivity_composite_5item_sensitivity': reactivity_composite5,
        'convergent_validity': convergence,
        'convergent_validity_sensitivity_5item':
            convergence_by_target[TIER3_COMPOSITE_5ITEM],
        'composite_note': (
            'Primary composite = Ruben, Blanch-Hartigan & Hall (2026) '
            'relationship-oriented component, items [1,2,3,6,7,10]. Item 6 '
            'retained per the validated construct despite low human-human '
            'reliability in this application (kappa_w = .10); the 5-item '
            'variant (item 6 removed) is an application-specific '
            'sensitivity check, not a replacement.'),
    }

    # ══════════════════════════════════════════════════════════════════════════
    # 3. MIXED-MODEL ANOVA (per model)
    # ══════════════════════════════════════════════════════════════════════════

    log(f"\n\n{'='*76}")
    log("PART 2: MIXED-MODEL ANOVA")
    log("  metric ~ framing + (1|scenario) + framing:scenario")
    log("  F = MS_framing / MS_interaction  (scenario = random)")
    log(f"{'='*76}\n")

    anova_results = {}

    for model in MODELS:
        log(f"\n  MODEL: {model}")
        log(f"  {'─'*70}")
        mrecs = [r for r in records if r['model'] == model]
        fl = [r['framing'] for r in mrecs]
        sl = [r['scenario'] for r in mrecs]

        model_res = {}
        for metric in METRIC_KEYS:
            y = [r[metric] for r in mrecs]
            res = mixed_anova(y, fl, sl)
            model_res[metric] = res

            p = res['omnibus']['p_value']
            log(f"    {metric:18s}  F({res['omnibus']['df1']},{res['omnibus']['df2']})="
                f"{res['omnibus']['F']:7.2f}  p={p:.2e} {_sig_stars(p):3s}  "
                f"eta2p={res['effect_sizes']['partial_eta2']:.3f}  "
                f"ICC={res['variance_components']['icc_scenario']:.3f}  "
                f"sig.pairs={res['n_significant_pairs']:2d}/15")

        anova_results[model] = model_res

    # ── Cross-model summary tables ───────────────────────────────────────────

    header = f"{'Metric':18s}" + "".join(f"  {short_model(m):>10s}" for m in MODELS)
    sep = "─" * len(header)

    log(f"\n\n{'='*76}")
    log("SUMMARY: OMNIBUS p-VALUES")
    log(f"{'='*76}")
    log(header); log(sep)
    for metric in METRIC_KEYS:
        row = f"{metric:18s}"
        for model in MODELS:
            p = anova_results[model][metric]['omnibus']['p_value']
            if p < 0.001:   row += f"  {'<.001***':>10s}"
            elif p < 0.01:  row += f"  {f'{p:.3f} **':>10s}"
            elif p < 0.05:  row += f"  {f'{p:.3f}  *':>10s}"
            else:            row += f"  {f'{p:.3f} ns':>10s}"
        log(row)

    log(f"\n{'='*76}")
    log("SUMMARY: PARTIAL eta-squared")
    log(f"{'='*76}")
    log(header); log(sep)
    for metric in METRIC_KEYS:
        row = f"{metric:18s}"
        for model in MODELS:
            eta2 = anova_results[model][metric]['effect_sizes']['partial_eta2']
            row += f"  {eta2:10.3f}"
        log(row)

    log(f"\n{'='*76}")
    log("SUMMARY: ICC (scenario)")
    log(f"{'='*76}")
    log(header); log(sep)
    for metric in METRIC_KEYS:
        row = f"{metric:18s}"
        for model in MODELS:
            icc = anova_results[model][metric]['variance_components']['icc_scenario']
            row += f"  {icc:10.3f}"
        log(row)

    # ══════════════════════════════════════════════════════════════════════════
    # 4. PERMUTATION F-TESTS
    # ══════════════════════════════════════════════════════════════════════════

    N_PERM = 2000
    perm_results = {}

    if not args.skip_permutation:
        log(f"\n\n{'='*76}")
        log(f"PART 3: PERMUTATION F-TESTS ({N_PERM} iterations)")
        log(f"  Restricted: framing labels shuffled within scenario blocks")
        log(f"{'='*76}\n")

        rng = np.random.default_rng(seed=2024)

        for model in MODELS:
            log(f"  {model}")
            mrecs = [r for r in records if r['model'] == model]
            fl = [r['framing'] for r in mrecs]
            sl = [r['scenario'] for r in mrecs]

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
                df1 = a - 1
                df2 = (a - 1) * (b - 1)
                p_param = float(sp_stats.f.sf(F_obs, df1, df2))
                agree = (p_perm < 0.05) == (p_param < 0.05)

                perm_results[model][metric] = {
                    'F_observed': round(float(F_obs), 3),
                    'p_parametric': float(p_param),
                    'p_permutation': round(float(p_perm), 4),
                    'agree_at_05': agree,
                }

                match = "Y" if agree else "N"
                log(f"    {metric:18s}  F={F_obs:7.2f}  "
                    f"p_param={p_param:.2e}  p_perm={p_perm:.4f}  agree={match}")

            n_agree = sum(1 for m in METRIC_KEYS
                          if perm_results[model][m]['agree_at_05'])
            log(f"    -> {n_agree}/{len(METRIC_KEYS)} agree\n")
    else:
        log(f"\n  [Permutation tests skipped (--skip-permutation)]")

    # ══════════════════════════════════════════════════════════════════════════
    # 5. WITHIN-FRAMING MODEL COMPARISONS
    # ══════════════════════════════════════════════════════════════════════════

    log(f"\n\n{'='*76}")
    log("PART 4: WITHIN-FRAMING MODEL COMPARISONS")
    log("  metric ~ model + (1|scenario)")
    log(f"{'='*76}\n")

    within_framing = {}

    for framing in FRAMINGS:
        fshort = FRAMING_SHORT.get(framing, framing)
        log(f"  Framing: {fshort}")
        frecs = [r for r in records if r['framing'] == framing]
        ml = [r['model'] for r in frecs]
        sl = [r['scenario'] for r in frecs]

        within_framing[framing] = {}

        for metric in METRIC_KEYS:
            y = np.array([r[metric] for r in frecs], dtype=np.float64)

            framings_u = sorted(set(ml))
            scenarios_u = sorted(set(sl))
            a, b = len(framings_u), len(scenarios_u)
            fi_map = {f: i for i, f in enumerate(framings_u)}
            bi_map = {s: i for i, s in enumerate(scenarios_u)}
            fi_arr = np.array([fi_map[f] for f in ml], dtype=np.int32)
            bi_arr = np.array([bi_map[s] for s in sl], dtype=np.int32)

            F_val = fast_mixed_F(y, fi_arr, bi_arr, a, b)
            df1, df2 = a - 1, (a - 1) * (b - 1)
            p_val = float(sp_stats.f.sf(F_val, df1, df2))

            grand_mean = y.mean()
            factor_means = np.array([y[fi_arr == i].mean() for i in range(a)])
            factor_ns = np.array([(fi_arr == i).sum() for i in range(a)])
            SS_A_real = np.sum(factor_ns * (factor_means - grand_mean)**2)
            block_means = np.array([y[bi_arr == i].mean() for i in range(b)])
            SS_AB_real = 0.0
            for i in range(a):
                for j in range(b):
                    mask = (fi_arr == i) & (bi_arr == j)
                    n_ij = mask.sum()
                    if n_ij > 0:
                        SS_AB_real += n_ij * (y[mask].mean() - factor_means[i]
                                              - block_means[j] + grand_mean)**2
            eta2 = SS_A_real / (SS_A_real + SS_AB_real) if (SS_A_real + SS_AB_real) > 0 else 0

            pairs = pairwise_from_mixed(y, ml, sl)
            n_sig = sum(1 for p in pairs if p['p_holm'] < 0.05)

            model_means = {}
            for m in MODELS:
                vals = [r[metric] for r in frecs if r['model'] == m]
                model_means[m] = round(float(np.mean(vals)), 4) if vals else 0

            n_model_pairs = len(list(combinations(MODELS, 2)))

            within_framing[framing][metric] = {
                'omnibus': {
                    'F': round(float(F_val), 3), 'df1': int(df1), 'df2': int(df2),
                    'p_value': float(p_val), 'partial_eta2': round(float(eta2), 4),
                },
                'model_means': model_means,
                'pairwise': pairs,
                'n_sig_pairs': n_sig,
            }

            log(f"    {metric:18s}  F({df1},{df2})={F_val:7.2f}  "
                f"p={p_val:.2e} {_sig_stars(p_val):3s}  "
                f"eta2p={eta2:.3f}  sig.pairs={n_sig:2d}/{n_model_pairs}")
        log("")

    # ══════════════════════════════════════════════════════════════════════════
    # 6. PCA + WARD CLUSTERING (dual analysis)
    # ══════════════════════════════════════════════════════════════════════════

    log(f"\n{'='*76}")
    log("PART 5: PCA + WARD CLUSTERING")
    log(f"  A: all {len(PCA_METRICS)} metrics  |  "
        f"B: {len(INTERPERSONAL_METRICS)} interpersonal only")
    log("  (composites excluded from feature matrices - double-counting fix)")
    log(f"{'='*76}")

    def run_pca_cluster(metric_list, label):
        profile_labels = []
        profile_matrix = []
        for model in MODELS:
            for framing in FRAMINGS:
                mrecs = [r for r in records
                         if r['model'] == model and r['framing'] == framing]
                vals = [np.mean([r[m] for r in mrecs]) for m in metric_list]
                profile_matrix.append(vals)
                profile_labels.append((model, framing))

        X = np.array(profile_matrix)
        X_std = X.std(axis=0)
        X_std[X_std < 1e-10] = 1.0
        X_z = (X - X.mean(axis=0)) / X_std

        cov_mat = np.cov(X_z.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        var_explained = eigenvalues / np.sum(eigenvalues)
        cumulative = np.cumsum(var_explained)
        scores = X_z @ eigenvectors

        dist = pdist(X_z, metric='euclidean')
        Z_link = linkage(dist, method='ward')
        cluster3 = fcluster(Z_link, t=3, criterion='maxclust')

        log(f"\n  {label}")
        log(f"  {'─'*60}")
        log(f"  PCA variance explained:")
        for i in range(min(5, len(eigenvalues))):
            bar = '|' * int(var_explained[i] * 40)
            log(f"    PC{i+1}: {var_explained[i]*100:5.1f}%  "
                f"cumul: {cumulative[i]*100:5.1f}%  {bar}")

        log(f"\n  Loadings (PC1, PC2):")
        for j, m in enumerate(metric_list):
            log(f"    {m:18s}  {eigenvectors[j,0]:7.3f}  {eigenvectors[j,1]:7.3f}")

        log(f"\n  Ward clustering (k=3):")
        for c in [1, 2, 3]:
            members = [profile_labels[i] for i in range(len(cluster3)) if cluster3[i] == c]
            log(f"    Cluster {c} ({len(members)} members):")
            for m, f in members:
                log(f"      {m:25s} x {FRAMING_SHORT.get(f, f)}")

        return {
            'metrics_used': list(metric_list),
            'pca': {
                'variance_explained': [round(float(v), 4) for v in var_explained],
                'cumulative_variance': [round(float(v), 4) for v in cumulative],
                'loadings': {
                    m: {f'PC{pc+1}': round(float(eigenvectors[j, pc]), 4)
                        for pc in range(min(4, len(eigenvalues)))}
                    for j, m in enumerate(metric_list)
                },
                'scores': {
                    f"{m} x {FRAMING_SHORT.get(f, f)}": {
                        f'PC{pc+1}': round(float(scores[i, pc]), 4)
                        for pc in range(min(4, len(eigenvalues)))}
                    for i, (m, f) in enumerate(profile_labels)
                },
            },
            'ward_k3': {
                'labels': {
                    f"{m} x {FRAMING_SHORT.get(f, f)}": int(cluster3[i])
                    for i, (m, f) in enumerate(profile_labels)
                },
            },
            'dendrogram': {
                'linkage_matrix': [[round(float(v), 4) for v in row]
                                   for row in Z_link.tolist()],
                'leaf_labels': [f"{short_model(m)}x{FRAMING_SHORT.get(f, f)[:5]}"
                                for m, f in profile_labels],
            },
        }

    cluster_all = run_pca_cluster(PCA_METRICS,
                                  f"Analysis A: all {len(PCA_METRICS)} metrics")
    cluster_int = run_pca_cluster(INTERPERSONAL_METRICS,
                                  f"Analysis B: {len(INTERPERSONAL_METRICS)} interpersonal")

    # ══════════════════════════════════════════════════════════════════════════
    # 7. SAVE
    # ══════════════════════════════════════════════════════════════════════════

    desc_path = str(OUT_DIR / 'analysis_results.json')
    with open(desc_path, 'w') as f:
        json.dump(desc_output, f, indent=2)

    stats_output = {
        'anova': anova_results,
        'permutation_tests': perm_results,
        'within_framing': {
            framing: {
                metric: within_framing[framing][metric]
                for metric in METRIC_KEYS
            }
            for framing in FRAMINGS
        },
        'clustering': {
            'all_metrics': cluster_all,
            'interpersonal_metrics': cluster_int,
        },
    }
    stats_path = str(OUT_DIR / 'stats_results.json')
    with open(stats_path, 'w') as f:
        json.dump(stats_output, f, indent=2)

    txt_path = str(OUT_DIR / 'analysis_summary.txt')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(_log_lines))

    log(f"\n{'='*76}")
    log(f"Descriptive results -> {desc_path}")
    log(f"Statistical results -> {stats_path}")
    log(f"Summary text        -> {txt_path}")


if __name__ == "__main__":
    main()
