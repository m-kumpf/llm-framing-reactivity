#!/usr/bin/env python3
"""
Tier-3 judge-validation gate ("module 1").

Validates the DeepSeek-V4-Flash LLM judge against the two human raters
(miri, noreen) on the 210-response double-coded validation sample and
assigns each of the 13 Ruben items a gate status:

    confirmatory          — judge is a trustworthy stand-in for the human
                            criterion; item may serve as a confirmatory
                            endpoint downstream.
    descriptive_saturated — judge tracks humans nearly perfectly but the
                            item has (almost) no variance in this
                            population; excluded from confirmatory
                            endpoints without being labelled unreliable.
    exploratory           — agreement insufficient (sub-label
                            criterion_unreliable when the two humans do
                            not agree with each other).

The pre-registered anchor (manuscript v10) is quadratic-weighted Cohen's
kappa, judge vs. the two-rater median, 95% bootstrap CI, lower bound >= 0.40.
Additions specified before inference (2026-07-15): a human-criterion guard
(kappa_HH >= 0.40), the saturated-descriptive pathway, and the composite
gate ICC(A,1) lower bound >= 0.50.

Input:  validation_ratings_<rater>_<ts>.json          (Comp2 root, 2 files)
        output/sample_validation/validation_sample.csv (join bridge)
        run-judge-tier3-deepseek/judge_*.csv           (7 files, 4,200 rows)
        codebook.json                                  (item-key source of truth)
Output: output/analyse_tier3_judge/
            agreement_per_item.csv   — full statistical battery, 13 rows
            gate_verdicts.json       — machine-readable contract for module 2
            analysis_summary.txt     — human-readable log
            run_metadata.json        — timestamp, seed, params, sources

Usage:
    python analyse_tier3_judge.py
    python analyse_tier3_judge.py --n-boot 2000      # faster, exploratory only
    python analyse_tier3_judge.py --force            # overwrite existing output
    python analyse_tier3_judge.py --self-test        # run built-in checks, no I/O
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

BASE = Path(__file__).resolve().parents[1]  # Comp2_framing-reactivity/

RATER_FILES = {
    "miri": BASE / "validation_ratings_miri_2026-07-13T08-38-32.json",
    "noreen": BASE / "validation_ratings_noreen_2026-07-12T20-44-16.json",
}
SAMPLE_CSV = BASE / "output" / "sample_validation" / "validation_sample.csv"
JUDGE_DIR = BASE / "run-judge-tier3-deepseek"
CODEBOOK = BASE / "codebook.json"
DEFAULT_OUTDIR = BASE / "output" / "analyse_tier3_judge"

SCALE_MIN, SCALE_MAX = 1, 3
N_SAMPLES = 210
N_JUDGE_ROWS = 4200

# Gate thresholds — printed to gate_verdicts.json so the methods section can
# cite them as specified-before-inference.
KAPPA_MED_LB_FLOOR = 0.40   # manuscript pre-registration (anchor rule)
HH_KAPPA_FLOOR = 0.40       # human-criterion guard (rule 1)
SAT_HUMAN_MODAL = 0.80      # saturation pathway (rule 2)
SAT_JUDGE_MODAL = 0.90
SAT_WPA_FLOOR = 0.90
SAT_AC2_LB_FLOOR = 0.60
COMPOSITE_ICC_LB_FLOOR = 0.50  # Koo & Li "moderate" floor

# Non-gating warning flags
FLAG_BIAS = 0.15
FLAG_TWO_STEP = 0.05
FLAG_WEAK_CRITERION = 0.60
FLAG_KNIFE_EDGE = 0.05
UNSTABLE_CI_FRAC = 0.02

COMPOSITE_ITEM_NS = [1, 2, 3, 6, 7, 10]  # relationship-oriented composite


# ═══════════════════════════════════════════════════════════════════════════════
# CORE STATISTICS — everything per item is a function of the joint
# (miri, noreen, judge) category counts, a length-27 vector. Bootstrap and
# jackknife therefore operate on (R, 27) count matrices, fully vectorised.
# ═══════════════════════════════════════════════════════════════════════════════

# joint code c = (m-1)*9 + (n-1)*3 + (j-1), c in 0..26
_C = np.arange(27)
VAL_MIRI = (_C // 9 + 1).astype(float)
VAL_NOREEN = ((_C // 3) % 3 + 1).astype(float)
VAL_JUDGE = (_C % 3 + 1).astype(float)
VAL_MEDIAN = (VAL_MIRI + VAL_NOREEN) / 2.0  # two raters: median == mean

GRID_INT = np.array([1.0, 2.0, 3.0])
GRID_HALF = np.array([1.0, 1.5, 2.0, 2.5, 3.0])


def _dist(gx, gy, power):
    """Normalised disagreement weights d(x,y) = (|x-y|/(max-min))^power."""
    return (np.abs(gx[:, None] - gy[None, :]) / (SCALE_MAX - SCALE_MIN)) ** power


class Pairing:
    """A rater pairing (x, y) with value maps over the 27 joint codes.

    All statistics take a counts matrix of shape (R, 27) — R replicates —
    and return a length-R vector. A single dataset is R = 1.
    """

    def __init__(self, vx, vy, gx, gy, power=2):
        self.vx, self.vy = vx, vy
        self.gx, self.gy = gx, gy
        self.dvec = (np.abs(vx - vy) / (SCALE_MAX - SCALE_MIN)) ** power
        self.Mx = (vx[:, None] == gx[None, :]).astype(float)  # 27 × |gx|
        self.My = (vy[:, None] == gy[None, :]).astype(float)
        self.D = _dist(gx, gy, power)

    def kappa(self, counts):
        """Weighted kappa: 1 - E[d observed] / E[d under marginal independence]."""
        counts = np.atleast_2d(counts).astype(float)
        n = counts.sum(axis=1)
        obs = counts @ self.dvec / n
        Rm = counts @ self.Mx
        Cm = counts @ self.My
        exp = np.einsum("rg,gh,rh->r", Rm, self.D, Cm) / n**2
        with np.errstate(divide="ignore", invalid="ignore"):
            k = 1.0 - obs / exp
        k[exp <= 0] = np.nan
        return k

    def ac2(self, counts):
        """Gwet's AC2 with the pairing's (quadratic) agreement weights.

        Requires gx == gy (a common category grid). Chance term:
        p_e = T_w / (K(K-1)) * sum_k pi_k (1 - pi_k), T_w = sum of all
        agreement weights, pi_k = average marginal prevalence.
        """
        assert np.array_equal(self.gx, self.gy), "AC2 needs a shared grid"
        counts = np.atleast_2d(counts).astype(float)
        n = counts.sum(axis=1)
        K = len(self.gx)
        W = 1.0 - self.D
        pa = counts @ (1.0 - self.dvec) / n
        pi = (counts @ self.Mx / n[:, None] + counts @ self.My / n[:, None]) / 2.0
        pe = W.sum() / (K * (K - 1)) * (pi * (1.0 - pi)).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ac = (pa - pe) / (1.0 - pe)
        ac[np.isclose(pe, 1.0)] = np.nan
        return ac

    def exact_agreement(self, counts):
        counts = np.atleast_2d(counts).astype(float)
        return counts @ (self.vx == self.vy).astype(float) / counts.sum(axis=1)

    def weighted_agreement(self, counts):
        counts = np.atleast_2d(counts).astype(float)
        return counts @ (1.0 - self.dvec) / counts.sum(axis=1)

    def mean_signed_diff(self, counts):
        """Mean of (x - y): positive = x scores higher than y."""
        counts = np.atleast_2d(counts).astype(float)
        return counts @ (self.vx - self.vy) / counts.sum(axis=1)

    def two_step_rate(self, counts):
        counts = np.atleast_2d(counts).astype(float)
        return counts @ (np.abs(self.vx - self.vy) >= 2).astype(float) / counts.sum(axis=1)


def make_pairings(power=2):
    return {
        "JMED": Pairing(VAL_JUDGE, VAL_MEDIAN, GRID_INT, GRID_HALF, power),
        "JM": Pairing(VAL_JUDGE, VAL_MIRI, GRID_INT, GRID_INT, power),
        "JN": Pairing(VAL_JUDGE, VAL_NOREEN, GRID_INT, GRID_INT, power),
        "HH": Pairing(VAL_MIRI, VAL_NOREEN, GRID_INT, GRID_INT, power),
    }


PAIRINGS = make_pairings(power=2)
JMED_LINEAR = Pairing(VAL_JUDGE, VAL_MEDIAN, GRID_INT, GRID_HALF, power=1)


def joint_codes(m, n, j):
    return ((m - 1) * 9 + (n - 1) * 3 + (j - 1)).astype(int)


def counts_from_codes(codes):
    return np.bincount(codes, minlength=27).astype(float)


def bootstrap_counts(codes, n_boot, rng):
    """(n_boot, 27) joint-category counts from paired response resampling."""
    n = len(codes)
    idx = rng.integers(0, n, size=(n_boot, n))
    flat = codes[idx] + 27 * np.arange(n_boot)[:, None]
    return (
        np.bincount(flat.ravel(), minlength=27 * n_boot)
        .reshape(n_boot, 27)
        .astype(float)
    )


def jackknife_counts(codes):
    """(n, 27) leave-one-out counts."""
    full = counts_from_codes(codes)
    eye = np.eye(27)
    return full[None, :] - eye[codes]


def percentile_ci(boots, alpha=0.05):
    b = boots[~np.isnan(boots)]
    if len(b) == 0:
        return (np.nan, np.nan)
    return (float(np.percentile(b, 100 * alpha / 2)), float(np.percentile(b, 100 * (1 - alpha / 2))))


def bca_ci(theta_hat, boots, jack, alpha=0.05):
    """BCa interval from an existing bootstrap distribution + jackknife values."""
    b = boots[~np.isnan(boots)]
    jk = jack[~np.isnan(jack)]
    if len(b) == 0 or len(jk) < 2 or np.isnan(theta_hat):
        return (np.nan, np.nan)
    prop = np.clip((b < theta_hat).mean() + 0.5 * (b == theta_hat).mean(), 1e-6, 1 - 1e-6)
    z0 = sp_stats.norm.ppf(prop)
    jm = jk.mean()
    num = ((jm - jk) ** 3).sum()
    den = 6.0 * (((jm - jk) ** 2).sum()) ** 1.5
    a = num / den if den > 0 else 0.0
    lo_hi = []
    for za in (sp_stats.norm.ppf(alpha / 2), sp_stats.norm.ppf(1 - alpha / 2)):
        adj = sp_stats.norm.cdf(z0 + (z0 + za) / (1 - a * (z0 + za)))
        lo_hi.append(float(np.percentile(b, 100 * np.clip(adj, 1e-6, 1 - 1e-6))))
    return tuple(lo_hi)


def icc_a1(mat):
    """ICC(A,1): two-way, absolute agreement, single measure (McGraw & Wong).

    mat: (n_subjects, k_raters).
    """
    mat = np.asarray(mat, dtype=float)
    n, k = mat.shape
    grand = mat.mean()
    row_means = mat.mean(axis=1)
    col_means = mat.mean(axis=0)
    ssr = k * ((row_means - grand) ** 2).sum()
    ssc = n * ((col_means - grand) ** 2).sum()
    sst = ((mat - grand) ** 2).sum()
    sse = sst - ssr - ssc
    msr = ssr / (n - 1)
    msc = ssc / (k - 1)
    mse = sse / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k / n * (msc - mse))


def icc_a1_boot_pairs(X, Y):
    """Vectorised ICC(A,1) for k = 2 raters. X, Y: (R, n) replicate matrices."""
    X = np.atleast_2d(X).astype(float)
    Y = np.atleast_2d(Y).astype(float)
    n = X.shape[1]
    cx, cy = X.mean(axis=1), Y.mean(axis=1)
    grand = (cx + cy) / 2.0
    m = (X + Y) / 2.0
    ssr = 2.0 * ((m - grand[:, None]) ** 2).sum(axis=1)
    ssc = n * ((cx - grand) ** 2 + (cy - grand) ** 2)
    sst = ((X - grand[:, None]) ** 2).sum(axis=1) + ((Y - grand[:, None]) ** 2).sum(axis=1)
    sse = sst - ssr - ssc
    msr = ssr / (n - 1)
    msc = ssc / 1.0
    mse = sse / (n - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        icc = (msr - mse) / (msr + mse + 2.0 / n * (msc - mse))
    return icc


# ═══════════════════════════════════════════════════════════════════════════════
# DATA ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════


def load_codebook():
    with open(CODEBOOK) as f:
        cb = json.load(f)
    items = [it["key"] for it in cb["items"]]
    labels = {it["key"]: it["label"] for it in cb["items"]}
    assert len(items) == 13, f"codebook has {len(items)} items, expected 13"
    assert cb["scale"]["values"] == [1, 2, 3], "codebook scale is not 1/2/3"
    return items, labels


def load_human_ratings(item_keys):
    frames, notes_counts = {}, {}
    for rater, path in RATER_FILES.items():
        with open(path) as f:
            recs = json.load(f)
        df = pd.DataFrame(recs).set_index("sample_id").sort_index()
        assert list(df.index) == list(range(1, N_SAMPLES + 1)), f"{rater}: sample_ids not 1..{N_SAMPLES}"
        assert (df["rater_id"] == rater).all(), f"{rater}: rater_id mismatch"
        missing = [k for k in item_keys if k not in df.columns]
        assert not missing, f"{rater}: missing item columns {missing}"
        vals = df[item_keys]
        assert vals.notna().all().all(), f"{rater}: NaN scores present"
        assert vals.isin([1, 2, 3]).all().all(), f"{rater}: scores outside 1..3"
        frames[rater] = vals.astype(int)
        notes_counts[rater] = int((df["notes"].fillna("").str.strip() != "").sum())
    return frames, notes_counts


def load_judge(item_keys):
    files = sorted(JUDGE_DIR.glob("judge_*.csv"))
    assert len(files) == 7, f"expected 7 judge CSVs, found {len(files)}"
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    assert len(df) == N_JUDGE_ROWS, f"judge rows: {len(df)} != {N_JUDGE_ROWS}"
    assert (df["status"] == "ok").all(), "judge rows with status != ok"
    vals = df[item_keys]
    assert vals.notna().all().all() and vals.isin([1, 2, 3]).all().all(), "judge scores outside 1..3"
    return df, files


def assemble(item_keys):
    """Return tidy per-sample frame with miri/noreen/judge scores per item."""
    humans, notes_counts = load_human_ratings(item_keys)
    judge, judge_files = load_judge(item_keys)

    sample = pd.read_csv(SAMPLE_CSV)
    assert len(sample) == N_SAMPLES, f"validation_sample rows: {len(sample)}"
    assert sample["sample_id"].is_unique
    placeholder = [c for c in sample.columns if c.startswith("item_")]
    assert sample[placeholder].isna().all().all(), (
        "validation_sample.csv item_* columns are not all-NaN placeholders — refusing to guess"
    )
    keys = sample[["sample_id", "_model", "vignette_id", "_run_number", "scenario_id", "framing_id"]].rename(
        columns={"_model": "model", "_run_number": "run_number"}
    )

    merged = keys.merge(
        judge,
        on=["model", "vignette_id", "run_number"],
        how="left",
        indicator=True,
        validate="one_to_one",
        suffixes=("", "_judge"),
    )
    n_both = int((merged["_merge"] == "both").sum())
    assert n_both == N_SAMPLES, f"join matched {n_both}/{N_SAMPLES} samples"
    assert (merged["scenario_id"] == merged["scenario_id_judge"]).all()
    assert (merged["framing_id"] == merged["framing_id_judge"]).all()
    merged = merged.sort_values("sample_id").set_index("sample_id")

    scores = {}
    for key in item_keys:
        scores[key] = pd.DataFrame(
            {
                "miri": humans["miri"][key],
                "noreen": humans["noreen"][key],
                "judge": merged[key].astype(int),
            }
        )
    integrity = {
        "n_samples": N_SAMPLES,
        "n_judge_rows": len(judge),
        "join_matched": n_both,
        "judge_files": [f.name for f in judge_files],
        "notes_counts": notes_counts,
    }
    return scores, integrity


# ═══════════════════════════════════════════════════════════════════════════════
# PER-ITEM ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════


def analyse_item(scores, n_boot, rng):
    """scores: DataFrame with miri/noreen/judge columns. Returns stats dict."""
    m = scores["miri"].to_numpy()
    n_arr = scores["noreen"].to_numpy()
    j = scores["judge"].to_numpy()
    codes = joint_codes(m, n_arr, j)
    full = counts_from_codes(codes)
    boot = bootstrap_counts(codes, n_boot, rng)
    jack = jackknife_counts(codes)

    out = {}
    out["dist"] = {
        "miri": np.bincount(m, minlength=4)[1:].tolist(),
        "noreen": np.bincount(n_arr, minlength=4)[1:].tolist(),
        "judge": np.bincount(j, minlength=4)[1:].tolist(),
    }

    boots_cache = {}
    for name, p in PAIRINGS.items():
        point = float(p.kappa(full)[0])
        b = p.kappa(boot)
        boots_cache[name] = b
        lo, hi = percentile_ci(b)
        out[f"kappa_{name}"] = point
        out[f"kappa_{name}_ci"] = (lo, hi)
        out[f"kappa_{name}_degen_frac"] = float(np.isnan(b).mean())
        out[f"pct_exact_{name}"] = float(p.exact_agreement(full)[0])

    jk = PAIRINGS["JMED"].kappa(jack)
    out["kappa_JMED_bca"] = bca_ci(out["kappa_JMED"], boots_cache["JMED"], jk)
    out["kappa_JMED_linear"] = float(JMED_LINEAR.kappa(full)[0])
    out["kappa_JMED_linear_ci"] = percentile_ci(JMED_LINEAR.kappa(boot))

    for name in ("JM", "JN", "HH"):
        p = PAIRINGS[name]
        point = float(p.ac2(full)[0])
        b = p.ac2(boot)
        out[f"ac2_{name}"] = point
        out[f"ac2_{name}_ci"] = percentile_ci(b)

    p_med = PAIRINGS["JMED"]
    out["weighted_pct_agree"] = float(p_med.weighted_agreement(full)[0])
    out["bias"] = float(p_med.mean_signed_diff(full)[0])
    out["bias_ci"] = percentile_ci(p_med.mean_signed_diff(boot))
    for name in ("JM", "JN", "HH"):
        out[f"two_step_{name}"] = float(PAIRINGS[name].two_step_rate(full)[0])

    d_noninf = boots_cache["HH"] - (boots_cache["JM"] + boots_cache["JN"]) / 2.0
    out["noninf_delta"] = float(out["kappa_HH"] - (out["kappa_JM"] + out["kappa_JN"]) / 2.0)
    out["noninf_delta_ci"] = percentile_ci(d_noninf)

    med = (m + n_arr) / 2.0
    rho, _ = sp_stats.spearmanr(j, med)
    out["spearman"] = float(rho) if np.isfinite(rho) else np.nan

    out["human_modal_share"] = float(
        max((np.bincount(m, minlength=4)[1:] + np.bincount(n_arr, minlength=4)[1:])) / (2 * len(m))
    )
    out["judge_modal_share"] = float(np.bincount(j, minlength=4)[1:].max() / len(j))
    return out


def apply_gate(s):
    """Gate decision rule, evaluated in order. Returns (status, rule_path, reasons, flags)."""
    flags = []
    if abs(s["bias"]) > FLAG_BIAS:
        flags.append("bias")
    if max(s["two_step_JM"], s["two_step_JN"]) > FLAG_TWO_STEP:
        flags.append("two_step")
    if s["kappa_HH"] < FLAG_WEAK_CRITERION:
        flags.append("weak_criterion")
    lb = s["kappa_JMED_ci"][0]
    if np.isfinite(lb) and abs(lb - KAPPA_MED_LB_FLOOR) < FLAG_KNIFE_EDGE:
        flags.append("knife_edge")
    if s["kappa_JMED_degen_frac"] > UNSTABLE_CI_FRAC:
        flags.append("unstable_ci")

    # Rule 1 — CONFIRMATORY (anchor rule + human-criterion guard)
    if np.isfinite(lb) and lb >= KAPPA_MED_LB_FLOOR and s["kappa_HH"] >= HH_KAPPA_FLOOR:
        return (
            "confirmatory",
            "rule1",
            [
                f"LB95(kappa_w judge-vs-median) = {lb:.3f} >= {KAPPA_MED_LB_FLOOR}",
                f"kappa_w human-human = {s['kappa_HH']:.3f} >= {HH_KAPPA_FLOOR}",
            ],
            flags,
        )

    # Rule 2 — DESCRIPTIVE-SATURATED
    modal_ok = (
        s["human_modal_share"] >= SAT_HUMAN_MODAL or s["judge_modal_share"] >= SAT_JUDGE_MODAL
    )
    ac2_lb = min(s["ac2_JM_ci"][0], s["ac2_JN_ci"][0])
    if modal_ok and s["weighted_pct_agree"] >= SAT_WPA_FLOOR and ac2_lb >= SAT_AC2_LB_FLOOR:
        return (
            "descriptive_saturated",
            "rule2",
            [
                f"modal share: humans {s['human_modal_share']:.2f} / judge {s['judge_modal_share']:.2f} "
                f"(thresholds {SAT_HUMAN_MODAL}/{SAT_JUDGE_MODAL})",
                f"weighted %agreement = {s['weighted_pct_agree']:.3f} >= {SAT_WPA_FLOOR}",
                f"min AC2 LB95 (J-M, J-N) = {ac2_lb:.3f} >= {SAT_AC2_LB_FLOOR}",
            ],
            flags,
        )

    # Rule 3 — EXPLORATORY
    reasons = [f"LB95(kappa_w judge-vs-median) = {lb:.3f} < {KAPPA_MED_LB_FLOOR}"]
    if s["kappa_HH"] < HH_KAPPA_FLOOR:
        flags.append("criterion_unreliable")
        reasons.append(
            f"kappa_w human-human = {s['kappa_HH']:.3f} < {HH_KAPPA_FLOOR} (criterion unreliable)"
        )
    return ("exploratory", "rule3", reasons, flags)


# ═══════════════════════════════════════════════════════════════════════════════
# COMPOSITE
# ═══════════════════════════════════════════════════════════════════════════════


def analyse_composite(scores, item_keys, n_boot, rng, exclude=()):
    comp_keys = [k for k in item_keys if int(k.split("_")[1]) in COMPOSITE_ITEM_NS]
    comp_keys = [k for k in comp_keys if k not in exclude]
    m = np.mean([scores[k]["miri"].to_numpy() for k in comp_keys], axis=0)
    n_arr = np.mean([scores[k]["noreen"].to_numpy() for k in comp_keys], axis=0)
    j = np.mean([scores[k]["judge"].to_numpy() for k in comp_keys], axis=0)
    h = (m + n_arr) / 2.0

    n = len(j)
    idx = rng.integers(0, n, size=(n_boot, n))

    def boot_icc(a, b):
        return icc_a1_boot_pairs(a[idx], b[idx])

    res = {
        "items": comp_keys,
        "icc_judge_vs_humanmean": float(icc_a1(np.column_stack([j, h]))),
        "icc_judge_vs_humanmean_ci": percentile_ci(boot_icc(j, h)),
        "icc_judge_vs_miri": float(icc_a1(np.column_stack([j, m]))),
        "icc_judge_vs_noreen": float(icc_a1(np.column_stack([j, n_arr]))),
        "icc_human_human": float(icc_a1(np.column_stack([m, n_arr]))),
        "icc_human_human_ci": percentile_ci(boot_icc(m, n_arr)),
        "pearson_judge_humanmean": float(np.corrcoef(j, h)[0, 1]),
    }
    return res


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT WRITERS
# ═══════════════════════════════════════════════════════════════════════════════


def write_outputs(results, comp, comp_sens, labels, integrity, outdir, args):
    rows = []
    for key, s in results.items():
        rows.append(
            {
                "item": key,
                "label": labels[key],
                **{f"miri_{c}": s["dist"]["miri"][c - 1] for c in (1, 2, 3)},
                **{f"noreen_{c}": s["dist"]["noreen"][c - 1] for c in (1, 2, 3)},
                **{f"judge_{c}": s["dist"]["judge"][c - 1] for c in (1, 2, 3)},
                "kappa_w_med": s["kappa_JMED"],
                "kappa_w_med_lo": s["kappa_JMED_ci"][0],
                "kappa_w_med_hi": s["kappa_JMED_ci"][1],
                "kappa_w_med_bca_lo": s["kappa_JMED_bca"][0],
                "kappa_w_med_bca_hi": s["kappa_JMED_bca"][1],
                "kappa_w_med_linear": s["kappa_JMED_linear"],
                "kappa_w_JM": s["kappa_JM"],
                "kappa_w_JM_lo": s["kappa_JM_ci"][0],
                "kappa_w_JM_hi": s["kappa_JM_ci"][1],
                "kappa_w_JN": s["kappa_JN"],
                "kappa_w_JN_lo": s["kappa_JN_ci"][0],
                "kappa_w_JN_hi": s["kappa_JN_ci"][1],
                "kappa_w_HH": s["kappa_HH"],
                "kappa_w_HH_lo": s["kappa_HH_ci"][0],
                "kappa_w_HH_hi": s["kappa_HH_ci"][1],
                "ac2_JM": s["ac2_JM"],
                "ac2_JM_lo": s["ac2_JM_ci"][0],
                "ac2_JM_hi": s["ac2_JM_ci"][1],
                "ac2_JN": s["ac2_JN"],
                "ac2_JN_lo": s["ac2_JN_ci"][0],
                "ac2_JN_hi": s["ac2_JN_ci"][1],
                "ac2_HH": s["ac2_HH"],
                "pct_exact_JM": s["pct_exact_JM"],
                "pct_exact_JN": s["pct_exact_JN"],
                "pct_exact_HH": s["pct_exact_HH"],
                "weighted_pct_agree_med": s["weighted_pct_agree"],
                "bias_judge_minus_median": s["bias"],
                "bias_lo": s["bias_ci"][0],
                "bias_hi": s["bias_ci"][1],
                "two_step_JM": s["two_step_JM"],
                "two_step_JN": s["two_step_JN"],
                "two_step_HH": s["two_step_HH"],
                "noninf_delta": s["noninf_delta"],
                "noninf_delta_lo": s["noninf_delta_ci"][0],
                "noninf_delta_hi": s["noninf_delta_ci"][1],
                "spearman": s["spearman"],
                "degen_boot_frac": s["kappa_JMED_degen_frac"],
                "human_modal_share": s["human_modal_share"],
                "judge_modal_share": s["judge_modal_share"],
                "flags": ";".join(s["flags"]),
                "status": s["status"],
                "rule_path": s["rule_path"],
            }
        )
    pd.DataFrame(rows).to_csv(outdir / "agreement_per_item.csv", index=False)

    config = {
        "KAPPA_MED_LB_FLOOR": KAPPA_MED_LB_FLOOR,
        "HH_KAPPA_FLOOR": HH_KAPPA_FLOOR,
        "SAT_HUMAN_MODAL": SAT_HUMAN_MODAL,
        "SAT_JUDGE_MODAL": SAT_JUDGE_MODAL,
        "SAT_WPA_FLOOR": SAT_WPA_FLOOR,
        "SAT_AC2_LB_FLOOR": SAT_AC2_LB_FLOOR,
        "COMPOSITE_ICC_LB_FLOOR": COMPOSITE_ICC_LB_FLOOR,
        "composite_items": COMPOSITE_ITEM_NS,
        "flags": {
            "bias": FLAG_BIAS,
            "two_step": FLAG_TWO_STEP,
            "weak_criterion": FLAG_WEAK_CRITERION,
            "knife_edge": FLAG_KNIFE_EDGE,
            "unstable_ci": UNSTABLE_CI_FRAC,
        },
        "n_boot": args.n_boot,
        "seed": args.seed,
        "weights": "quadratic (linear as sensitivity)",
        "median_definition": "two-rater median == mean, half-point grid",
        "deviations_from_manuscript_prespec": [
            "rule1 adds human-criterion guard kappa_HH >= 0.40",
            "rule2 adds descriptive_saturated pathway for degenerate-marginal items",
            "composite gated by ICC(A,1) LB95 >= 0.50 (manuscript specified only item-level rule)",
        ],
    }
    comp_status = (
        "confirmatory"
        if comp["icc_judge_vs_humanmean_ci"][0] >= COMPOSITE_ICC_LB_FLOOR
        else "exploratory"
    )
    verdicts = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config,
        "items": {
            key: {
                "status": s["status"],
                "rule_path": s["rule_path"],
                "reasons": s["reasons"],
                "flags": s["flags"],
                "stats": {
                    "kappa_w_med": s["kappa_JMED"],
                    "kappa_w_med_ci": list(s["kappa_JMED_ci"]),
                    "kappa_w_HH": s["kappa_HH"],
                    "ac2_JM": s["ac2_JM"],
                    "ac2_JN": s["ac2_JN"],
                    "weighted_pct_agree": s["weighted_pct_agree"],
                    "bias": s["bias"],
                },
            }
            for key, s in results.items()
        },
        "composite": {
            "status": comp_status,
            "items": comp["items"],
            "icc_judge_vs_humanmean": comp["icc_judge_vs_humanmean"],
            "icc_judge_vs_humanmean_ci": list(comp["icc_judge_vs_humanmean_ci"]),
            "icc_human_human": comp["icc_human_human"],
            "icc_human_human_ci": list(comp["icc_human_human_ci"]),
            "icc_judge_vs_miri": comp["icc_judge_vs_miri"],
            "icc_judge_vs_noreen": comp["icc_judge_vs_noreen"],
            "pearson_judge_humanmean": comp["pearson_judge_humanmean"],
            "sensitivity_excl_item06": {
                "items": comp_sens["items"],
                "icc_judge_vs_humanmean": comp_sens["icc_judge_vs_humanmean"],
                "icc_judge_vs_humanmean_ci": list(comp_sens["icc_judge_vs_humanmean_ci"]),
            },
        },
    }
    with open(outdir / "gate_verdicts.json", "w") as f:
        json.dump(verdicts, f, indent=2)

    lines = []
    lines.append("Tier-3 judge-validation gate — analysis summary")
    lines.append("=" * 70)
    lines.append(f"Generated (UTC): {verdicts['generated_utc']}")
    lines.append(f"Bootstrap: {args.n_boot} iterations, seed {args.seed}, percentile CIs (BCa sensitivity)")
    lines.append("")
    lines.append("Integrity checks (all hard-asserted):")
    lines.append(f"  human ratings: 2 raters x {integrity['n_samples']} samples, complete, values in 1..3")
    lines.append(f"  judge rows: {integrity['n_judge_rows']} (7 files), all status=ok")
    lines.append(f"  join human->sample->judge: {integrity['join_matched']}/{integrity['n_samples']} one-to-one")
    lines.append(f"  rater notes (non-empty): {integrity['notes_counts']}")
    lines.append("")
    lines.append(f"{'item':<34}{'kappa_med [95% CI]':<24}{'kappa_HH':<10}{'AC2 J-M/J-N':<14}{'status':<24}flags")
    lines.append("-" * 120)
    for key, s in results.items():
        ci = f"{s['kappa_JMED']:.2f} [{s['kappa_JMED_ci'][0]:.2f}, {s['kappa_JMED_ci'][1]:.2f}]"
        ac = f"{s['ac2_JM']:.2f}/{s['ac2_JN']:.2f}"
        lines.append(
            f"{key:<34}{ci:<24}{s['kappa_HH']:<10.2f}{ac:<14}{s['status']:<24}{';'.join(s['flags'])}"
        )
    lines.append("")
    lines.append("Composite (relationship-oriented: items 01,02,03,06,07,10):")
    lines.append(
        f"  ICC(A,1) judge vs human-mean = {comp['icc_judge_vs_humanmean']:.3f} "
        f"[{comp['icc_judge_vs_humanmean_ci'][0]:.3f}, {comp['icc_judge_vs_humanmean_ci'][1]:.3f}] "
        f"-> {comp_status.upper()} (floor {COMPOSITE_ICC_LB_FLOOR})"
    )
    lines.append(
        f"  human-human ICC = {comp['icc_human_human']:.3f} "
        f"[{comp['icc_human_human_ci'][0]:.3f}, {comp['icc_human_human_ci'][1]:.3f}] (ceiling)"
    )
    lines.append(
        f"  sensitivity excl. item_06: ICC = {comp_sens['icc_judge_vs_humanmean']:.3f} "
        f"[{comp_sens['icc_judge_vs_humanmean_ci'][0]:.3f}, {comp_sens['icc_judge_vs_humanmean_ci'][1]:.3f}]"
    )
    lines.append("")
    counts = pd.Series([s["status"] for s in results.values()]).value_counts().to_dict()
    lines.append(f"Verdict counts: {counts}")
    (outdir / "analysis_summary.txt").write_text("\n".join(lines) + "\n")

    meta = {
        "script": "analyse_tier3_judge.py",
        "timestamp_utc": verdicts["generated_utc"],
        "seed": args.seed,
        "n_boot": args.n_boot,
        "sources": {
            "rater_files": {k: str(v) for k, v in RATER_FILES.items()},
            "validation_sample": str(SAMPLE_CSV),
            "judge_files": integrity["judge_files"],
            "codebook": str(CODEBOOK),
        },
        "row_counts": {
            "human_per_rater": integrity["n_samples"],
            "judge": integrity["n_judge_rows"],
            "joined": integrity["join_matched"],
        },
        "outputs": [
            "agreement_per_item.csv",
            "gate_verdicts.json",
            "analysis_summary.txt",
        ],
    }
    with open(outdir / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-TESTS
# ═══════════════════════════════════════════════════════════════════════════════


def _reference_weighted_kappa(x, y, power=2):
    """Independent (slow, loop-based) weighted kappa for cross-checking."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    cats = np.unique(np.concatenate([x, y]))
    n = len(x)
    obs = np.zeros((len(cats), len(cats)))
    for xi, yi in zip(x, y):
        obs[np.where(cats == xi)[0][0], np.where(cats == yi)[0][0]] += 1
    obs /= n
    rm, cm = obs.sum(1), obs.sum(0)
    exp = np.outer(rm, cm)
    d = (np.abs(cats[:, None] - cats[None, :]) / (SCALE_MAX - SCALE_MIN)) ** power
    return 1 - (obs * d).sum() / (exp * d).sum()


def _reference_ac2(x, y, cats=(1, 2, 3), power=2):
    """Independent loop-based Gwet AC2 (quadratic agreement weights)."""
    cats = np.asarray(cats, float)
    K = len(cats)
    d = (np.abs(cats[:, None] - cats[None, :]) / (SCALE_MAX - SCALE_MIN)) ** power
    w = 1 - d
    n = len(x)
    pa = np.mean([w[int(a) - 1, int(b) - 1] for a, b in zip(x, y)])
    pi = np.array([(np.mean(np.asarray(x) == c) + np.mean(np.asarray(y) == c)) / 2 for c in cats])
    pe = w.sum() / (K * (K - 1)) * (pi * (1 - pi)).sum()
    return (pa - pe) / (1 - pe)


def self_test():
    rng = np.random.default_rng(0)
    failures = []

    def check(name, ok, detail=""):
        print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    # 1. kappa on identical vectors == 1 (incl. half-point grid)
    m = rng.integers(1, 4, 210)
    n_arr = rng.integers(1, 4, 210)
    codes = joint_codes(m, n_arr, m)
    k_jm = PAIRINGS["JM"].kappa(counts_from_codes(codes))[0]
    check("kappa(judge==miri) == 1", np.isclose(k_jm, 1.0))
    med_int = ((m + n_arr) / 2.0)
    j_eq_med = np.where(np.isclose(med_int % 1, 0), med_int, np.nan)
    mask = ~np.isnan(j_eq_med)
    codes2 = joint_codes(m[mask], n_arr[mask], j_eq_med[mask].astype(int))
    k_med = PAIRINGS["JMED"].kappa(counts_from_codes(codes2))[0]
    check("kappa(judge==median where integer) == 1", np.isclose(k_med, 1.0))

    # 2. binary 2x2 classic: table [[20,5],[10,15]] -> kappa = 0.4
    x = np.array([1] * 25 + [2] * 25)
    y = np.array([1] * 20 + [2] * 5 + [1] * 10 + [2] * 15)
    codes3 = joint_codes(x, y, np.ones(50, int))
    k_bin = PAIRINGS["HH"].kappa(counts_from_codes(codes3))[0]
    check("binary kappa == 0.4 (Cohen classic)", np.isclose(k_bin, 0.4), f"got {k_bin:.4f}")

    # 3. quadratic kappa vs independent reference implementation, random data
    a = rng.integers(1, 4, 500)
    b = np.clip(a + rng.integers(-1, 2, 500), 1, 3)
    codes4 = joint_codes(a, b, np.ones(500, int))
    k_fast = PAIRINGS["HH"].kappa(counts_from_codes(codes4))[0]
    k_ref = _reference_weighted_kappa(a, b, power=2)
    check("kappa_w matches independent reference", np.isclose(k_fast, k_ref), f"{k_fast:.6f} vs {k_ref:.6f}")

    # 4. AC2 vs independent reference; defined with a near-constant rater
    c1 = np.ones(210, int)
    c1[:5] = 2
    c2 = np.ones(210, int)
    c2[:12] = 2
    codes5 = joint_codes(c1, c2, np.ones(210, int))
    ac_fast = PAIRINGS["HH"].ac2(counts_from_codes(codes5))[0]
    ac_ref = _reference_ac2(c1, c2)
    check("AC2 matches reference (skewed data)", np.isclose(ac_fast, ac_ref), f"{ac_fast:.6f} vs {ac_ref:.6f}")
    check("AC2 defined and high under near-constant rater", ac_fast > 0.9, f"got {ac_fast:.3f}")
    kap_skew = PAIRINGS["HH"].kappa(counts_from_codes(codes5))[0]
    check("kappa-paradox regime reproduced (kappa << AC2)", kap_skew < 0.6, f"kappa {kap_skew:.3f}")

    # 5. ICC(A,1) on Shrout & Fleiss (1979) data -> ICC(2,1) = 0.29
    sf = np.array(
        [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]],
        dtype=float,
    )
    icc = icc_a1(sf)
    check("ICC(A,1) Shrout-Fleiss == 0.29", abs(icc - 0.29) < 0.005, f"got {icc:.4f}")
    xk2 = sf[:, :2]
    icc_g = icc_a1(xk2)
    icc_v = icc_a1_boot_pairs(xk2[:, 0][None, :], xk2[:, 1][None, :])[0]
    check("vectorised k=2 ICC matches general ICC", np.isclose(icc_g, icc_v), f"{icc_g:.6f} vs {icc_v:.6f}")

    # 6. degenerate bootstrap accounting: constant raters -> kappa NaN
    const_codes = joint_codes(np.ones(210, int), np.ones(210, int), np.ones(210, int))
    kc = PAIRINGS["HH"].kappa(counts_from_codes(const_codes))[0]
    check("kappa NaN when both raters constant", np.isnan(kc))
    bc = bootstrap_counts(const_codes, 100, rng)
    kb = PAIRINGS["HH"].kappa(bc)
    check("degenerate replicates counted", np.isnan(kb).mean() == 1.0)

    # 7. bootstrap counts preserve n and marginals in expectation
    bc2 = bootstrap_counts(codes4, 2000, rng)
    check("bootstrap counts sum to n", np.all(bc2.sum(axis=1) == 500))

    # 8. codebook loads with 13 items
    try:
        item_keys, _ = load_codebook()
        check("codebook has 13 items on 1..3 scale", len(item_keys) == 13)
    except Exception as e:  # noqa: BLE001
        check("codebook loads", False, str(e))

    print()
    if failures:
        print(f"SELF-TEST FAILED: {failures}")
        return 1
    print("All self-tests passed.")
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Tier-3 judge-validation gate (module 1)")
    parser.add_argument("--n-boot", type=int, default=10000, help="bootstrap iterations (default 10000)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--force", action="store_true", help="overwrite existing outputs")
    parser.add_argument("--self-test", action="store_true", help="run built-in checks and exit")
    args = parser.parse_args()

    if args.self_test:
        sys.exit(self_test())

    outdir = args.output_dir
    if outdir.exists() and any(outdir.iterdir()) and not args.force:
        sys.exit(f"Output dir {outdir} is not empty — use --force to overwrite.")
    outdir.mkdir(parents=True, exist_ok=True)

    item_keys, labels = load_codebook()
    scores, integrity = assemble(item_keys)
    print(f"Assembled: {integrity['join_matched']}/{N_SAMPLES} samples joined, "
          f"{integrity['n_judge_rows']} judge rows verified.")

    rng = np.random.default_rng(args.seed)
    results = {}
    for key in item_keys:
        s = analyse_item(scores[key], args.n_boot, rng)
        status, rule_path, reasons, flags = apply_gate(s)
        s.update(status=status, rule_path=rule_path, reasons=reasons, flags=flags)
        results[key] = s
        print(f"  {key:<36} kappa_med={s['kappa_JMED']:.2f} "
              f"[{s['kappa_JMED_ci'][0]:.2f},{s['kappa_JMED_ci'][1]:.2f}]  -> {status}"
              + (f"  ({';'.join(flags)})" if flags else ""))

    comp = analyse_composite(scores, item_keys, args.n_boot, rng)
    comp_sens = analyse_composite(
        scores, item_keys, args.n_boot, rng, exclude=("item_06_nonjudgmental_language",)
    )

    summary = write_outputs(results, comp, comp_sens, labels, integrity, outdir, args)

    print()
    print(summary)
    print(f"\nOutputs written to {outdir}")


if __name__ == "__main__":
    main()
