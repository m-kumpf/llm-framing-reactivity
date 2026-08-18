#!/usr/bin/env python3
"""
Tier-3 inference ("module 2").

Tests framing effects on the gate-confirmed Tier-3 endpoints using the full
4,200-response DeepSeek-V4-Flash judge dataset. Endpoints are taken from
output/analyse_tier3_judge/gate_verdicts.json — NEVER hard-coded:

    primary   = relationship-oriented composite, if gate-confirmed
    secondary = gate-confirmed single items
    fallback  = items 01, 02 labelled exploratory, if nothing is confirmed

Design (pre-specified 2026-07-15, after gate review):
  - Unit of analysis: (model x framing x scenario) cell means — 420 cells,
    10 runs aggregated against pseudoreplication.
  - Per endpoint, one pooled two-way analysis: model main effect, framing
    main effect, model x framing interaction (scenario as crossed blocking
    factor). No per-model p-values; per-model descriptives only.
  - Main effects: restricted permutation of labels within strata
    (framing permuted within model x scenario; model permuted within
    framing x scenario), 10,000 permutations, no Gaussian p-values.
  - Interaction: Freedman-Lane residual permutation — residuals of the
    additive model (model + framing + scenario) permuted within scenario
    strata, F recomputed on reconstructed responses.
  - Holm correction across ALL endpoint x effect tests (one family).
  - Effect sizes for single items (not the composite), response level:
    proportional-odds ORs and boundary-honest binomial P(score<3) ORs,
    framing effect-coded so each OR is vs. the model's OWN grand mean
    (per-model fits). These are effect sizes; inference comes from the
    permutation tests.

Input:  run-judge-tier3-deepseek/judge_*.csv
        output/analyse_tier3_judge/gate_verdicts.json
        codebook.json
Output: output/analyse_tier3/
            endpoints.json            — resolved endpoints + roles
            cell_means.csv            — 420 rows x endpoints
            permutation_tests.csv     — endpoint x effect: F, partial eta2, p_perm, p_holm
            model_framing_means.csv   — per endpoint x model x framing: mean, sd, delta
            effects_proportional_odds.csv — item x model x framing: PO OR vs own-model mean
            effects_binomial_ceiling.csv  — item x model x framing: OR of P(score<3)
            analysis_summary.txt, run_metadata.json

Usage:
    python analyse_tier3.py
    python analyse_tier3.py --n-perm 2000    # faster, exploratory only
    python analyse_tier3.py --force
    python analyse_tier3.py --self-test
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

BASE = Path(__file__).resolve().parents[1]
JUDGE_DIR = BASE / "run-judge-tier3-deepseek"
GATE_JSON = BASE / "output" / "analyse_tier3_judge" / "gate_verdicts.json"
CODEBOOK = BASE / "codebook.json"
DEFAULT_OUTDIR = BASE / "output" / "analyse_tier3"


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
# fixed display/output row order for models
MODEL_ORDER = [
    "Claude Sonnet 4.6", "GPT-5.3", "Gemini 3.1 Pro", "Kimi K2.5",
    "Qwen 3.5", "MiniMax M2.7", "GLM 5",
]
FRAMING_ID_SHORT = {
    "A": "Anxious", "B": "Stoic", "C": "Angry",
    "D": "Hyper-rational", "E": "Overwhelmed", "F": "Humor",
}
N_MODELS, N_FRAMINGS, N_SCENARIOS, N_RUNS = 7, 6, 10, 10
N_CELLS = N_MODELS * N_FRAMINGS * N_SCENARIOS
FALLBACK_ITEM_NS = [1, 2]

# Ruben, Blanch-Hartigan & Hall (2026, JGIM) chatbot composites. The 6-item
# relationship-oriented composite (gate config) is PRIMARY; the 5-item
# variant (item 6 removed) is an application-specific sensitivity check
# only. Item 9 is reverse-scored inside the conscientious composite.
COMPOSITE_5ITEM_NS = [1, 2, 3, 7, 10]
RUBEN_OTHER_COMPOSITES = {
    "composite_conscientious": {"ns": [9, 13], "reverse": [9]},
    "composite_guiding": {"ns": [5, 12], "reverse": []},
    "composite_technical": {"ns": [8, 11], "reverse": []},
}

EFFECTS = ["model", "framing", "interaction"]


# ═══════════════════════════════════════════════════════════════════════════════
# BALANCED THREE-FACTOR ANOVA ON CELL MEANS (vectorised over a batch axis)
# ═══════════════════════════════════════════════════════════════════════════════
# Y has shape (..., M, F, S). The design is fully crossed and balanced, so
# sums of squares decompose orthogonally. The error term pools all
# scenario-involving interactions (m:s, f:s, m:f:s), df = 369.

DF = {"model": N_MODELS - 1, "framing": N_FRAMINGS - 1, "scenario": N_SCENARIOS - 1}
DF["interaction"] = DF["model"] * DF["framing"]
DF["resid"] = (N_CELLS - 1) - DF["model"] - DF["framing"] - DF["scenario"] - DF["interaction"]


def anova_ss(Y):
    """Return dict of SS arrays (batch-shaped) for the balanced design."""
    Y = np.asarray(Y, dtype=float)
    grand = Y.mean(axis=(-3, -2, -1), keepdims=True)
    m_mean = Y.mean(axis=(-2, -1), keepdims=True)
    f_mean = Y.mean(axis=(-3, -1), keepdims=True)
    s_mean = Y.mean(axis=(-3, -2), keepdims=True)
    mf_mean = Y.mean(axis=-1, keepdims=True)
    ss = {}
    ss["model"] = (N_FRAMINGS * N_SCENARIOS) * ((m_mean - grand) ** 2).sum(axis=(-3, -2, -1))
    ss["framing"] = (N_MODELS * N_SCENARIOS) * ((f_mean - grand) ** 2).sum(axis=(-3, -2, -1))
    ss["scenario"] = (N_MODELS * N_FRAMINGS) * ((s_mean - grand) ** 2).sum(axis=(-3, -2, -1))
    ss["interaction"] = N_SCENARIOS * (
        (mf_mean - m_mean - f_mean + grand) ** 2
    ).sum(axis=(-3, -2, -1))
    ss["total"] = ((Y - grand) ** 2).sum(axis=(-3, -2, -1))
    ss["resid"] = ss["total"] - ss["model"] - ss["framing"] - ss["scenario"] - ss["interaction"]
    return ss


def anova_f(Y):
    """F statistics and partial eta^2 for model, framing, interaction."""
    ss = anova_ss(Y)
    out = {}
    for eff in EFFECTS:
        ms_e = ss[eff] / DF[eff]
        ms_r = ss["resid"] / DF["resid"]
        out[f"F_{eff}"] = ms_e / ms_r
        out[f"eta2_{eff}"] = ss[eff] / (ss[eff] + ss["resid"])
    return out


def additive_fit(Y):
    """Fitted values of the additive model model + framing + scenario (balanced)."""
    grand = Y.mean()
    m_mean = Y.mean(axis=(1, 2), keepdims=True)
    f_mean = Y.mean(axis=(0, 2), keepdims=True)
    s_mean = Y.mean(axis=(0, 1), keepdims=True)
    return m_mean + f_mean + s_mean - 2 * grand


def permutation_tests(Y, n_perm, rng):
    """Restricted-permutation p-values for the three effects on Y (M,F,S)."""
    obs = anova_f(Y[None, ...])
    results = {}

    # framing main effect: permute framing labels within model x scenario strata
    batch = np.broadcast_to(Y, (n_perm,) + Y.shape).copy()
    perm = rng.permuted(batch, axis=2)  # batch dims are (P, M, F, S) -> framing is axis 2
    f_null = anova_f(perm)["F_framing"]
    results["framing"] = (1 + np.sum(f_null >= obs["F_framing"][0])) / (1 + n_perm)

    # model main effect: permute model labels within framing x scenario strata
    batch = np.broadcast_to(Y, (n_perm,) + Y.shape).copy()
    perm = rng.permuted(batch, axis=1)  # axis 1 = model
    m_null = anova_f(perm)["F_model"]
    results["model"] = (1 + np.sum(m_null >= obs["F_model"][0])) / (1 + n_perm)

    # interaction: Freedman-Lane — permute additive-model residuals within
    # scenario strata, reconstruct, recompute F_interaction
    fitted = additive_fit(Y)
    resid = Y - fitted
    flat = np.broadcast_to(
        resid.reshape(N_MODELS * N_FRAMINGS, N_SCENARIOS),
        (n_perm, N_MODELS * N_FRAMINGS, N_SCENARIOS),
    ).copy()
    rperm = rng.permuted(flat, axis=1).reshape(n_perm, N_MODELS, N_FRAMINGS, N_SCENARIOS)
    y_star = fitted[None, ...] + rperm
    i_null = anova_f(y_star)["F_interaction"]
    results["interaction"] = (1 + np.sum(i_null >= obs["F_interaction"][0])) / (1 + n_perm)

    stats = {eff: {"F": float(obs[f"F_{eff}"][0]), "eta2_partial": float(obs[f"eta2_{eff}"][0]),
                   "p_perm": float(results[eff])} for eff in EFFECTS}
    return stats


def holm(pvals):
    """Holm step-down adjusted p-values (dict key -> p)."""
    keys = list(pvals.keys())
    p = np.array([pvals[k] for k in keys])
    order = np.argsort(p)
    n = len(p)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return {k: float(adj[i]) for i, k in enumerate(keys)}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA / ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════


def load_judge(item_keys):
    files = sorted(JUDGE_DIR.glob("judge_*.csv"))
    assert len(files) == 7, f"expected 7 judge CSVs, found {len(files)}"
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    assert len(df) == N_MODELS * N_FRAMINGS * N_SCENARIOS * N_RUNS
    assert (df["status"] == "ok").all()
    assert df[item_keys].isin([1, 2, 3]).all().all()
    unknown = set(df["model"]) - set(MODEL_SHORT)
    assert not unknown, f"unmapped model ids: {unknown}"
    df["model_short"] = df["model"].map(MODEL_SHORT)
    df["framing_short"] = df["framing_id"].map(FRAMING_ID_SHORT)
    counts = df.groupby(["model", "framing_id", "scenario_id"]).size()
    assert (counts == N_RUNS).all(), "unbalanced cells detected"
    return df, files


def resolve_endpoints(item_keys):
    """Read gate verdicts; return (endpoints dict, gate payload)."""
    with open(GATE_JSON) as f:
        gate = json.load(f)
    confirmed = [k for k in item_keys if gate["items"][k]["status"] == "confirmatory"]
    composite_ok = gate["composite"]["status"] == "confirmatory"
    comp_ns = gate["config"]["composite_items"]
    comp_keys = [k for k in item_keys if int(k.split("_")[1]) in comp_ns]

    endpoints = {}
    if composite_ok:
        endpoints["composite_relationship"] = {
            "role": "primary", "kind": "composite", "items": comp_keys}
    for k in confirmed:
        endpoints[k] = {"role": "secondary", "kind": "item", "items": [k]}
    if not endpoints:
        for n in FALLBACK_ITEM_NS:
            k = [key for key in item_keys if int(key.split("_")[1]) == n][0]
            endpoints[k] = {"role": "exploratory_fallback", "kind": "item", "items": [k]}

    # Sensitivity + Ruben secondary composites — folded into the SAME Holm
    # family as the primary endpoints.
    if composite_ok:
        endpoints["composite_relationship_5item"] = {
            "role": "sensitivity", "kind": "composite",
            "items": [k for k in item_keys
                      if int(k.split("_")[1]) in COMPOSITE_5ITEM_NS]}
    for name, spec in RUBEN_OTHER_COMPOSITES.items():
        keys = [k for k in item_keys if int(k.split("_")[1]) in spec["ns"]]
        statuses = [gate["items"][k]["status"] for k in keys]
        role = ("secondary_ruben" if all(s == "confirmatory" for s in statuses)
                else "exploratory_ruben")
        endpoints[name] = {
            "role": role, "kind": "composite", "items": keys,
            "reverse": [k for k in keys
                        if int(k.split("_")[1]) in spec["reverse"]]}
    return endpoints, gate


def add_composite_columns(df, endpoints):
    """Compute composite endpoint columns on the response-level frame."""
    for name, spec in endpoints.items():
        if spec["kind"] != "composite" or name in df.columns:
            continue
        vals = df[spec["items"]].astype(float).copy()
        for rk in spec.get("reverse", []):
            vals[rk] = 4 - vals[rk]  # reverse 1-3 ordinal
        df[name] = vals.mean(axis=1)


def cell_mean_array(df, value_col, model_order, framing_ids, scenario_ids):
    """Return (M, F, S) array of cell means for value_col."""
    cells = df.groupby(["model_short", "framing_id", "scenario_id"])[value_col].mean()
    Y = np.empty((N_MODELS, N_FRAMINGS, N_SCENARIOS))
    for mi, m in enumerate(model_order):
        for fi, f in enumerate(framing_ids):
            for si, s in enumerate(scenario_ids):
                Y[mi, fi, si] = cells.loc[(m, f, s)]
    assert not np.isnan(Y).any()
    return Y


# ═══════════════════════════════════════════════════════════════════════════════
# EFFECT SIZES (single items, response level, per model, framing effect-coded)
# ═══════════════════════════════════════════════════════════════════════════════


def sum_coded_design(framing_ids, levels):
    """Sum (deviation) coding: columns for levels[:-1]; last level = -1 row."""
    X = np.zeros((len(framing_ids), len(levels) - 1))
    for i, f in enumerate(framing_ids):
        j = levels.index(f)
        if j < len(levels) - 1:
            X[i, j] = 1.0
        else:
            X[i, :] = -1.0
    return X


def _recover_last(params, cov):
    """Effect coding: coefficient and SE of the omitted last level (= -sum)."""
    beta_last = -params.sum()
    var_last = float(np.ones(len(params)) @ cov @ np.ones(len(params)))
    return beta_last, np.sqrt(var_last)


def po_or_per_model(sub, item, levels):
    """Proportional-odds ORs per framing vs own-model grand mean. Returns rows."""
    rows = []
    y = sub[item].astype(int)
    X = sum_coded_design(sub["framing_id"].tolist(), levels)
    if y.nunique() < 2:
        return [{"framing_id": f, "OR": np.nan, "OR_lo": np.nan, "OR_hi": np.nan,
                 "note": "degenerate: single observed category"} for f in levels]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            endog = pd.Series(pd.Categorical(y, ordered=True), name="score")
            mod = OrderedModel(endog, X, distr="logit")
            res = mod.fit(method="bfgs", maxiter=200, disp=False)
        if not res.mle_retvals.get("converged", True) or np.isnan(res.params).any():
            raise RuntimeError("nonconvergence/separation")
        k = len(levels) - 1
        betas = np.asarray(res.params[:k])
        cov = np.asarray(res.cov_params())[:k, :k]
        ses = np.sqrt(np.diag(cov))
        b_last, se_last = _recover_last(betas, cov)
        all_b = np.append(betas, b_last)
        all_se = np.append(ses, se_last)
        for f, b, se in zip(levels, all_b, all_se):
            note = "quasi-separation: OR unstable" if abs(b) > 5 else ""
            rows.append({"framing_id": f, "OR": float(np.exp(b)),
                         "OR_lo": float(np.exp(b - 1.96 * se)),
                         "OR_hi": float(np.exp(b + 1.96 * se)), "note": note})
    except Exception as e:  # noqa: BLE001
        rows = [{"framing_id": f, "OR": np.nan, "OR_lo": np.nan, "OR_hi": np.nan,
                 "note": f"fit failed: {type(e).__name__}"} for f in levels]
    return rows


def binom_or_per_model(sub, item, levels):
    """Binomial GLM ORs for P(score < 3) per framing vs own-model grand mean.

    Cluster-robust (scenario) CIs. OR > 1: framing pushes scores BELOW ceiling
    more often than the model's average.
    """
    rows = []
    y = (sub[item].astype(int) < 3).astype(float)
    X = sm.add_constant(sum_coded_design(sub["framing_id"].tolist(), levels))
    if y.nunique() < 2:
        return [{"framing_id": f, "OR": np.nan, "OR_lo": np.nan, "OR_hi": np.nan,
                 "note": "degenerate: never (or always) at ceiling"} for f in levels]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.GLM(y, X, family=sm.families.Binomial()).fit(
                cov_type="cluster", cov_kwds={"groups": sub["scenario_id"].to_numpy()})
        k = len(levels) - 1
        betas = np.asarray(res.params[1:])
        cov = np.asarray(res.cov_params())[1:, 1:]
        ses = np.sqrt(np.diag(cov))
        b_last, se_last = _recover_last(betas, cov)
        all_b = np.append(betas, b_last)
        all_se = np.append(ses, se_last)
        for f, b, se in zip(levels, all_b, all_se):
            note = "quasi-separation: OR unstable" if abs(b) > 5 else ""
            rows.append({"framing_id": f, "OR": float(np.exp(b)),
                         "OR_lo": float(np.exp(b - 1.96 * se)),
                         "OR_hi": float(np.exp(b + 1.96 * se)), "note": note})
    except Exception as e:  # noqa: BLE001
        rows = [{"framing_id": f, "OR": np.nan, "OR_lo": np.nan, "OR_hi": np.nan,
                 "note": f"fit failed: {type(e).__name__}"} for f in levels]
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# ITEM-LEVEL PCA — empirical membership check for item 6
# ═══════════════════════════════════════════════════════════════════════════════


def item_pca_core(X, item_keys, rel_ns=COMPOSITE_5ITEM_NS, n_report=4):
    """PCA (correlation matrix) of item cell means.

    Mirrors how Ruben et al. derived the composites. Identifies the
    component that carries the relationship items OTHER than item 6
    (rel_ns), sign-fixed so their mean loading is positive, and returns
    (loadings [k x n_report], var_explained, rel_comp_index).
    """
    X = np.asarray(X, dtype=float)
    mu, sd = X.mean(axis=0), X.std(axis=0, ddof=1)
    sd[sd < 1e-10] = 1.0
    Z = (X - mu) / sd
    C = np.corrcoef(Z, rowvar=False)
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evals, evecs = np.maximum(evals[order], 0), evecs[:, order]
    var_exp = evals / evals.sum()
    loadings = evecs * np.sqrt(evals)
    n_report = min(n_report, loadings.shape[1])
    rel_idx = [i for i, k in enumerate(item_keys)
               if int(k.split("_")[1]) in rel_ns]
    mean_abs = [np.abs(loadings[rel_idx, c]).mean() for c in range(n_report)]
    rel_comp = int(np.argmax(mean_abs))
    if loadings[rel_idx, rel_comp].mean() < 0:
        loadings[:, rel_comp] *= -1
    return loadings[:, :n_report], var_exp[:n_report], rel_comp


def item_pca_membership(df, item_keys, outdir):
    """Run the 13-item PCA on the 420 cell means; emit loadings CSV."""
    cells = df.groupby(["model", "framing_id", "scenario_id"])[item_keys].mean()
    assert len(cells) == N_CELLS
    loadings, var_exp, rel_comp = item_pca_core(cells.to_numpy(), item_keys)

    tab = pd.DataFrame(loadings,
                       columns=[f"PC{i + 1}" for i in range(loadings.shape[1])],
                       index=item_keys)
    tab["ruben_relationship_member"] = [
        int(k.split("_")[1]) in [1, 2, 3, 6, 7, 10] for k in item_keys]
    tab.to_csv(outdir / "item_pca_loadings.csv")

    item6 = "item_06_nonjudgmental_language"
    i6 = item_keys.index(item6)
    result = {
        "n_cells": int(N_CELLS),
        "variance_explained": [round(float(v), 4) for v in var_exp],
        "relationship_component": f"PC{rel_comp + 1}",
        "identified_by": "max mean |loading| of items 1,2,3,7,10",
        "item_06_loading_on_relationship_component":
            round(float(loadings[i6, rel_comp]), 4),
        "mean_loading_items_1_2_3_7_10": round(float(np.mean(
            [loadings[item_keys.index(k), rel_comp] for k in item_keys
             if int(k.split("_")[1]) in COMPOSITE_5ITEM_NS])), 4),
    }

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-TESTS
# ═══════════════════════════════════════════════════════════════════════════════


def self_test():
    rng = np.random.default_rng(0)
    failures = []

    def check(name, ok, detail=""):
        print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    # 1. balanced ANOVA decomposition vs statsmodels OLS anova_lm
    Y = rng.normal(size=(N_MODELS, N_FRAMINGS, N_SCENARIOS))
    mi, fi, si = np.meshgrid(range(N_MODELS), range(N_FRAMINGS), range(N_SCENARIOS), indexing="ij")
    df = pd.DataFrame({"y": Y.ravel(), "m": mi.ravel().astype(str),
                       "f": fi.ravel().astype(str), "s": si.ravel().astype(str)})
    ols = sm.formula.ols("y ~ C(m) + C(f) + C(s) + C(m):C(f)", data=df).fit()
    tab = sm.stats.anova_lm(ols, typ=2)
    ours = anova_f(Y[None, ...])
    ok = (np.isclose(ours["F_model"][0], tab.loc["C(m)", "F"])
          and np.isclose(ours["F_framing"][0], tab.loc["C(f)", "F"])
          and np.isclose(ours["F_interaction"][0], tab.loc["C(m):C(f)", "F"]))
    check("ANOVA F matches statsmodels anova_lm", ok,
          f"model {ours['F_model'][0]:.4f} vs {tab.loc['C(m)', 'F']:.4f}")

    # 2. residual df consistent with OLS
    check("residual df == 369", DF["resid"] == int(ols.df_resid), f"{DF['resid']} vs {int(ols.df_resid)}")

    # 3. permutation p under pure noise is not extreme (smoke, fixed seed)
    stats = permutation_tests(Y, 500, np.random.default_rng(1))
    ok = all(stats[eff]["p_perm"] > 0.01 for eff in EFFECTS)
    check("null-data permutation p not extreme", ok,
          str({e: round(stats[e]["p_perm"], 3) for e in EFFECTS}))

    # 4. injected framing effect detected; interaction stays null
    Y2 = Y + np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5])[None, :, None]
    stats2 = permutation_tests(Y2, 500, np.random.default_rng(2))
    check("injected framing main effect detected", stats2["framing"]["p_perm"] < 0.01,
          f"p = {stats2['framing']['p_perm']:.4f}")
    check("no spurious interaction under additive truth",
          stats2["interaction"]["p_perm"] > 0.01, f"p = {stats2['interaction']['p_perm']:.4f}")

    # 5. injected interaction detected by Freedman-Lane
    Y3 = Y.copy()
    Y3[0, 0, :] += 2.0
    Y3[1, 1, :] += 2.0
    stats3 = permutation_tests(Y3, 500, np.random.default_rng(3))
    check("injected interaction detected", stats3["interaction"]["p_perm"] < 0.01,
          f"p = {stats3['interaction']['p_perm']:.4f}")

    # 6. Holm: known worked example (p = .01,.04,.03,.005 -> .03,.06,.06,.02)
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005})
    expect = {"a": 0.03, "b": 0.06, "c": 0.06, "d": 0.02}
    check("Holm adjustment matches worked example",
          all(np.isclose(adj[k], expect[k]) for k in expect), str(adj))

    # 7. sum coding: recovered last-level coefficient equals -sum of others
    levels = list("ABCDEF")
    X = sum_coded_design(["A", "B", "F", "C"], levels)
    check("sum coding rows correct",
          np.array_equal(X[2], -np.ones(5)) and X[0, 0] == 1 and X[3, 2] == 1)
    betas = rng.normal(size=5)
    cov = np.eye(5) * 0.04
    b_last, se_last = _recover_last(betas, cov)
    check("omitted-level recovery: beta=-sum, SE=sqrt(1'Cov1)",
          np.isclose(b_last, -betas.sum()) and np.isclose(se_last, np.sqrt(5 * 0.04)))

    # 8. endpoint resolution logic on synthetic gate payloads
    item_keys = [f"item_{n:02d}_x" for n in range(1, 14)]

    def fake_gate(comp_status, item_statuses):
        return {"config": {"composite_items": [1, 2]},
                "items": {k: {"status": item_statuses.get(k, "exploratory")} for k in item_keys},
                "composite": {"status": comp_status}}

    import json as _json
    import tempfile

    def resolve_with(payload):
        global GATE_JSON
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            _json.dump(payload, tf)
            tmp = tf.name
        old = GATE_JSON
        try:
            globals()["GATE_JSON"] = Path(tmp)
            return resolve_endpoints(item_keys)[0]
        finally:
            globals()["GATE_JSON"] = old

    eps = resolve_with(fake_gate("confirmatory", {"item_03_x": "confirmatory"}))
    check("endpoints: composite primary + confirmed item",
          eps.get("composite_relationship", {}).get("role") == "primary"
          and eps.get("item_03_x", {}).get("role") == "secondary")
    eps2 = resolve_with(fake_gate("exploratory", {}))
    fallback_items = {k for k, v in eps2.items()
                      if v["role"] == "exploratory_fallback"}
    ruben_ok = all(eps2[n]["role"] in ("secondary_ruben", "exploratory_ruben")
                   for n in RUBEN_OTHER_COMPOSITES)
    check("endpoints: fallback 01/02 exploratory when nothing confirmed "
          "(+ Ruben composites, no relationship composite)",
          fallback_items == {"item_01_x", "item_02_x"} and ruben_ok
          and "composite_relationship" not in eps2
          and "composite_relationship_5item" not in eps2)

    # 9. composite reversal: 4 - x maps 1->3, 2->2, 3->1
    fake = pd.DataFrame({"item_09_hurried_impression": [1, 2, 3],
                         "item_13_collaborative_language": [2, 2, 2]})
    eps_rev = {"composite_conscientious": {
        "kind": "composite", "items": list(fake.columns),
        "reverse": ["item_09_hurried_impression"]}}
    add_composite_columns(fake, eps_rev)
    check("item-9 reversal in conscientious composite",
          np.allclose(fake["composite_conscientious"], [2.5, 2.0, 1.5]),
          str(fake["composite_conscientious"].tolist()))

    # 10. item PCA membership: correlated block is identified and a variable
    # correlated with the block loads positively on that component
    keys13 = [f"item_{n:02d}_x" for n in range(1, 14)]
    base = rng.normal(size=(420, 1))
    Xs = rng.normal(size=(420, 13)) * 0.6
    for n in [1, 2, 3, 7, 10, 6]:  # block: rel items + a correlated "item 6"
        Xs[:, n - 1] += base[:, 0]
    lo, ve, comp = item_pca_core(Xs, keys13)
    i6_load = lo[5, comp]
    check("item PCA: correlated item 6 loads on relationship component",
          i6_load > 0.5, f"loading = {i6_load:.3f} on PC{comp + 1}")

    # 11. PO model sanity: strong framing shift yields OR > 1 for shifted framing
    n = 600
    fr = np.tile(list("ABCDEF"), n // 6)
    score = rng.integers(1, 3, n)  # others: mix of 1s and 2s
    mask_a = fr == "A"
    score[mask_a] = rng.choice([1, 2, 3], size=mask_a.sum(), p=[0.05, 0.25, 0.70])  # strong, not separated
    sub = pd.DataFrame({"framing_id": fr, "item": score, "scenario_id": np.tile(range(10), 60)})
    rows = po_or_per_model(sub, "item", list("ABCDEF"))
    or_a = [r for r in rows if r["framing_id"] == "A"][0]["OR"]
    check("PO OR: ceiling-shifted framing gets OR >> 1", or_a > 2, f"OR_A = {or_a:.2f}")

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
    parser = argparse.ArgumentParser(description="Tier-3 inference (module 2)")
    parser.add_argument("--n-perm", type=int, default=10000, help="permutations (default 10000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        sys.exit(self_test())

    outdir = args.output_dir
    if outdir.exists() and any(outdir.iterdir()) and not args.force:
        sys.exit(f"Output dir {outdir} is not empty — use --force to overwrite.")
    outdir.mkdir(parents=True, exist_ok=True)

    with open(CODEBOOK) as f:
        item_keys = [it["key"] for it in json.load(f)["items"]]
    endpoints, gate = resolve_endpoints(item_keys)
    df, judge_files = load_judge(item_keys)

    model_order = MODEL_ORDER
    framing_ids = sorted(df["framing_id"].unique())          # A..F
    scenario_ids = sorted(df["scenario_id"].unique())        # S01..S10

    add_composite_columns(df, endpoints)

    print(f"Endpoints from gate ({GATE_JSON.name}):")
    for name, spec in endpoints.items():
        print(f"  {name:<36} role={spec['role']}")

    rng = np.random.default_rng(args.seed)
    perm_results, cell_frames, mf_rows = {}, {}, []
    for name, spec in endpoints.items():
        col = name
        Y = cell_mean_array(df, col, model_order, framing_ids, scenario_ids)
        cell_frames[name] = Y
        stats = permutation_tests(Y, args.n_perm, rng)
        perm_results[name] = stats
        print(f"  {name:<36} F_model={stats['model']['F']:.1f} "
              f"F_framing={stats['framing']['F']:.1f} F_int={stats['interaction']['F']:.1f}")

        mf = Y.mean(axis=2)          # (M, F) means over scenarios
        sd = Y.std(axis=2, ddof=1)
        base = mf.mean(axis=1, keepdims=True)
        for mi, m in enumerate(model_order):
            for fi, fid in enumerate(framing_ids):
                mf_rows.append({
                    "endpoint": name, "model": m,
                    "framing": FRAMING_ID_SHORT[fid], "framing_id": fid,
                    "mean": mf[mi, fi], "sd_across_scenarios": sd[mi, fi],
                    "model_baseline": base[mi, 0], "delta": mf[mi, fi] - base[mi, 0],
                })

    # Holm in TWO separate families: the CONFIRMATORY family (primary
    # composite + gate-confirmed single items) and a SUPPLEMENTARY family
    # (5-item sensitivity + Ruben composites), so the supplementary
    # endpoints cannot affect the confirmatory endpoints' p_holm.
    CONFIRMATORY_ROLES = ("primary", "secondary", "exploratory_fallback")
    families = {"confirmatory": [e for e in perm_results
                                 if endpoints[e]["role"] in CONFIRMATORY_ROLES],
                "sensitivity_ruben": [e for e in perm_results
                                      if endpoints[e]["role"]
                                      not in CONFIRMATORY_ROLES]}
    family_sizes = {}
    for fam_name, eps in families.items():
        fam = {f"{e}::{eff}": perm_results[e][eff]["p_perm"]
               for e in eps for eff in EFFECTS}
        family_sizes[fam_name] = len(fam)
        if not fam:
            continue
        adjusted = holm(fam)
        for key, p_adj in adjusted.items():
            e, eff = key.split("::")
            perm_results[e][eff]["p_holm"] = p_adj
            perm_results[e][eff]["holm_family"] = fam_name

    # Effect sizes for single-item endpoints (response level, per model)
    single_items = [n for n, s in endpoints.items() if s["kind"] == "item"]
    po_rows, bin_rows = [], []
    for item in single_items:
        for m in model_order:
            sub = df[df["model_short"] == m]
            for r in po_or_per_model(sub, item, framing_ids):
                po_rows.append({"item": item, "model": m,
                                "framing": FRAMING_ID_SHORT[r["framing_id"]], **r})
            for r in binom_or_per_model(sub, item, framing_ids):
                bin_rows.append({"item": item, "model": m,
                                 "framing": FRAMING_ID_SHORT[r["framing_id"]], **r})

    # ── outputs ──────────────────────────────────────────────────────────────
    cm_rows = []
    for name, Y in cell_frames.items():
        for mi, m in enumerate(model_order):
            for fi, fid in enumerate(framing_ids):
                for si, s in enumerate(scenario_ids):
                    cm_rows.append({"endpoint": name, "model": m, "framing_id": fid,
                                    "scenario_id": s, "cell_mean": Y[mi, fi, si]})
    pd.DataFrame(cm_rows).to_csv(outdir / "cell_means.csv", index=False)

    pt_rows = [{"endpoint": e, "role": endpoints[e]["role"], "effect": eff,
                "F": perm_results[e][eff]["F"], "df1": DF[eff], "df2": DF["resid"],
                "eta2_partial": perm_results[e][eff]["eta2_partial"],
                "p_perm": perm_results[e][eff]["p_perm"],
                "p_holm": perm_results[e][eff]["p_holm"],
                "holm_family": perm_results[e][eff]["holm_family"],
                "significant_005": perm_results[e][eff]["p_holm"] < 0.05}
               for e in perm_results for eff in EFFECTS]
    pd.DataFrame(pt_rows).to_csv(outdir / "permutation_tests.csv", index=False)

    pd.DataFrame(mf_rows).to_csv(outdir / "model_framing_means.csv", index=False)
    pd.DataFrame(po_rows).to_csv(outdir / "effects_proportional_odds.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(outdir / "effects_binomial_ceiling.csv", index=False)

    with open(outdir / "endpoints.json", "w") as f:
        json.dump({"source_gate": _rel(GATE_JSON), "endpoints": endpoints,
                   "holm_families": {
                       name: {"endpoints": eps,
                              "n_tests": family_sizes[name]}
                       for name, eps in families.items()},
                   "gate_generated_utc": gate.get("generated_utc")}, f, indent=2)

    pca_result = item_pca_membership(df, item_keys, outdir)

    lines = []
    lines.append("Tier-3 inference (module 2) — analysis summary")
    lines.append("=" * 70)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines.append(f"Generated (UTC): {now}")
    lines.append(f"Permutations: {args.n_perm}, seed {args.seed}; TWO Holm "
                 f"families: confirmatory ({family_sizes['confirmatory']} "
                 f"tests) and sensitivity/Ruben "
                 f"({family_sizes['sensitivity_ruben']} tests) — corrected "
                 "separately so supplementary endpoints cannot affect "
                 "confirmatory p_holm")
    lines.append(f"Unit: {N_CELLS} model x framing x scenario cell means "
                 f"({N_RUNS} runs aggregated)")
    lines.append("")
    lines.append(f"{'endpoint':<36}{'effect':<14}{'F':>8}{'eta2p':>8}{'p_perm':>10}{'p_holm':>10}  sig")
    lines.append("-" * 92)
    supp_started = False
    for e in perm_results:
        if (not supp_started
                and endpoints[e]["role"] not in CONFIRMATORY_ROLES):
            lines.append("--- separate Holm family: sensitivity / Ruben "
                         "composites ---")
            supp_started = True
        for eff in EFFECTS:
            s = perm_results[e][eff]
            lines.append(f"{e:<36}{eff:<14}{s['F']:>8.1f}{s['eta2_partial']:>8.3f}"
                         f"{s['p_perm']:>10.4f}{s['p_holm']:>10.4f}  "
                         f"{'*' if s['p_holm'] < 0.05 else ''}")
    lines.append("")
    n_po_fail = sum(1 for r in po_rows if r["note"])
    n_bin_fail = sum(1 for r in bin_rows if r["note"])
    lines.append(f"Effect-size fits: PO {len(po_rows) - n_po_fail}/{len(po_rows)} ok, "
                 f"binomial {len(bin_rows) - n_bin_fail}/{len(bin_rows)} ok "
                 f"(failures/degenerates carry notes in the CSVs)")
    lines.append("")
    lines.append("Composite policy: the 6-item Ruben (2026) relationship-oriented")
    lines.append("composite [items 1,2,3,6,7,10] is PRIMARY. Item 6 is retained per")
    lines.append("the validated construct; its human-human reliability was low in")
    lines.append("this application (kappa_w H-H = .10). The 5-item variant (item 6")
    lines.append("removed) is an application-specific sensitivity check. Empirical")
    lines.append("membership (13-item PCA over the 420 cell means): item 6 loads")
    lines.append(f"{pca_result['item_06_loading_on_relationship_component']:+.2f} on "
                 f"{pca_result['relationship_component']}, the component carrying items "
                 f"1,2,3,7,10 (their mean loading "
                 f"{pca_result['mean_loading_items_1_2_3_7_10']:+.2f}).")
    summary = "\n".join(lines)
    (outdir / "analysis_summary.txt").write_text(summary + "\n")

    meta = {
        "script": "analyse_tier3.py", "timestamp_utc": now,
        "seed": args.seed, "n_perm": args.n_perm,
        "sources": {"judge_files": [f.name for f in judge_files],
                    "gate_verdicts": _rel(GATE_JSON), "codebook": _rel(CODEBOOK)},
        "endpoints": {k: v["role"] for k, v in endpoints.items()},
        "design": "420 cell means; restricted permutation (main effects), "
                  "Freedman-Lane within-scenario (interaction); Holm in two "
                  "separate families: confirmatory "
                  f"({family_sizes['confirmatory']} tests: primary composite "
                  "+ gate-confirmed items) and sensitivity/Ruben "
                  f"({family_sizes['sensitivity_ruben']} tests), so "
                  "supplementary endpoints cannot affect confirmatory p_holm",
        "composite_policy": {
            "primary": "6-item Ruben (2026) relationship-oriented composite "
                       "[1,2,3,6,7,10]; item 6 retained per the validated "
                       "construct despite low human-human reliability in this "
                       "application (kappa_w H-H = .10)",
            "sensitivity": "5-item variant [1,2,3,7,10] (item 6 removed), "
                           "application-specific robustness check, same Holm "
                           "family",
            "ruben_secondary": {n: s["ns"] for n, s in
                                RUBEN_OTHER_COMPOSITES.items()},
            "item_09_reversed_in_conscientious": True,
            "item6_pca_membership": pca_result,
        },
    }
    with open(outdir / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print()
    print(summary)
    print(f"\nOutputs written to {outdir}")


if __name__ == "__main__":
    main()
