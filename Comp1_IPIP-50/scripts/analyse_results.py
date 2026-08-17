#!/usr/bin/env python3
"""Analyse IPIP Big-Five factor scores across LLMs."""

import argparse
import logging
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy import stats

from utils import check_output_path, discover_model_files


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FACTOR_COLUMNS = [
    "Extraversion", "Agreeableness", "Conscientiousness",
    "Emotional_Stability", "Intellect_Openness",
]


def compute_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive statistics per model per factor."""
    records = []
    for model, group in df.groupby("model"):
        for factor in FACTOR_COLUMNS:
            vals = group[factor].dropna()
            records.append({
                "model": model,
                "factor": factor,
                "mean": vals.mean(),
                "sd": vals.std(),
                "median": vals.median(),
                "min": vals.min(),
                "max": vals.max(),
                "n": len(vals),
            })
    return pd.DataFrame(records)


def rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    """Compute rank-biserial correlation as effect size for Mann-Whitney U."""
    return 1 - (2 * u_stat) / (n1 * n2)


def holm_correction(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni step-down correction to a list of p-values."""
    n = len(p_values)
    if n == 0:
        return []
    sorted_indices = sorted(range(n), key=lambda i: p_values[i])
    corrected = [0.0] * n
    cummax = 0.0
    for rank, idx in enumerate(sorted_indices):
        if pd.isna(p_values[idx]):
            corrected[idx] = float("nan")
        else:
            adjusted = p_values[idx] * (n - rank)
            cummax = max(cummax, adjusted)
            corrected[idx] = min(cummax, 1.0)
    return corrected


def run_statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Run Kruskal-Wallis and pairwise Mann-Whitney U tests across models."""
    models = df["model"].unique()
    if len(models) < 2:
        logger.warning("Only %d model(s) found — skipping statistical tests", len(models))
        return pd.DataFrame()

    records = []

    # --- Pass 1: Kruskal-Wallis omnibus tests ---
    kw_results = []
    for factor in FACTOR_COLUMNS:
        groups = [df[df["model"] == m][factor].dropna().values for m in models]

        if all(len(g) >= 2 for g in groups):
            h_stat, p_val = stats.kruskal(*groups)
        else:
            h_stat, p_val = float("nan"), float("nan")

        kw_results.append({
            "test": "Kruskal-Wallis",
            "factor": factor,
            "model_1": "all",
            "model_2": "all",
            "statistic": h_stat,
            "p_value": p_val,
            "effect_size": float("nan"),
            "effect_size_type": "",
        })

    # Holm-correct KW p-values across factors
    kw_ps = [r["p_value"] for r in kw_results]
    kw_corrected = holm_correction(kw_ps)
    for result, p_corr in zip(kw_results, kw_corrected):
        result["p_corrected"] = p_corr
        result["significant_005"] = p_corr < 0.05 if not pd.isna(p_corr) else False
    records.extend(kw_results)

    # --- Pass 2: Pairwise Mann-Whitney U (gated on corrected KW p-value) ---
    for kw in kw_results:
        if not kw["significant_005"]:
            continue
        factor = kw["factor"]

        mwu_results = []
        for m1, m2 in combinations(models, 2):
            g1 = df[df["model"] == m1][factor].dropna().values
            g2 = df[df["model"] == m2][factor].dropna().values

            if len(g1) >= 1 and len(g2) >= 1:
                u_stat, u_p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                r_rb = rank_biserial(u_stat, len(g1), len(g2))
            else:
                u_stat, u_p, r_rb = float("nan"), float("nan"), float("nan")

            mwu_results.append({
                "test": "Mann-Whitney U",
                "factor": factor,
                "model_1": m1,
                "model_2": m2,
                "statistic": u_stat,
                "p_value": u_p,
                "effect_size": r_rb,
                "effect_size_type": "rank-biserial r",
            })

        raw_ps = [r["p_value"] for r in mwu_results]
        corrected_ps = holm_correction(raw_ps)
        for result, p_corr in zip(mwu_results, corrected_ps):
            result["p_corrected"] = p_corr
            result["significant_005"] = p_corr < 0.05 if not pd.isna(p_corr) else False
        records.extend(mwu_results)

    return pd.DataFrame(records)


def main() -> None:
    """Run the analysis pipeline."""
    parser = argparse.ArgumentParser(description="Analyse IPIP Big-Five factor scores")
    parser.add_argument("--input-dir", type=Path, default=Path("output/score-responses"), help="Directory containing per-model scored_responses CSVs")
    parser.add_argument("--output-dir", type=Path, default=Path("output/analyse-results"), help="Output directory for CSVs")
    parser.add_argument("--force", action="store_true", help="Overwrite output files if they exist")
    args = parser.parse_args()

    # Check all output paths before doing any work
    output_files = [
        args.output_dir / "summary_statistics.csv",
        args.output_dir / "statistical_tests.csv",
    ]
    for path in output_files:
        check_output_path(path, args.force)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Discover and load per-model scored response files
    model_files = discover_model_files(args.input_dir, "scored_responses")
    if not model_files:
        logger.error("No scored_responses_*.csv files found in %s", args.input_dir)
        raise SystemExit(1)

    logger.info("Found %d model file(s): %s", len(model_files), list(model_files.keys()))

    dfs = []
    for model, path in sorted(model_files.items()):
        logger.info("Loading scored responses for %s from %s", model, path)
        model_df = pd.read_csv(path, dtype={"model": str, "repetition": int})
        dfs.append(model_df)

    df = pd.concat(dfs, ignore_index=True)

    # Filter to complete runs only
    complete = df[df["complete"] == True].copy()
    logger.info("Loaded %d runs (%d complete) across %d models", len(df), len(complete), len(model_files))

    if complete.empty:
        logger.error("No complete runs found — cannot analyse")
        raise SystemExit(1)

    # Descriptive statistics
    logger.info("Computing descriptive statistics...")
    summary = compute_descriptive_stats(complete)
    summary_path = args.output_dir / "summary_statistics.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Summary statistics saved to %s", summary_path)

    # Print summary table
    logger.info("\nDESCRIPTIVE STATISTICS:")
    for model in complete["model"].unique():
        logger.info("  Model: %s", model)
        model_stats = summary[summary["model"] == model]
        for _, row in model_stats.iterrows():
            logger.info(
                "    %-25s  M=%.1f  SD=%.1f  Mdn=%.1f  [%.0f, %.0f]  n=%d",
                row["factor"], row["mean"], row["sd"], row["median"],
                row["min"], row["max"], int(row["n"]),
            )


    # Statistical tests
    logger.info("Running statistical tests...")
    test_results = run_statistical_tests(complete)
    if not test_results.empty:
        tests_path = args.output_dir / "statistical_tests.csv"
        test_results.to_csv(tests_path, index=False)
        logger.info("Statistical test results saved to %s", tests_path)

        # Print significant results
        sig = test_results[test_results["significant_005"] == True]
        if not sig.empty:
            logger.info("\nSIGNIFICANT RESULTS (p < 0.05):")
            for _, row in sig.iterrows():
                logger.info(
                    "  %s | %s | %s vs %s | stat=%.2f | p=%.4f | p_corr=%.4f | r=%.3f",
                    row["test"], row["factor"], row["model_1"], row["model_2"],
                    row["statistic"], row["p_value"], row["p_corrected"],
                    row["effect_size"] if not pd.isna(row["effect_size"]) else 0,
                )
        else:
            logger.info("No statistically significant differences found.")
    else:
        logger.info("Statistical tests skipped (insufficient models).")

    logger.info("Analysis complete.")


if __name__ == "__main__":
    main()
