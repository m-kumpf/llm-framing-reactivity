#!/usr/bin/env python3
"""Validate raw IPIP Big-Five responses: re-parse, deduplicate, and check completeness."""

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

from utils import check_output_path, discover_model_files, model_output_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_response(raw: str) -> float:
    """Extract a valid 1-5 integer from a raw response string, or return NaN.

    Uses the same logic as collect_responses.parse_likert_response:
    finds all standalone digits 1-5 and accepts only if there is exactly one.
    """
    if pd.isna(raw):
        return float("nan")
    text = str(raw).strip()
    if not text:
        return float("nan")
    matches = re.findall(r'\b([1-5])\b', text)
    if len(matches) == 1:
        return int(matches[0])
    return float("nan")


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add parsed_response and is_valid columns to the dataframe."""
    df = df.copy()
    df["parsed_response"] = df["raw_response"].apply(parse_response)
    df["is_valid"] = df["parsed_response"].notna()
    return df


def cleanup_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate and keep only valid rows.

    For each (model, item_number, repetition) group, keeps the last valid row
    (by original row order, so appended reruns take precedence).
    Groups with no valid row are dropped.
    """
    df = df.copy()
    df["_row_order"] = range(len(df))

    deduped = []
    for _key, group in df.groupby(["model", "item_number", "repetition"]):
        valid = group[group["is_valid"] == True]
        if len(valid) > 0:
            deduped.append(valid.sort_values("_row_order").iloc[[-1]])

    df = pd.concat(deduped, ignore_index=True)
    df = df.drop(columns=["_row_order"])
    return df


def check_completeness(df: pd.DataFrame, model: str, expected_reps: int = 10) -> bool:
    """Check that every item has the expected number of valid responses.

    Returns True if complete, False otherwise.
    """
    expected_items = 50
    expected_total = expected_items * expected_reps
    actual_total = len(df)
    is_complete = True

    # Check each item
    grouped = df.groupby("item_number")["repetition"].apply(set)
    expected_rep_set = set(range(1, expected_reps + 1))

    incomplete_items = 0
    for item_num in range(1, expected_items + 1):
        if item_num not in grouped.index:
            missing_reps = list(expected_rep_set)
            logger.warning(
                "%s item %d has 0/%d valid responses (missing reps: %s)",
                model, item_num, expected_reps, missing_reps,
            )
            incomplete_items += 1
            is_complete = False
        else:
            present = grouped[item_num]
            missing = sorted(expected_rep_set - present)
            if missing:
                logger.warning(
                    "%s item %d has %d/%d valid responses (missing reps: %s)",
                    model, item_num, len(present), expected_reps, missing,
                )
                incomplete_items += 1
                is_complete = False

    logger.info(
        "%s: %d/%d valid responses, %d incomplete items",
        model, actual_total, expected_total, incomplete_items,
    )
    return is_complete


def print_summary(all_dfs: dict[str, pd.DataFrame]) -> None:
    """Print per-model validation summary across all models."""
    logger.info("=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)

    total_all = 0

    for model, df in sorted(all_dfs.items()):
        count = len(df)
        logger.info("  %-45s  valid: %4d", model, count)
        total_all += count

    logger.info("-" * 60)
    logger.info("  %-45s  valid: %4d", "TOTAL", total_all)


def main() -> None:
    """Run the validation pipeline."""
    parser = argparse.ArgumentParser(description="Validate IPIP Big-Five raw responses")
    parser.add_argument("--input-dir", type=Path, default=Path("output/collect-responses"), help="Directory containing per-model raw_responses CSVs")
    parser.add_argument("--output-dir", type=Path, default=Path("output/validate-responses"), help="Output directory for per-model validated CSVs")
    parser.add_argument("--force", action="store_true", help="Overwrite output files if they exist")
    args = parser.parse_args()

    # Discover per-model raw response files
    model_files = discover_model_files(args.input_dir, "raw_responses")
    if not model_files:
        logger.error("No raw_responses_*.csv files found in %s", args.input_dir)
        raise SystemExit(1)

    logger.info("Found %d model file(s): %s", len(model_files), list(model_files.keys()))

    # Pre-check all output paths before processing
    for model in model_files:
        out_path = model_output_path(args.output_dir, "validated_responses", model)
        check_output_path(out_path, args.force)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_validated: dict[str, pd.DataFrame] = {}

    for model, input_path in sorted(model_files.items()):
        logger.info("Reading raw responses for %s from %s", model, input_path)
        df = pd.read_csv(
            input_path,
            dtype={"item_number": int, "repetition": int, "model": str, "raw_response": str},
        )
        logger.info("Loaded %d rows for %s", len(df), model)

        df = validate_dataframe(df)

        invalid_count = (~df["is_valid"]).sum()
        if invalid_count > 0:
            logger.warning("%s: %d rows failed parsing (will be dropped)", model, invalid_count)

        df = cleanup_dataframe(df)
        df["parsed_response"] = df["parsed_response"].astype(int)

        out_path = model_output_path(args.output_dir, "validated_responses", model)
        df.to_csv(out_path, index=False)
        logger.info("Validated responses saved to %s (%d valid rows)", out_path, len(df))

        all_validated[model] = df

    print_summary(all_validated)

    # Completeness check across all models
    all_complete = True
    for model, df in sorted(all_validated.items()):
        if not check_completeness(df, model):
            all_complete = False

    if all_complete:
        logger.info("All models have complete data (500/500 valid responses each)")
    else:
        logger.error("Some models have incomplete data — see warnings above")
        logger.error("Re-run collect_responses.py --retry-errors to fill gaps, then re-validate")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
