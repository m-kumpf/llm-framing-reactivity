#!/usr/bin/env python3
"""Score validated IPIP Big-Five responses into factor scores."""

import argparse
import logging
from pathlib import Path

import pandas as pd

from utils import check_output_path, discover_model_files, model_output_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Map factor labels from the item bank to output column names
FACTOR_COLUMN_MAP = {
    "Extraversion": "Extraversion",
    "Agreeableness": "Agreeableness",
    "Conscientiousness": "Conscientiousness",
    "Emotional Stability": "Emotional_Stability",
    "Intellect/Openness": "Intellect_Openness",
}

FACTOR_COLUMNS = list(FACTOR_COLUMN_MAP.values())


def compute_scored_value(parsed_response: float, keying: str) -> float:
    """Compute scored value with reverse scoring for negatively keyed items."""
    if pd.isna(parsed_response):
        return float("nan")
    if keying == "+":
        return parsed_response
    elif keying == "-":
        return 6 - parsed_response
    else:
        raise ValueError(f"Unknown keying value: {keying}")


def score_item_level(validated_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    """Merge validated responses with item keying and compute scored values."""
    # Filter to valid responses only
    valid = validated_df[validated_df["is_valid"] == True].copy()

    # Merge with item info
    merged = valid.merge(
        items_df[["item_number", "factor_label", "keying"]],
        on="item_number",
        how="left",
    )

    # Compute scored values
    merged["scored_value"] = merged.apply(
        lambda row: compute_scored_value(row["parsed_response"], row["keying"]),
        axis=1,
    )

    # Select output columns
    output_cols = [
        "model", "item_number", "repetition", "raw_response",
        "parsed_response", "scored_value", "factor_label", "keying",
    ]
    return merged[output_cols].copy()


def aggregate_factor_scores(item_level_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate item-level scores into factor scores per model and repetition."""
    records = []

    for (model, rep), group in item_level_df.groupby(["model", "repetition"]):
        row = {"model": model, "repetition": rep}

        all_complete = True
        for factor_label, col_name in FACTOR_COLUMN_MAP.items():
            factor_items = group[group["factor_label"] == factor_label]
            if len(factor_items) != 10 or factor_items["scored_value"].isna().any():
                row[col_name] = float("nan")
                all_complete = False
            else:
                row[col_name] = factor_items["scored_value"].sum()

        row["complete"] = all_complete
        records.append(row)

    result = pd.DataFrame(records)

    # Ensure correct column order
    col_order = ["model", "repetition"] + FACTOR_COLUMNS + ["complete"]
    return result[col_order]


def main() -> None:
    """Run the scoring pipeline."""
    parser = argparse.ArgumentParser(description="Score validated IPIP Big-Five responses")
    parser.add_argument("--input-dir", type=Path, default=Path("output/validate-responses"), help="Directory containing per-model validated_responses CSVs")
    parser.add_argument("--output-dir", type=Path, default=Path("output/score-responses"), help="Output directory for per-model scored CSVs")
    parser.add_argument("--items", type=Path, default=Path("ipip_big5_50_items.csv"), help="IPIP items CSV")
    parser.add_argument("--force", action="store_true", help="Overwrite output files if they exist")
    args = parser.parse_args()

    # Discover per-model validated response files
    model_files = discover_model_files(args.input_dir, "validated_responses")
    if not model_files:
        logger.error("No validated_responses_*.csv files found in %s", args.input_dir)
        raise SystemExit(1)

    logger.info("Found %d model file(s): %s", len(model_files), list(model_files.keys()))

    # Pre-check all output paths before processing
    for model in model_files:
        check_output_path(model_output_path(args.output_dir, "scored_responses", model), args.force)
        check_output_path(model_output_path(args.output_dir, "scored_item_level", model), args.force)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading items from %s", args.items)
    items_df = pd.read_csv(args.items, dtype={"item_number": int, "keying": str})

    for model, input_path in sorted(model_files.items()):
        logger.info("Loading validated responses for %s from %s", model, input_path)
        validated_df = pd.read_csv(
            input_path,
            dtype={"item_number": int, "repetition": int, "model": str, "raw_response": str},
        )

        # Item-level scoring
        logger.info("Computing item-level scores for %s...", model)
        item_level = score_item_level(validated_df, items_df)

        item_out = model_output_path(args.output_dir, "scored_item_level", model)
        item_level.to_csv(item_out, index=False)
        logger.info("Item-level scores saved to %s (%d rows)", item_out, len(item_level))

        # Factor-level aggregation
        logger.info("Aggregating factor scores for %s...", model)
        factor_scores = aggregate_factor_scores(item_level)

        scored_out = model_output_path(args.output_dir, "scored_responses", model)
        factor_scores.to_csv(scored_out, index=False)

        complete_count = factor_scores["complete"].sum()
        total_count = len(factor_scores)
        logger.info(
            "Factor scores saved to %s (%d runs, %d complete)",
            scored_out, total_count, int(complete_count),
        )

        # Print summary for this model
        if not factor_scores.empty:
            complete = factor_scores[factor_scores["complete"] == True]
            if not complete.empty:
                logger.info("Factor score summary for %s (complete runs):", model)
                for col in FACTOR_COLUMNS:
                    vals = complete[col]
                    logger.info(
                        "  %-25s  mean=%.1f  sd=%.1f  range=[%.0f, %.0f]",
                        col, vals.mean(), vals.std(), vals.min(), vals.max(),
                    )


if __name__ == "__main__":
    main()
