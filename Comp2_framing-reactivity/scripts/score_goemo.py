#!/usr/bin/env python3
"""
Score Component 2 responses for perceived emotional valence using
GoEmotions RoBERTa (SamLowe/roberta-base-go_emotions).

Produces two Tier 2 metrics:
    pos_emotion  – mean perceived positive emotional valence
    neg_emotion  – mean perceived negative emotional valence

Scoring approach:
    1. Split each response into sentences (same splitter as Tier 1)
    2. Run each sentence through GoEmotions RoBERTa (28-label classifier)
    3. Sum raw probabilities across positive/negative label groups per sentence
    4. Average sentence-level scores → response-level pos_emotion, neg_emotion

Model: SamLowe/roberta-base-go_emotions (HuggingFace)
    Trained on GoEmotions dataset (Demszky et al., ACL 2020)
    28 emotion labels, multi-label classification
    Label groupings from Demszky et al. 2020, Table 3

Usage:
    python score_goemo.py                                # default dirs, MPS device
    python score_goemo.py --save-sentences               # also save sentence-level detail
    python score_goemo.py --device cpu                   # force CPU (slower)
    python score_goemo.py --batch-size 16                # adjust batch size
    python score_goemo.py --force                        # overwrite existing output
    python score_goemo.py --input-dir path/to/csvs       # custom input dir
    python score_goemo.py --output-dir path/to/out       # custom output dir
"""

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# GOEMOTIONS LABEL GROUPINGS (Demszky et al. 2020, Table 3)
# ═══════════════════════════════════════════════════════════════════════════════

POSITIVE_LABELS = frozenset({
    "admiration", "amusement", "approval", "caring",
    "desire", "excitement", "gratitude", "joy",
    "love", "optimism", "pride", "relief",
})

NEGATIVE_LABELS = frozenset({
    "anger", "annoyance", "disappointment", "disapproval",
    "disgust", "embarrassment", "fear", "grief",
    "nervousness", "remorse", "sadness",
})

# Excluded (ambiguous): confusion, curiosity, realization, surprise, neutral

ALL_28_LABELS = sorted(POSITIVE_LABELS | NEGATIVE_LABELS | {
    "confusion", "curiosity", "realization", "surprise", "neutral",
})


# ═══════════════════════════════════════════════════════════════════════════════
# SENTENCE SPLITTER (identical to Tier 1 — score_deterministic.py)
# ═══════════════════════════════════════════════════════════════════════════════

ABBREVIATIONS = [
    "e.g.", "i.e.", "etc.", "vs.", "approx.",
    "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.",
    "Sr.", "Jr.",
    "a.m.", "p.m.",
    "no.", "No.",
]

_ABBREV_MAP = {a: a.replace(".", "\x00") for a in ABBREVIATIONS}
_ABBREV_RESTORE = {v: k for k, v in _ABBREV_MAP.items()}

_DECIMAL_PAT = re.compile(r"(\d)\.(\d)")


def sentences(text: str) -> list[str]:
    """Split text into sentence-like segments.

    Handles abbreviations (e.g., Dr., i.e.), decimal numbers (3.5),
    bullet points, numbered lists, and markdown headers as boundaries.

    Identical to the Tier 1 sentence splitter in score_deterministic.py.
    """
    # Step 1: Protect abbreviations
    protected = text
    for orig, placeholder in _ABBREV_MAP.items():
        protected = protected.replace(orig, placeholder)

    # Step 2: Protect decimal numbers (e.g., 3.5 -> 3\x015)
    protected = _DECIMAL_PAT.sub(
        lambda m: m.group(1) + "\x01" + m.group(2), protected
    )

    # Step 3: Bullet/list items as sentence boundaries
    protected = re.sub(r"(?m)^\s*[-*•]\s+", ". ", protected)
    protected = re.sub(r"(?m)^\s*\d+[.)]\s+", ". ", protected)

    # Step 4: Markdown headers as segment boundaries
    protected = re.sub(r"(?m)^#{1,6}\s+(.*)", r". \1.", protected)

    # Step 5: Split on sentence-ending punctuation
    raw_segments = re.split(r"[.!?]+(?:\s|$)", protected)

    # Step 6: Restore placeholders and filter
    result = []
    for seg in raw_segments:
        seg = seg.strip()
        for placeholder, orig in _ABBREV_RESTORE.items():
            seg = seg.replace(placeholder, orig)
        seg = seg.replace("\x01", ".")
        if len(seg) > 5:
            result.append(seg)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL NAME MAPPING (same as Tier 1)
# ═══════════════════════════════════════════════════════════════════════════════

_MODEL_NAMES = {
    "claude":  "Claude Sonnet 4.6",
    "gpt":     "GPT-5.3",
    "gemini":  "Gemini 3.1 Pro",
    "kimi":    "Kimi K2.5",
    "qwen":    "Qwen 3.5",
    "minimax": "MiniMax M2.7",
    "glm":     "GLM-5",
}


def model_short_name(raw: str) -> str:
    """Map raw OpenRouter model identifier to short display name."""
    raw_lower = raw.lower()
    for key, name in _MODEL_NAMES.items():
        if key in raw_lower:
            return name
    return raw


# ═══════════════════════════════════════════════════════════════════════════════
# IDENTIFIER COLUMNS
# ═══════════════════════════════════════════════════════════════════════════════

ID_COLUMNS = [
    "vignette_id",
    "scenario_id",
    "scenario_label",
    "framing_id",
    "framing_label",
    "model",
    "run_number",
]


# ═══════════════════════════════════════════════════════════════════════════════
# SCORING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def load_classifier(device: str, model_name: str):
    """Load the GoEmotions pipeline. Returns the classifier."""
    from transformers import pipeline as hf_pipeline

    logger.info("Loading model: %s on device: %s", model_name, device)
    t0 = time.time()

    classifier = hf_pipeline(
        task="text-classification",
        model=model_name,
        top_k=None,
        device=device,
    )

    logger.info("Model loaded in %.1fs", time.time() - t0)
    return classifier


def score_sentences(classifier, sents: list[str], batch_size: int) -> list[dict]:
    """Run classifier on a list of sentences. Returns per-sentence score dicts.

    Each dict has keys: pos_score, neg_score, plus all 28 individual labels.
    """
    if not sents:
        return []

    # The pipeline returns list of list-of-dicts when given a list of strings
    raw_outputs = classifier(sents, truncation=True, batch_size=batch_size)

    results = []
    for sent_output in raw_outputs:
        # sent_output is a list of 28 dicts: [{"label": "joy", "score": 0.03}, ...]
        label_scores = {item["label"]: item["score"] for item in sent_output}

        pos_score = sum(label_scores.get(lb, 0.0) for lb in POSITIVE_LABELS)
        neg_score = sum(label_scores.get(lb, 0.0) for lb in NEGATIVE_LABELS)

        result = {
            "pos_score": pos_score,
            "neg_score": neg_score,
        }
        # Also store individual label scores (for sentence-level detail output)
        result.update(label_scores)
        results.append(result)

    return results


def score_response(classifier, response_text: str, batch_size: int) -> dict:
    """Score a single response. Returns dict with response-level metrics.

    Returns:
        dict with keys: pos_emotion, neg_emotion, n_sentences
        Also includes a 'sentence_details' key with list of per-sentence dicts
        (only used if --save-sentences is active).
    """
    sents = sentences(response_text)
    n_sents = len(sents)

    if n_sents == 0:
        return {
            "pos_emotion": 0.0,
            "neg_emotion": 0.0,
            "n_sentences": 0,
            "sentence_details": [],
        }

    sent_scores = score_sentences(classifier, sents, batch_size)

    # Unweighted average across sentences
    pos_emotion = np.mean([s["pos_score"] for s in sent_scores])
    neg_emotion = np.mean([s["neg_score"] for s in sent_scores])

    # Attach sentence text to details for optional output
    for i, detail in enumerate(sent_scores):
        detail["sentence_text"] = sents[i]
        detail["sent_idx"] = i

    return {
        "pos_emotion": round(float(pos_emotion), 6),
        "neg_emotion": round(float(neg_emotion), 6),
        "n_sentences": n_sents,
        "sentence_details": sent_scores,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PER-MODEL PROCESSING (with resume support)
# ═══════════════════════════════════════════════════════════════════════════════

def sanitize_filename(model_id: str) -> str:
    """Convert model ID to safe filename: 'anthropic/claude-sonnet-4.6' -> 'anthropic_claude-sonnet-4_6'"""
    return model_id.replace("/", "_").replace(".", "_")


def process_model_file(
    classifier,
    csv_path: Path,
    parts_dir: Path,
    sent_parts_dir: Path | None,
    batch_size: int,
) -> tuple[Path, Path | None]:
    """Score all responses in a single model CSV.

    Returns paths to the response-level and (optionally) sentence-level output files.
    """
    df = pd.read_csv(csv_path)
    model_id = df["model"].iloc[0]
    safe_name = sanitize_filename(model_id)

    response_out = parts_dir / f"{safe_name}.csv"
    sentence_out = sent_parts_dir / f"{safe_name}.csv" if sent_parts_dir else None

    # Resume check: if response output exists and has correct row count, skip
    if response_out.exists():
        existing = pd.read_csv(response_out)
        if len(existing) == len(df):
            logger.info("  %s already scored (%d rows) — skipping", model_id, len(df))
            return response_out, sentence_out
        else:
            logger.warning(
                "  Partial output for %s (%d/%d rows) — re-scoring",
                model_id, len(existing), len(df),
            )

    short_name = model_short_name(model_id)
    logger.info("  Scoring %s (%d responses)...", model_id, len(df))

    response_rows = []
    sentence_rows = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"  {short_name}",
        leave=True,
    ):
        text = row.get("response_text", "")
        if not text or (isinstance(text, str) and not text.strip()):
            logger.warning(
                "  Empty response_text: %s run %s — scoring as zero",
                row.get("vignette_id"), row.get("run_number"),
            )
            result = {
                "pos_emotion": 0.0,
                "neg_emotion": 0.0,
                "n_sentences": 0,
                "sentence_details": [],
            }
        else:
            result = score_response(classifier, text, batch_size)

        # Build response-level row
        resp_row = {col: row[col] for col in ID_COLUMNS}
        resp_row["model_short"] = short_name
        resp_row["pos_emotion"] = result["pos_emotion"]
        resp_row["neg_emotion"] = result["neg_emotion"]
        resp_row["n_sentences"] = result["n_sentences"]
        response_rows.append(resp_row)

        # Build sentence-level rows (if requested)
        if sent_parts_dir is not None:
            for detail in result["sentence_details"]:
                sent_row = {
                    "vignette_id": row["vignette_id"],
                    "model": model_id,
                    "model_short": short_name,
                    "run_number": row["run_number"],
                    "sent_idx": detail["sent_idx"],
                    "sentence_text": detail["sentence_text"],
                    "pos_score": round(detail["pos_score"], 6),
                    "neg_score": round(detail["neg_score"], 6),
                }
                # Add all 28 individual label scores
                for label in ALL_28_LABELS:
                    sent_row[label] = round(detail.get(label, 0.0), 6)
                sentence_rows.append(sent_row)

    # Save response-level output
    resp_df = pd.DataFrame(response_rows)
    resp_df.to_csv(response_out, index=False)
    logger.info("  Saved %d response scores to %s", len(resp_df), response_out.name)

    # Save sentence-level output
    if sent_parts_dir is not None and sentence_rows:
        sent_df = pd.DataFrame(sentence_rows)
        sent_df.to_csv(sentence_out, index=False)
        logger.info("  Saved %d sentence scores to %s", len(sent_df), sentence_out.name)

    return response_out, sentence_out


# ═══════════════════════════════════════════════════════════════════════════════
# CONCATENATE PARTS INTO FINAL OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

def concatenate_parts(parts_dir: Path, output_path: Path, label: str) -> int:
    """Concatenate per-model CSV parts into a single output file. Returns total rows."""
    part_files = sorted(parts_dir.glob("*.csv"))
    if not part_files:
        logger.warning("No %s part files found in %s", label, parts_dir)
        return 0

    dfs = [pd.read_csv(f) for f in part_files]
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(output_path, index=False)
    logger.info("Combined %s: %d rows -> %s", label, len(combined), output_path)
    return len(combined)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score Component 2 responses for perceived emotional valence "
                    "using GoEmotions RoBERTa",
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("output/clean_responses"),
        help="Directory containing responses_*.csv (default: output/clean_responses)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/score_goemo"),
        help="Output directory (default: output/score_goemo)",
    )
    parser.add_argument(
        "--model-name", type=str,
        default="SamLowe/roberta-base-go_emotions",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="Device for inference (default: cpu; use mps/cuda if available)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for sentence-level inference (default: 32)",
    )
    parser.add_argument(
        "--save-sentences", action="store_true",
        help="Also save sentence-level detail (large file)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Delete existing per-model parts and re-score everything",
    )
    args = parser.parse_args()

    # ── Discover input files ──────────────────────────────────────────────
    csv_files = sorted(args.input_dir.glob("responses_*.csv"))
    if not csv_files:
        logger.error("No responses_*.csv files found in %s", args.input_dir)
        raise SystemExit(1)

    logger.info("Found %d response file(s) in %s:", len(csv_files), args.input_dir)
    for f in csv_files:
        logger.info("  %s", f.name)

    # ── Set up output directories ─────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = args.output_dir / "_parts"
    parts_dir.mkdir(exist_ok=True)

    sent_parts_dir = None
    if args.save_sentences:
        sent_parts_dir = args.output_dir / "_sentence_parts"
        sent_parts_dir.mkdir(exist_ok=True)

    # ── Force: clear existing parts ───────────────────────────────────────
    if args.force:
        for f in parts_dir.glob("*.csv"):
            f.unlink()
            logger.info("  Deleted %s", f)
        if sent_parts_dir:
            for f in sent_parts_dir.glob("*.csv"):
                f.unlink()

    # ── Load model ────────────────────────────────────────────────────────
    classifier = load_classifier(args.device, args.model_name)

    # ── Process each model file ───────────────────────────────────────────
    t_start = time.time()

    for csv_file in csv_files:
        logger.info("Processing %s ...", csv_file.name)
        process_model_file(
            classifier=classifier,
            csv_path=csv_file,
            parts_dir=parts_dir,
            sent_parts_dir=sent_parts_dir,
            batch_size=args.batch_size,
        )

    # ── Concatenate into final output ─────────────────────────────────────
    final_response_path = args.output_dir / "scored_goemo.csv"
    total_responses = concatenate_parts(parts_dir, final_response_path, "response-level")

    if args.save_sentences and sent_parts_dir:
        final_sentence_path = args.output_dir / "scored_goemo_sentences.csv"
        total_sentences = concatenate_parts(sent_parts_dir, final_sentence_path, "sentence-level")
    else:
        total_sentences = 0

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    expected_responses = len(csv_files) * 600

    logger.info("=" * 60)
    logger.info("SCORING COMPLETE")
    logger.info("=" * 60)
    logger.info("  Models scored:    %d", len(csv_files))
    logger.info("  Total responses:  %d (expected %d)", total_responses, expected_responses)
    if total_sentences:
        logger.info("  Total sentences:  %d", total_sentences)
    logger.info("  Elapsed time:     %.1f min", elapsed / 60)
    logger.info("  Output:           %s", final_response_path)
    if args.save_sentences:
        logger.info("  Sentence detail:  %s", final_sentence_path)

    if total_responses != expected_responses:
        logger.warning("Row count mismatch — expected %d, got %d", expected_responses, total_responses)


if __name__ == "__main__":
    main()
