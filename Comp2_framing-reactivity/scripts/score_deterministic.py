#!/usr/bin/env python3
"""
Deterministic scoring — Tier 1 metrics for Component 2.

Reads cleaned response CSVs from output/clean_responses/, scores each
response on 8 deterministic linguistic metrics (+ FK grade as
supplementary), and writes a single merged scored_deterministic.csv.

Tier 1 (8 metrics, fully deterministic):
    word_count, format_density, questions, avg_sent_len, coleman_liau,
    first_person, second_person, hedging

Supplementary:
    fk_grade (Flesch-Kincaid Grade Level, for cross-study comparability)

Usage:
    python score_deterministic.py                          # default dirs
    python score_deterministic.py --input-dir path/to/csvs # custom input
    python score_deterministic.py --output-dir path/to/out # custom output
    python score_deterministic.py --validate               # run unit tests
    python score_deterministic.py --force                  # overwrite existing
"""

import argparse
import csv
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC KEYS
# ═══════════════════════════════════════════════════════════════════════════════

METRIC_KEYS = [
    "word_count",
    "format_density",
    "questions",
    "avg_sent_len",
    "coleman_liau",
    "first_person",
    "second_person",
    "hedging",
]

SUPPLEMENTARY_KEYS = [
    "fk_grade",
]

# Columns carried from input to output (identifiers only, no bulky text)
ID_COLUMNS = [
    "vignette_id",
    "scenario_id",
    "scenario_label",
    "framing_id",
    "framing_label",
    "model",
    "run_number",
]

OUTPUT_COLUMNS = ID_COLUMNS + ["model_short"] + METRIC_KEYS + SUPPLEMENTARY_KEYS


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL NAME MAPPING
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
# TIER 1 — CORE TEXT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def words(text: str) -> list[str]:
    """Extract words (alphanumeric tokens including numbers like 500mg, 3)."""
    return re.findall(r"[a-zA-Z0-9']+", text)


# ── Sentence splitting with abbreviation/decimal/bullet protection ───────────

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
# TIER 1 — PATTERN COUNTING
# ═══════════════════════════════════════════════════════════════════════════════

def count_pattern(text: str, patterns: list[str]) -> int:
    """Count occurrences of regex patterns in text (case-insensitive)."""
    return sum(len(re.findall(p, text, re.IGNORECASE)) for p in patterns)


# ── Pronoun patterns (deduplicated, no contraction-specific patterns) ────────

FIRST_PERSON_PATS = [
    r"\bI\b",
    r"\bme\b",
    r"\bmy\b",
    r"\bmine\b",
    r"\bmyself\b",
    r"\bwe\b",
    r"\bus\b",
    r"\bour\b",
    r"\bours\b",
    r"\bourselves\b",
]

SECOND_PERSON_PATS = [
    r"\byou\b",
    r"\byour\b",
    r"\byours\b",
    r"\byourself\b",
    r"\byourselves\b",
]


# ── Hedging patterns (65 lemmas / 89 word forms) ─────────────────────────
# Source: Hyland (2019, pp. 218–224), 60 lemmas retained from 69 total.
# Domain-specific additions: 5 lemmas.
# Two adaptations: "supposed to" and "rather than" excluded via lookahead.

HEDGING_PATS = [
    # ── A. Modal hedges (5 lemmas) ───────────────────────────────────
    r'\bmay\b',
    r'\bmaybe\b',
    r'\bmight\b',
    r'\bcould(?:n\'?t)?\b',
    r'\bought\b',

    # ── B. Epistemic lexical verbs (11 lemmas) ───────────────────────
    r'\bsuggest(?:s|ed|ing)?\b',
    r'\bindicat(?:e[ds]?|ing)\b',
    r'\bappear(?:s|ed|ing)?\b',
    r'\bseem(?:s|ed|ing)?\b',
    r'\bassum(?:e[ds]?|ing)\b',
    r'\bsuppos(?:e[s]?|ing)\b',
    r'\bsupposed\b(?!\s+to\b)',              # supposed (hedge) NOT "supposed to"
    r'\bsuspect(?:s|ed|ing)?\b',
    r'\bestimat(?:e[ds]?|ing)\b',
    r'\bdoubt(?:s|ed|ing|ful)?\b',
    r'\bguess(?:es|ed|ing)?\b',
    r'\btend(?:s|ed|ing)?\s+to\b',

    # ── C. Possibility / epistemic markers (10 lemmas) ───────────────
    r'\bpossib(?:le|ly)\b',
    r'\bprobab(?:le|ly)\b',
    r'\bapparent(?:ly)?\b',
    r'\bplausib(?:le|ly)\b',
    r'\blikely\b',
    r'\bunlikely\b',
    r'\bperhaps\b',
    r'\bpresumab(?:le|ly)\b',
    r'\buncertain(?:ly)?\b',
    r'\bunclear(?:ly)?\b',

    # ── D. Frequency adverbs (14 lemmas) ─────────────────────────────
    r'\btypical(?:ly)?\b',
    r'\bgenerally\b',
    r'\busually\b',
    r'\bsometimes\b',
    r'\boften\b',
    r'\bfrequently\b',
    r'\bin general\b',
    r'\bin most cases\b',
    r'\bin most instances\b',
    r'\bon the whole\b',
    r'\blargely\b',
    r'\bmainly\b',
    r'\bmostly\b',
    r'\bbroadly\b',

    # ── E. Downtoners (9 lemmas) ─────────────────────────────────────
    r'\bsomewhat\b',
    r'\bfairly\b',
    r'\brelatively\b',
    r'\balmost\b',
    r'\bessentially\b',
    r'\brather\b(?!\s+than\b)',              # rather (downtoner) NOT "rather than"
    r'\ba certain amount\b',
    r'\ba certain extent\b',
    r'\ba certain level\b',

    # ── F. Precision hedges (2 lemmas) ───────────────────────────────
    r'\bapproximately\b',
    r'\broughly\b',

    # ── G. Attribution / perspective hedges (9 lemmas) ───────────────
    r'\bfrom my perspective\b',
    r'\bfrom our perspective\b',
    r'\bfrom this perspective\b',
    r'\bin my opinion\b',
    r'\bin my view\b',
    r'\bin our opinion\b',
    r'\bin our view\b',
    r'\bin this view\b',
    r'\bto my knowledge\b',

    # ── H. Domain-specific additions (5 lemmas) ──────────────────────
    r'\bpotential(?:ly)?\b',
    r'\bnot necessarily\b',
    r'\bin many cases\b',
    r'\bin some cases\b',
    r'\boccasionally\b',
]


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 1 — READABILITY
# ═══════════════════════════════════════════════════════════════════════════════

def coleman_liau_index(text: str) -> float:
    """Compute Coleman-Liau Index (character-based, no syllable counting)."""
    w = words(text)
    s = sentences(text)
    wc = len(w)
    if wc == 0:
        return 0.0

    letters = sum(sum(1 for ch in word if ch.isalnum()) for word in w)

    L = letters / wc * 100  # letters per 100 words
    S = len(s) / wc * 100   # sentences per 100 words

    return 0.0588 * L - 0.296 * S - 15.8


def syllable_count(word: str) -> int:
    """Estimate syllable count (heuristic, used only for FK supplementary)."""
    word = word.lower()
    if not word:
        return 1
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for i in range(1, len(word)):
        if word[i] in vowels and word[i - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    return max(count, 1)


def flesch_kincaid_grade(text: str) -> float:
    """Compute Flesch-Kincaid Grade Level (supplementary only)."""
    s = sentences(text)
    w = words(text)
    if not s or not w:
        return 0.0
    return (
        0.39 * (len(w) / len(s))
        + 11.8 * (sum(syllable_count(x) for x in w) / len(w))
        - 15.59
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SCORING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def score_response(text: str) -> dict:
    """Score a response text on all Tier 1 metrics + FK supplementary.

    Returns dict with metric keys and rounded float values.
    """
    w = words(text)
    wc = len(w)
    norm = 100 / max(wc, 1)
    sents = sentences(text)

    # Format density: headers + bold + bullets + numbered lists
    lines = text.split("\n")
    fmt = (
        sum(1 for line in lines if re.match(r"#{1,6}\s", line.strip()))
        + len(re.findall(r"\*\*[^*]+\*\*", text))
        + sum(1 for line in lines if re.match(r"\s*[-*•]\s", line.strip()))
        + sum(1 for line in lines if re.match(r"\s*\d+[.)]\s", line.strip()))
    )

    return {
        "word_count":     wc,
        "format_density": round(fmt * norm, 4),
        "questions":      round(count_pattern(text, [r"\?"]) * norm, 4),
        "avg_sent_len":   round(wc / max(len(sents), 1), 2),
        "coleman_liau":   round(coleman_liau_index(text), 2),
        "first_person":   round(count_pattern(text, FIRST_PERSON_PATS) * norm, 4),
        "second_person":  round(count_pattern(text, SECOND_PERSON_PATS) * norm, 4),
        "hedging":        round(count_pattern(text, HEDGING_PATS) * norm, 4),
        "fk_grade":       round(flesch_kincaid_grade(text), 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_validation() -> bool:
    """Run unit tests for all Tier 1 scoring functions."""
    passed = 0
    failed = 0

    def check(description: str, condition: bool):
        nonlocal passed, failed
        status = "PASS" if condition else "FAIL"
        if condition:
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] {description}")

    print("words() tokeniser:")
    check(
        '"take 500mg daily" -> 3 words',
        words("take 500mg daily") == ["take", "500mg", "daily"],
    )
    check(
        '"3 times per day" -> 4 words',
        words("3 times per day") == ["3", "times", "per", "day"],
    )

    print("\nPronoun deduplication:")
    check(
        "\"I'm feeling concerned\" -> first_person fires once",
        count_pattern("I'm feeling concerned", FIRST_PERSON_PATS) == 1,
    )
    check(
        "\"you're doing well\" -> second_person fires once",
        count_pattern("you're doing well", SECOND_PERSON_PATS) == 1,
    )
    check(
        '"my concerns" -> first_person fires once',
        count_pattern("my concerns", FIRST_PERSON_PATS) == 1,
    )
    check(
        '"your doctor" -> second_person fires once',
        count_pattern("your doctor", SECOND_PERSON_PATS) == 1,
    )
    check(
        '"We can manage this together" -> first_person fires once',
        count_pattern("We can manage this together", FIRST_PERSON_PATS) == 1,
    )

    print("\nHedging patterns — basic matching:")
    check(
        '"This may help" -> 1 hedge (may)',
        count_pattern("This may help", HEDGING_PATS) == 1,
    )
    check(
        '"probably" fires once',
        count_pattern("This is probably fine", HEDGING_PATS) == 1,
    )

    print("\nHedging patterns — excluded items:")
    check(
        '"should" excluded (directive, not hedge)',
        count_pattern("You should take this", HEDGING_PATS) == 0,
    )
    check(
        '"would" excluded (politeness, not hedge)',
        count_pattern("I would recommend seeing a doctor", HEDGING_PATS) == 0,
    )
    check(
        '"feel better" excluded (sensation, not hedge)',
        count_pattern("You should feel better soon", HEDGING_PATS) == 0,
    )

    print("\nHedging patterns — adaptations:")
    check(
        '"rather than" excluded',
        count_pattern("rather than surgery", HEDGING_PATS) == 0,
    )
    check(
        '"rather common" fires once (downtoner)',
        count_pattern("rather common", HEDGING_PATS) == 1,
    )
    check(
        '"supposed to take" excluded',
        count_pattern("you are supposed to take this", HEDGING_PATS) == 0,
    )
    check(
        '"the supposed cause" fires once (hedge)',
        count_pattern("the supposed cause", HEDGING_PATS) == 1,
    )

    print("\nHedging patterns — inflectional coverage:")
    check(
        '"suggesting" fires (verb -ing form)',
        count_pattern("Research is suggesting a link", HEDGING_PATS) == 1,
    )
    check(
        '"indicated" fires (verb -ed form)',
        count_pattern("Studies indicated a risk", HEDGING_PATS) == 1,
    )
    check(
        '"indicating" fires (verb -ing form)',
        count_pattern("results indicating a trend", HEDGING_PATS) == 1,
    )
    check(
        '"appeared" fires (verb -ed form)',
        count_pattern("symptoms appeared mild", HEDGING_PATS) == 1,
    )
    check(
        '"seemed" fires (verb -ed form)',
        count_pattern("the condition seemed stable", HEDGING_PATS) == 1,
    )
    check(
        '"assuming" fires (verb -ing form)',
        count_pattern("assuming no complications", HEDGING_PATS) == 1,
    )
    check(
        '"guessing" fires (verb -ing form)',
        count_pattern("I am guessing this is mild", HEDGING_PATS) == 1,
    )
    check(
        '"doubting" fires (verb -ing form)',
        count_pattern("doubting the diagnosis", HEDGING_PATS) == 1,
    )
    check(
        '"suspecting" fires (verb -ing form)',
        count_pattern("suspecting an infection", HEDGING_PATS) == 1,
    )
    check(
        '"estimating" fires (verb -ing form)',
        count_pattern("estimating the risk", HEDGING_PATS) == 1,
    )
    check(
        '"tending to" fires (verb -ing form)',
        count_pattern("tending to improve over time", HEDGING_PATS) == 1,
    )

    print("\nHedging patterns — multi-hedge sentences:")
    check(
        '"suggests it could possibly indicate" -> 4 hedges',
        count_pattern("This suggests it could possibly indicate a problem", HEDGING_PATS) == 4,
    )
    check(
        '"suggests patients often experience relatively mild" -> 3 hedges',
        count_pattern("Research suggests patients often experience relatively mild symptoms", HEDGING_PATS) == 3,
    )

    print("\nSentence splitting:")
    check(
        '"e.g. blood pressure" -> no false split',
        len(sentences("This is important, e.g. blood pressure should be monitored regularly.")) == 1,
    )
    check(
        '"Dr. Smith" -> no false split',
        len(sentences("Dr. Smith recommends daily exercise for better health outcomes.")) == 1,
    )
    check(
        '"approximately 3.5 mg" -> no false split',
        len(sentences("Take approximately 3.5 mg of the medication twice daily.")) == 1,
    )
    check(
        '"- item1\\n- item2" -> 2 segments',
        len(sentences("- Maintain a healthy diet\n- Exercise regularly")) == 2,
    )
    check(
        '"## Overview\\nThis is a summary." -> 2 segments',
        len(sentences("## Overview\nThis is a summary.")) == 2,
    )

    print("\nColeman-Liau Index:")
    sample = "The patient should take metformin twice daily with meals. Regular blood glucose monitoring is recommended."
    cli = coleman_liau_index(sample)
    check(
        f"Plausible grade level for clinical text (got {cli:.1f}, expect 6-16)",
        6 <= cli <= 16,
    )

    print("\nFlesch-Kincaid Grade Level:")
    fk = flesch_kincaid_grade(sample)
    check(
        f"Plausible FK grade for clinical text (got {fk:.1f}, expect 4-16)",
        4 <= fk <= 16,
    )

    print("\nFormat density:")
    md_text = "## Heading\n- bullet one\n- bullet two\n1. numbered\n**bold text**"
    lines = md_text.split("\n")
    fmt = (
        sum(1 for line in lines if re.match(r"#{1,6}\s", line.strip()))
        + len(re.findall(r"\*\*[^*]+\*\*", md_text))
        + sum(1 for line in lines if re.match(r"\s*[-*•]\s", line.strip()))
        + sum(1 for line in lines if re.match(r"\s*\d+[.)]\s", line.strip()))
    )
    check(f"5 format elements in sample (got {fmt})", fmt == 5)

    print("\nQuestion counting:")
    check(
        '"Is this okay? What about this?" -> 2 questions',
        count_pattern("Is this okay? What about this?", [r"\?"]) == 2,
    )

    print("\nscore_response() integration:")
    test_text = "## Overview\nYou should take your medication. I recommend consulting Dr. Smith.\n- Stay active\n- Eat well\nIs this clear?"
    scores = score_response(test_text)
    check("word_count is int", isinstance(scores["word_count"], int))
    check("all metric keys present", all(k in scores for k in METRIC_KEYS + SUPPLEMENTARY_KEYS))
    check(f"word_count > 0 (got {scores['word_count']})", scores["word_count"] > 0)
    check(f"questions > 0 (got {scores['questions']})", scores["questions"] > 0)
    check(f"second_person > 0 (got {scores['second_person']})", scores["second_person"] > 0)
    check(f"first_person > 0 (got {scores['first_person']})", scores["first_person"] > 0)
    check("hedging key present", "hedging" in scores)

    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed:
        print("SOME CHECKS FAILED — review above.")
        return False
    print("All checks passed.")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# CLI — SCORE CSVs
# ═══════════════════════════════════════════════════════════════════════════════

def find_response_csvs(target_dir: Path) -> list[Path]:
    """Find all responses_*.csv files in the target directory."""
    files = sorted(target_dir.glob("responses_*.csv"))
    return files


def main() -> None:
    """Run the Tier 1 scoring pipeline."""
    parser = argparse.ArgumentParser(
        description="Score Component 2 responses with Tier 1 deterministic metrics"
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("output/clean_responses"),
        help="Directory containing responses_*.csv (default: output/clean_responses)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/score_deterministic"),
        help="Output directory (default: output/score_deterministic)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite output file if it exists",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run validation checks and exit",
    )
    args = parser.parse_args()

    if args.validate:
        success = run_validation()
        sys.exit(0 if success else 1)

    # Discover input files
    csv_files = find_response_csvs(args.input_dir)
    if not csv_files:
        logger.error("No responses_*.csv files found in %s", args.input_dir)
        raise SystemExit(1)

    logger.info("Found %d response file(s) in %s", len(csv_files), args.input_dir)
    for f in csv_files:
        logger.info("  %s", f.name)

    # Check output path
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "scored_deterministic.csv"
    if output_path.exists() and not args.force:
        logger.error("%s already exists. Use --force to overwrite.", output_path)
        raise SystemExit(1)

    # Score all files
    total_rows = 0

    with open(output_path, "w", newline="", encoding="utf-8") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for csv_file in csv_files:
            file_rows = 0

            with open(csv_file, encoding="utf-8") as in_fh:
                for row in csv.DictReader(in_fh):
                    text = row.get("response_text", "")
                    if not text or not text.strip():
                        logger.warning(
                            "Empty response_text: %s run %s — skipping",
                            row.get("vignette_id"), row.get("run_number"),
                        )
                        continue

                    scores = score_response(text)

                    out_row = {col: row[col] for col in ID_COLUMNS}
                    out_row["model_short"] = model_short_name(row["model"])
                    out_row.update(scores)

                    writer.writerow(out_row)
                    file_rows += 1

            total_rows += file_rows
            logger.info("  Scored %d rows from %s", file_rows, csv_file.name)

    expected = len(csv_files) * 600
    logger.info(
        "Done: %d rows scored (expected %d), output: %s",
        total_rows, expected, output_path,
    )
    if total_rows != expected:
        logger.warning("Row count mismatch — expected %d, got %d", expected, total_rows)


if __name__ == "__main__":
    main()
