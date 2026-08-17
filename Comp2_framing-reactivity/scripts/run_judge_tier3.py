#!/usr/bin/env python3
"""
Tier-3 LLM-as-judge scoring for Component 2.

Sends each cleaned (patient -> response) pair to an OpenAI-compatible
judge endpoint, parses the strict-JSON 13-item score object, and writes
per-model CSVs. The production run used deepseek-v4-flash (see
run-judge-tier3-deepseek/SUMMARY.md).

Design:
  - One API call per response. The strict JSON schema (built from
    codebook.json via build_response_schema) hard-guarantees 13
    integer keys in {1,2,3}.
  - Resumable: on startup, scans existing per-model CSVs and skips
    rows whose (model, vignette_id, run_number) tuple is already done.
  - Exponential-backoff retry on transient/network errors.
  - finish_reason="length" is treated as a HARD FAILURE (the response
    was truncated and may be missing items) — flagged loudly and
    written with status="error" rather than truncated scores.
  - Judge-as-parameter: --judge-model and --judge-endpoint come from
    the CLI, so a different judge is a one-flag change that writes to
    a parallel directory via --output-dir.
  - Blind by construction: build_prompt() takes only patient + response
    text. The script never passes metadata into the prompt assembler.

Usage:

    # Smoke test (3 rows spanning framings + models)
    python scripts/run_judge_tier3.py --smoke

    # Restrict to one model, first 10 rows
    python scripts/run_judge_tier3.py \\
        --models anthropic/claude-sonnet-4.6 --max-rows 10

    # Full run (4,200 rows)
    python scripts/run_judge_tier3.py

    # Dry run — list what would be scored, no API calls
    python scripts/run_judge_tier3.py --dry-run

The judge endpoint and key come from --judge-endpoint / JUDGE_ENDPOINT
and JUDGE_API_KEY (via .env or the environment).

Resume is automatic: re-running the same command picks up where it
left off. To force a re-score, delete the per-model CSV(s) under the
output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

SCRIPTS_DIR = Path(__file__).resolve().parent
COMP2_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from build_judge_prompt import (  # noqa: E402
    build_prompt,
    build_response_schema,
    load_codebook,
    load_fewshots,
)

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(COMP2_ROOT / ".env")
except ImportError:
    pass  # dotenv is optional; user can export the key manually


# ───────────────────────────────────────────────────────────────────────────
# Constants
# ───────────────────────────────────────────────────────────────────────────

DEFAULT_INPUT_DIR = COMP2_ROOT / "output" / "clean_responses"
DEFAULT_OUTPUT_DIR = COMP2_ROOT / "output" / "run-judge-tier3"
DEFAULT_JUDGE_MODEL = "deepseek-v4-flash"
DEFAULT_TEMPERATURE = 0.01
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_MAX_TOKENS = 16384

# Retry policy for transient/network errors
MAX_RETRIES = 5
BACKOFF_BASE = 2.0    # seconds; capped at BACKOFF_MAX
BACKOFF_MAX = 60.0
BACKOFF_JITTER = 0.25  # ± fraction

# Smoke test: deterministic 3-row selection spanning framings + models
SMOKE_PICKS = [
    ("anthropic/claude-sonnet-4.6", "A"),  # Anxious / Catastrophising
    ("openai/gpt-5.3-chat",          "D"),  # Hyper-rational
    ("minimax/minimax-m2.7",         "B"),  # Stoic / Minimal
]

LOG_FMT = "%(asctime)s %(levelname)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%H:%M:%S")
logger = logging.getLogger("run_judge_tier3")


# ───────────────────────────────────────────────────────────────────────────
# Row identity + IO helpers
# ───────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RowKey:
    model: str
    vignette_id: str
    run_number: str  # kept as string to match CSV faithfully


def sanitize_model_name(model_id: str) -> str:
    """Map e.g. 'anthropic/claude-sonnet-4.6' -> 'anthropic__claude-sonnet-4.6'.

    Slashes become double underscores; dots are preserved (matches the
    Comp1 score-responses naming).
    """
    return model_id.replace("/", "__")


def output_path_for(model_id: str, output_dir: Path) -> Path:
    return output_dir / f"judge_{sanitize_model_name(model_id)}.csv"


def iter_input_rows(input_dir: Path) -> Iterator[dict]:
    csvs = sorted(input_dir.glob("responses_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No responses_*.csv files found in {input_dir}")
    for csv_path in csvs:
        with csv_path.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                yield row


def load_completed_keys(output_dir: Path) -> set[RowKey]:
    """Scan existing per-model judge CSVs and return the set of completed keys.

    A row is treated as completed if it has a non-empty 'status' that is not
    'error'. Errors are re-tried on resume.
    """
    completed: set[RowKey] = set()
    if not output_dir.exists():
        return completed
    for path in sorted(output_dir.glob("judge_*.csv")):
        with path.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                status = (row.get("status") or "").strip()
                if status and status != "error":
                    completed.add(
                        RowKey(
                            model=row["model"],
                            vignette_id=row["vignette_id"],
                            run_number=str(row["run_number"]),
                        )
                    )
    return completed


# ───────────────────────────────────────────────────────────────────────────
# OpenAI client + scoring
# ───────────────────────────────────────────────────────────────────────────

def make_client(endpoint: str, api_key: str):
    from openai import OpenAI  # imported lazily so --help / --dry-run don't require it
    return OpenAI(api_key=api_key, base_url=endpoint)


def _sleep_backoff(attempt: int) -> None:
    wait = min(BACKOFF_BASE ** attempt, BACKOFF_MAX)
    wait *= (1.0 + random.uniform(-BACKOFF_JITTER, BACKOFF_JITTER))
    time.sleep(max(wait, 0.1))


def call_judge(
    client,
    judge_model: str,
    messages: list[dict],
    schema: dict,
    *,
    max_tokens: int,
    temperature: float,
    reasoning_effort: str,
) -> dict:
    """One scoring call. Returns a dict with the parsed scores + metadata.

    Shape of returned dict:
        {
            "scores": {item_key: int, ...} or None on failure,
            "finish_reason": str,
            "prompt_tokens": int,
            "completion_tokens": int,
            "reasoning_content": str,
            "status": "ok" | "error",
            "error_message": str,
        }
    """
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=judge_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ruben_13_item_scores",
                        "strict": True,
                        "schema": schema,
                    },
                },
            )
            choice = resp.choices[0]
            finish_reason = choice.finish_reason or ""
            usage = getattr(resp, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            # some judge backends return a separate reasoning_content field
            reasoning_content = getattr(choice.message, "reasoning_content", "") or ""

            if finish_reason == "length":
                msg = (
                    f"finish_reason='length' — response truncated at "
                    f"max_tokens={max_tokens}; treating as failure"
                )
                logger.warning("  %s", msg)
                return {
                    "scores": None,
                    "finish_reason": finish_reason,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "reasoning_content": reasoning_content,
                    "status": "error",
                    "error_message": msg,
                }

            content = choice.message.content or ""
            try:
                scores = json.loads(content)
            except json.JSONDecodeError as e:
                # With strict schema the server should never emit invalid JSON,
                # but defensively retry once on a transient hiccup.
                msg = f"JSON parse error: {e}; content head={content[:200]!r}"
                if attempt < MAX_RETRIES:
                    logger.warning("  %s — retry %d/%d", msg, attempt + 1, MAX_RETRIES)
                    _sleep_backoff(attempt)
                    continue
                return {
                    "scores": None,
                    "finish_reason": finish_reason,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "reasoning_content": reasoning_content,
                    "status": "error",
                    "error_message": msg,
                }

            return {
                "scores": scores,
                "finish_reason": finish_reason,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "reasoning_content": reasoning_content,
                "status": "ok",
                "error_message": "",
            }

        except Exception as e:  # network/timeout/server errors → backoff retry
            last_err = e
            if attempt < MAX_RETRIES:
                logger.warning(
                    "  API error (attempt %d/%d): %s — backing off",
                    attempt + 1, MAX_RETRIES, e,
                )
                _sleep_backoff(attempt)
                continue
            break

    return {
        "scores": None,
        "finish_reason": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_content": "",
        "status": "error",
        "error_message": f"exhausted {MAX_RETRIES} retries: {last_err!r}",
    }


# ───────────────────────────────────────────────────────────────────────────
# CSV writer
# ───────────────────────────────────────────────────────────────────────────

def build_output_columns(item_keys: list[str]) -> list[str]:
    return (
        ["model", "scenario_id", "framing_id", "vignette_id", "run_number"]
        + list(item_keys)
        + [
            "judge_model",
            "reasoning_effort",
            "temperature",
            "finish_reason",
            "prompt_tokens",
            "completion_tokens",
            "reasoning_content",
            "status",
            "error_message",
            "timestamp",
        ]
    )


def open_appender(out_path: Path, columns: list[str]):
    """Return (file_handle, csv.DictWriter). Writes header iff file is new/empty."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = (not out_path.exists()) or out_path.stat().st_size == 0
    fh = out_path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(fh, fieldnames=columns)
    if new_file:
        writer.writeheader()
        fh.flush()
    return fh, writer


# ───────────────────────────────────────────────────────────────────────────
# Row filtering / smoke selection
# ───────────────────────────────────────────────────────────────────────────

def _matches_filter(row: dict, models: set[str] | None, framings: set[str] | None) -> bool:
    if models is not None and row["model"] not in models:
        return False
    if framings is not None and row["framing_id"] not in framings:
        return False
    return True


def select_smoke_rows(input_dir: Path) -> list[dict]:
    """Pick the first matching row for each (model, framing_id) in SMOKE_PICKS."""
    want = {pick: None for pick in SMOKE_PICKS}
    for row in iter_input_rows(input_dir):
        key = (row["model"], row["framing_id"])
        if key in want and want[key] is None:
            want[key] = row
            if all(v is not None for v in want.values()):
                break
    missing = [k for k, v in want.items() if v is None]
    if missing:
        raise RuntimeError(f"Smoke selection: could not find rows for {missing}")
    return [want[k] for k in SMOKE_PICKS]  # type: ignore[index]


# ───────────────────────────────────────────────────────────────────────────
# Main scoring loop
# ───────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> int:
    codebook = load_codebook()
    fewshots = load_fewshots()
    item_keys = [it["key"] for it in codebook["items"]]
    schema = build_response_schema(codebook)
    columns = build_output_columns(item_keys)

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    # Build row stream
    if args.smoke:
        rows: list[dict] = select_smoke_rows(input_dir)
        logger.info("Smoke mode: %d hand-picked rows", len(rows))
    else:
        models_filter = set(args.models) if args.models else None
        framings_filter = set(args.framings) if args.framings else None
        rows = [
            r for r in iter_input_rows(input_dir)
            if _matches_filter(r, models_filter, framings_filter)
        ]
        if args.max_rows is not None:
            rows = rows[: args.max_rows]
        logger.info(
            "Selected %d rows (models=%s, framings=%s, max_rows=%s)",
            len(rows), args.models or "all", args.framings or "all", args.max_rows,
        )

    # Resume: drop already-completed rows
    completed = load_completed_keys(output_dir)
    if completed:
        before = len(rows)
        rows = [
            r for r in rows
            if RowKey(r["model"], r["vignette_id"], str(r["run_number"])) not in completed
        ]
        logger.info(
            "Resume: skipping %d already-completed rows (remaining: %d)",
            before - len(rows), len(rows),
        )

    if args.dry_run:
        logger.info("Dry run — no API calls. First 5 row keys:")
        for r in rows[:5]:
            logger.info(
                "  model=%s framing=%s vignette=%s run=%s",
                r["model"], r["framing_id"], r["vignette_id"], r["run_number"],
            )
        return 0

    if not rows:
        logger.info("Nothing to do. Exiting.")
        return 0

    # Endpoint + API key required from here on
    endpoint = args.judge_endpoint or os.environ.get("JUDGE_ENDPOINT") or ""
    if not endpoint:
        logger.error(
            "No judge endpoint. Pass --judge-endpoint or set JUDGE_ENDPOINT "
            "(any OpenAI-compatible base URL)."
        )
        return 2
    api_key = os.environ.get("JUDGE_API_KEY") or ""
    if not api_key:
        logger.error(
            "JUDGE_API_KEY not set. Add it to .env or export it before running."
        )
        return 2

    client = make_client(endpoint, api_key)

    # Group rows by model so we open one appender per per-model file
    writers: dict[str, tuple] = {}  # model_id -> (fh, writer)

    try:
        n_ok = 0
        n_err = 0
        n_length = 0
        for i, row in enumerate(rows, start=1):
            model_id = row["model"]
            if model_id not in writers:
                out_path = output_path_for(model_id, output_dir)
                writers[model_id] = open_appender(out_path, columns)

            fh, writer = writers[model_id]

            patient = row["prompt"]
            response = row["response_text"]
            messages = build_prompt(patient, response, codebook, fewshots)

            logger.info(
                "[%d/%d] %s framing=%s vignette=%s run=%s",
                i, len(rows), model_id, row["framing_id"],
                row["vignette_id"], row["run_number"],
            )

            result = call_judge(
                client,
                judge_model=args.judge_model,
                messages=messages,
                schema=schema,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                reasoning_effort=args.reasoning_effort,
            )

            # Build output row
            out_row = {
                "model": model_id,
                "scenario_id": row["scenario_id"],
                "framing_id": row["framing_id"],
                "vignette_id": row["vignette_id"],
                "run_number": row["run_number"],
                "judge_model": args.judge_model,
                "reasoning_effort": args.reasoning_effort,
                "temperature": args.temperature,
                "finish_reason": result["finish_reason"],
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "reasoning_content": result["reasoning_content"],
                "status": result["status"],
                "error_message": result["error_message"],
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            if result["scores"]:
                for k in item_keys:
                    out_row[k] = result["scores"].get(k, "")
            else:
                for k in item_keys:
                    out_row[k] = ""

            writer.writerow(out_row)
            fh.flush()

            if result["status"] == "ok":
                n_ok += 1
            else:
                n_err += 1
            if result["finish_reason"] == "length":
                n_length += 1

        logger.info(
            "Done. ok=%d  error=%d  length-truncated=%d  total=%d",
            n_ok, n_err, n_length, n_ok + n_err,
        )
        return 0 if n_err == 0 else 1

    finally:
        for fh, _ in writers.values():
            fh.close()


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Tier-3 LLM-as-judge scoring for Component 2 (Ruben 13-item).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
        help=f"Directory of cleaned responses CSVs (default: {DEFAULT_INPUT_DIR.relative_to(COMP2_ROOT)})",
    )
    p.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for per-model judge CSVs (default: {DEFAULT_OUTPUT_DIR.relative_to(COMP2_ROOT)})",
    )
    p.add_argument(
        "--judge-model", type=str, default=DEFAULT_JUDGE_MODEL,
        help=f"Judge model name (default: {DEFAULT_JUDGE_MODEL})",
    )
    p.add_argument(
        "--judge-endpoint", type=str, default=None,
        help="OpenAI-compatible base URL (default: env JUDGE_ENDPOINT)",
    )
    p.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})",
    )
    p.add_argument(
        "--reasoning-effort", type=str, default=DEFAULT_REASONING_EFFORT,
        choices=["low", "medium", "high"],
        help=f"Top-level reasoning_effort (default: {DEFAULT_REASONING_EFFORT})",
    )
    p.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help=f"max_tokens per call (default: {DEFAULT_MAX_TOKENS})",
    )
    p.add_argument(
        "--models", nargs="*", default=None,
        help="Filter: only score these subject-model IDs (space-separated).",
    )
    p.add_argument(
        "--framings", nargs="*", default=None,
        help="Filter: only score these framing IDs (A B C D E F).",
    )
    p.add_argument(
        "--max-rows", type=int, default=None,
        help="Cap total rows processed (after filters, before resume-skip).",
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="Smoke mode: 3 hand-picked rows across framings + models.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="List what would be scored; make no API calls.",
    )
    return p


def main() -> int:
    args = build_argparser().parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
