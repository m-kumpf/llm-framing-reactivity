#!/usr/bin/env python3
"""Collect IPIP Big-Five personality questionnaire responses from LLMs via OpenRouter API."""

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import re
import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

from utils import model_output_path

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# CSV column order
CSV_COLUMNS = [
    "model", "item_number", "repetition", "raw_response", "parsed_response",
    "timestamp", "temperature", "status", "error_message",
]


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_items(items_path: Path) -> pd.DataFrame:
    """Load IPIP Big-Five items from CSV."""
    df = pd.read_csv(items_path, dtype={"item_number": int, "keying": str})
    assert len(df) == 50, f"Expected 50 items, got {len(df)}"
    return df

def load_existing_responses(output_path: Path, retry_errors: bool = False) -> set[tuple[str, int, int]]:
    """Load existing (model, item_number, repetition) tuples for resumption."""
    if not output_path.exists():
        return set()
    df = pd.read_csv(output_path, dtype={"item_number": int, "repetition": int, "model": str})
    if retry_errors:
        df = df[df["status"] == "success"]
    existing = set(zip(df["model"], df["item_number"], df["repetition"]))
    logger.info("Found %d existing responses in %s", len(existing), output_path)
    return existing

def parse_likert_response(raw: str) -> int | None:
    """Extract integer 1-5 from a response: exactly one standalone digit 1-5."""
    text = str(raw).strip() if raw else ""
    matches = re.findall(r'\b([1-5])\b', text)
    if len(matches) == 1:
        return int(matches[0])
    return None

def call_openrouter(
    model: str,
    item_text: str,
    config: dict,
    temperature: float,
    api_key: str,
) -> dict:
    """Make a single API call to OpenRouter with exponential backoff."""
    api_cfg = config["api"]
    prompts = config["prompts"]

    user_prompt = prompts["user"].replace("{item_text}", item_text)

    max_tokens = api_cfg.get("max_tokens", config["defaults"]["max_tokens"])

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    max_retries = api_cfg["max_retries"]
    backoff_base = api_cfg["backoff_base"]
    backoff_max = api_cfg["backoff_max"]
    jitter_max = api_cfg["jitter_max"]

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                api_cfg["base_url"],
                headers=headers,
                json=payload,
                timeout=30,
            )

            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < max_retries:
                    wait = min(backoff_base ** attempt, backoff_max) + random.uniform(0, jitter_max)
                    logger.warning(
                        "HTTP %d from %s, retrying in %.1fs (attempt %d/%d)",
                        resp.status_code, model, wait, attempt + 1, max_retries,
                    )
                    time.sleep(wait)
                    continue
                logger.error("HTTP %d from %s (final attempt): %s", resp.status_code, model, resp.text)
                resp.raise_for_status()

            if not resp.ok:
                logger.error("HTTP %d from %s: %s", resp.status_code, model, resp.text)
                resp.raise_for_status()
            data = resp.json()
            raw_response = data["choices"][0]["message"]["content"]

            content_str = str(raw_response).strip() if raw_response else ""
            parsed = parse_likert_response(content_str)

            if parsed is None:
                if attempt < max_retries:
                    wait = min(backoff_base ** attempt, backoff_max) + random.uniform(0, jitter_max)
                    logger.warning(
                        "Invalid response from %s (attempt %d/%d): '%s', retrying in %.1fs",
                        model, attempt + 1, max_retries, content_str[:80], wait,
                    )
                    time.sleep(wait)
                    continue
                logger.error(
                    "No valid 1-5 response from %s after %d attempts. Last response: '%s'",
                    model, max_retries + 1, content_str[:80],
                )
                return {
                    "raw_response": raw_response or "",
                    "parsed_response": "",
                    "status": "error",
                    "error_message": f"No valid 1-5 response after {max_retries + 1} attempts",
                }

            return {
                "raw_response": raw_response,
                "parsed_response": parsed,
                "status": "success",
                "error_message": "",
            }

        except Exception as e:
            if attempt < max_retries and ("429" in str(e) or "500" in str(e) or "502" in str(e) or "503" in str(e)):
                wait = min(backoff_base ** attempt, backoff_max) + random.uniform(0, jitter_max)
                logger.warning("Error: %s, retrying in %.1fs", e, wait)
                time.sleep(wait)
                continue
            logger.error("API call failed for %s: %s", model, e)
            return {"raw_response": "", "parsed_response": "", "status": "error", "error_message": str(e)}

    return {"raw_response": "", "parsed_response": "", "status": "error", "error_message": "Max retries exceeded"}


def save_response_row(output_path: Path, row: dict, write_header: bool) -> None:
    """Append a single response row to the CSV file."""
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_run_metadata(
    output_dir: Path,
    models: list[str],
    temperature: float,
    repeats: int,
    config: dict,
) -> None:
    """Save run metadata to JSON for reproducibility."""
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models": models,
        "temperature": temperature,
        "repeats": repeats,
        "config": config,
    }
    meta_path = output_dir / "run_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Run metadata saved to %s", meta_path)


def main() -> None:
    """Run the data collection pipeline."""
    parser = argparse.ArgumentParser(description="Collect IPIP Big-Five responses from LLMs")
    parser.add_argument("--output-dir", type=Path, default=Path("output/collect-responses"), help="Output directory for per-model CSVs")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model identifiers (overrides config)")
    parser.add_argument("--repeats", type=int, default=None, help="Number of repetitions per item per model")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Config YAML path")
    parser.add_argument("--items", type=Path, default=Path("ipip_big5_50_items.csv"), help="Items CSV path")
    parser.add_argument("--retry-errors", action="store_true", help="Re-run calls that previously failed")
    args = parser.parse_args()

    config = load_config(args.config)

    # Resolve parameters: CLI overrides config defaults
    models = args.models.split(",") if args.models else config["models"]
    repeats = args.repeats if args.repeats is not None else config["defaults"]["repeats"]
    temperature = args.temperature if args.temperature is not None else config["defaults"]["temperature"]

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY environment variable not set")
        raise SystemExit(1)

    items_df = load_items(args.items)

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save run metadata
    save_run_metadata(args.output_dir, models, temperature, repeats, config)

    rate_limit_delay = config["api"].get("rate_limit_delay", 0.5)
    calls_per_model = len(items_df) * repeats
    total_calls = len(models) * calls_per_model
    completed = 0
    errors_total = 0
    run_start = time.monotonic()

    logger.info(
        "Starting collection: %d models x %d items x %d reps = %d calls",
        len(models), len(items_df), repeats, total_calls,
    )

    for model_idx, model in enumerate(models, 1):
        output_path = model_output_path(args.output_dir, "raw_responses", model)
        existing = load_existing_responses(output_path, retry_errors=args.retry_errors)
        write_header = not output_path.exists()

        skipped = len(existing)
        completed += skipped
        model_new = 0
        model_errors = 0
        model_start = time.monotonic()
        short_name = model.split("/")[-1]

        logger.info(
            "[%d/%d] %s — %d existing, %d to collect",
            model_idx, len(models), model, skipped, calls_per_model - skipped,
        )

        for _, item in items_df.iterrows():
            item_num = int(item["item_number"])
            item_text = item["item_text"]

            for rep in range(1, repeats + 1):
                if (model, item_num, rep) in existing:
                    continue

                result = call_openrouter(model, item_text, config, temperature, api_key)

                row = {
                    "model": model,
                    "item_number": item_num,
                    "repetition": rep,
                    "raw_response": result["raw_response"],
                    "parsed_response": result["parsed_response"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "temperature": temperature,
                    "status": result["status"],
                    "error_message": result["error_message"],
                }

                save_response_row(output_path, row, write_header)
                write_header = False
                completed += 1
                model_new += 1
                if result["status"] != "success":
                    model_errors += 1

                # Compact overwriting progress line
                elapsed = time.monotonic() - run_start
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total_calls - completed) / rate if rate > 0 else 0
                sys.stderr.write(
                    f"\r  {short_name}  {model_new}/{calls_per_model - skipped}"
                    f"  |  overall {completed}/{total_calls} ({100*completed/total_calls:.0f}%)"
                    f"  |  {rate:.1f} calls/s  ETA {int(eta//60)}m{int(eta%60):02d}s   "
                )
                sys.stderr.flush()

                time.sleep(rate_limit_delay)

        model_elapsed = time.monotonic() - model_start
        errors_total += model_errors
        sys.stderr.write("\r" + " " * 100 + "\r")  # clear progress line
        sys.stderr.flush()
        logger.info(
            "[%d/%d] %s — done: %d collected, %d errors, %.0fs",
            model_idx, len(models), short_name, model_new, model_errors, model_elapsed,
        )

    run_elapsed = time.monotonic() - run_start
    logger.info(
        "Collection complete: %d responses, %d errors, %.0fs total (%.1f calls/s)",
        completed, errors_total, run_elapsed,
        completed / run_elapsed if run_elapsed > 0 else 0,
    )


if __name__ == "__main__":
    main()