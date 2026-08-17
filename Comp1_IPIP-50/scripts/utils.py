#!/usr/bin/env python3
"""Shared utilities for the IPIP Big-Five personality testing pipeline."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def sanitize_model_name(model: str) -> str:
    """Convert model ID to filesystem-safe string: 'anthropic/claude-sonnet-4.6' -> 'anthropic__claude-sonnet-4.6'."""
    return model.replace("/", "__")


def unsanitize_model_name(sanitized: str) -> str:
    """Reverse of sanitize_model_name (first __ only)."""
    return sanitized.replace("__", "/", 1)


def model_output_path(output_dir: Path, stem: str, model: str) -> Path:
    """Build per-model output path, e.g. output/raw_responses_anthropic__claude-sonnet-4.6.csv."""
    return output_dir / f"{stem}_{sanitize_model_name(model)}.csv"


def discover_model_files(input_dir: Path, stem: str) -> dict[str, Path]:
    """Glob for stem_*.csv files and return {model_id: path} dict."""
    pattern = f"{stem}_*.csv"
    files = {}
    for path in sorted(input_dir.glob(pattern)):
        # Extract sanitized model name from filename: stem_<sanitized>.csv
        filename = path.stem  # e.g. "raw_responses_anthropic__claude-sonnet-4.6"
        prefix = f"{stem}_"
        if not filename.startswith(prefix):
            continue
        sanitized = filename[len(prefix):]
        model_id = unsanitize_model_name(sanitized)
        files[model_id] = path
    return files


def check_output_path(path: Path, force: bool) -> None:
    """Exit if output file exists and --force not set."""
    if path.exists():
        if force:
            logger.warning("Overwriting existing file: %s", path)
        else:
            logger.error("%s already exists. Use --force to overwrite.", path)
            raise SystemExit(1)
