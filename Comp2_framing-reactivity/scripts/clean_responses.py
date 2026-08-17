#!/usr/bin/env python3
"""
Clean and validate Comp2 response CSVs.

For each responses_*.csv in the input directory:
  1. Drop rows where finish_reason is empty or "error"
  2. Drop rows where response_text starts with "ERROR" (case-insensitive)
  3. Deduplicate: for each (vignette_id, run_number), keep the last valid row
  4. Validate: exactly 600 rows (60 vignettes × 10 runs)
  5. Sort by run_number, then scenario_id, then framing_id
  6. Save to output directory (originals untouched)

Usage:
    python clean_responses.py                          # default dirs
    python clean_responses.py --input-dir path/to/raw --output-dir path/to/clean

example:
python clean_responses.py --input-dir output/collect_responses/04.04.26 --output-dir output/clean_responses

"""

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def is_valid_row(row):
    """Return True if the row represents a usable response."""
    # Reject empty or error finish_reason
    fr = (row.get("finish_reason") or "").strip()
    if not fr or fr == "error":
        return False

    # Reject ERROR prefix in response_text
    text = (row.get("response_text") or "").strip()
    if text.upper().startswith("ERROR"):
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Clean and validate Comp2 response CSVs")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Directory containing responses_*.csv "
                             "(defaults to output/collect_responses)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for cleaned CSVs "
                             "(defaults to output/clean_responses)")
    args = parser.parse_args()

    input_dir = args.input_dir or str(PROJECT_ROOT / "output" / "collect_responses")
    output_dir = args.output_dir or str(PROJECT_ROOT / "output" / "clean_responses")

    # collect_responses.py writes into a date subdirectory, so search one
    # level deep as well as the directory itself.
    csv_files = sorted(
        glob.glob(os.path.join(input_dir, "responses_*.csv"))
        + glob.glob(os.path.join(input_dir, "*", "responses_*.csv"))
    )
    if not csv_files:
        print(f"No responses_*.csv files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Found {len(csv_files)} file(s)\n")

    all_ok = True

    for filepath in csv_files:
        filename = os.path.basename(filepath)
        print(f"{'─'*60}")
        print(f"  {filename}")
        print(f"{'─'*60}")

        with open(filepath, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            all_rows = list(reader)

        print(f"  Raw rows: {len(all_rows)}")

        # ── Stage 1: filter invalid rows ─────────────────────────────
        valid_rows = []
        dropped = []
        for idx, r in enumerate(all_rows, start=1):
            if is_valid_row(r):
                valid_rows.append((idx, r))
            else:
                fr = (r.get("finish_reason") or "").strip()
                text_preview = (r.get("response_text") or "")[:80]
                dropped.append((idx, r.get("vignette_id", "?"),
                                r.get("run_number", "?"), fr, text_preview))

        if dropped:
            print(f"  Dropped: {len(dropped)}")
            for idx, vid, run, fr, preview in dropped:
                reason = f"finish_reason='{fr}'"
                if preview.strip().upper().startswith("ERROR"):
                    reason = "ERROR response"
                print(f"    row {idx}: {vid} run {run} — {reason}")

        # ── Stage 2: deduplicate (last valid row per cell) ───────────
        by_key = {}
        for orig_idx, r in valid_rows:
            key = (r["vignette_id"], r["run_number"])
            by_key[key] = r  # last one wins

        deduped = list(by_key.values())
        n_dupes = len(valid_rows) - len(deduped)
        if n_dupes:
            print(f"  Duplicates removed: {n_dupes}")

        # ── Stage 3: validate completeness ───────────────────────────
        expected_keys = {
            (f"S{s:02d}_{f}", str(run))
            for s in range(1, 11)
            for f in "ABCDEF"
            for run in range(1, 11)
        }
        actual_keys = {(r["vignette_id"], r["run_number"]) for r in deduped}
        missing = sorted(expected_keys - actual_keys)

        if len(deduped) == 600 and not missing:
            print("  OK: 600/600 valid rows")
        else:
            print(f"  WARNING: {len(deduped)}/600 valid rows")
            if missing:
                print(f"  Missing cells ({len(missing)}):")
                for vid, run in missing[:15]:
                    print(f"    {vid} run {run}")
                if len(missing) > 15:
                    print(f"    ... and {len(missing) - 15} more")
            all_ok = False

        # ── Stage 4: sort ────────────────────────────────────────────
        def sort_key(r):
            return (int(r["run_number"]),
                    r["scenario_id"],
                    r["framing_id"])

        deduped.sort(key=sort_key)

        # ── Stage 5: write ───────────────────────────────────────────
        out_path = os.path.join(output_dir, filename)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(deduped)

        print(f"  Saved: {out_path}")
        print()

    # ── Final summary ────────────────────────────────────────────────
    print(f"{'='*60}")
    if all_ok:
        print(f"All {len(csv_files)} files have 600 valid rows.")
    else:
        print("WARNING: some files have missing cells — see above.")
    print(f"Cleaned files in: {output_dir}")


if __name__ == "__main__":
    main()
