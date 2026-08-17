#!/usr/bin/env python3
"""
Drift test — guarantees codebook.json `items` are byte-identical to the
RUBRIC array in every rater UI (pilot and validation).

If this test ever fails, the judge prompt and the human rater UI have
diverged and the human-vs-judge reliability comparison is invalid until
the two are reconciled.

Also cross-checks the codebook keys against the score keys of every
few-shot example in tier3_fewshot_examples.json.

Stdlib only. Run from the Comp2 root:

    python scripts/test_codebook_matches_ui.py

Exit code 0 on pass, 1 on any failure (with a diff-style report).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

COMP2_ROOT = Path(__file__).resolve().parent.parent
# Every rater UI is checked against the codebook. Missing files are skipped so
# the test still passes before the validation UI has been built.
UI_PATHS = [
    COMP2_ROOT / "output" / "sample_pilot" / "rater_ui.html",
    COMP2_ROOT / "output" / "sample_validation" / "rater_ui.html",
]
CODEBOOK_PATH = COMP2_ROOT / "codebook.json"
FEWSHOT_PATH = COMP2_ROOT / "tier3_fewshot_examples.json"

ITEM_FIELDS = ("key", "n", "label", "definition", "example")
RUBRIC_LITERAL_PAT = re.compile(r"^const RUBRIC = (\[.*\]);\s*$", re.MULTILINE)


def extract_rubric_from_ui(ui_text: str, ui_path: Path) -> list[dict]:
    match = RUBRIC_LITERAL_PAT.search(ui_text)
    if match is None:
        raise RuntimeError(
            f"Could not locate `const RUBRIC = [...];` in {ui_path}. "
            "The rater UI's rubric literal format may have changed."
        )
    return json.loads(match.group(1))


def load_codebook_items(codebook_path: Path) -> list[dict]:
    data = json.loads(codebook_path.read_text(encoding="utf-8"))
    return data["items"]


def load_fewshot_examples(fewshot_path: Path) -> list[dict]:
    return json.loads(fewshot_path.read_text(encoding="utf-8"))["examples"]


def compare_items(ui_items: list[dict], cb_items: list[dict]) -> list[str]:
    """Return a list of failure messages. Empty list means full agreement."""
    failures: list[str] = []

    if len(ui_items) != len(cb_items):
        failures.append(
            f"Length mismatch: rater_ui has {len(ui_items)} items, "
            f"codebook has {len(cb_items)}"
        )
        return failures

    for idx, (ui_item, cb_item) in enumerate(zip(ui_items, cb_items)):
        for field in ITEM_FIELDS:
            ui_val = ui_item.get(field)
            cb_val = cb_item.get(field)
            if ui_val != cb_val:
                failures.append(
                    f"Item index {idx} ({ui_item.get('key', '?')}) field "
                    f"'{field}' differs:\n"
                    f"    ui:       {ui_val!r}\n"
                    f"    codebook: {cb_val!r}"
                )
    return failures


def check_n_sequence(items: list[dict]) -> list[str]:
    expected = list(range(1, 14))
    actual = [it.get("n") for it in items]
    if actual != expected:
        return [f"`n` sequence is {actual}, expected {expected}"]
    return []


def check_against_fewshot(items: list[dict], examples: list[dict]) -> list[str]:
    """Every few-shot example must score exactly the codebook's item keys."""
    failures: list[str] = []
    cb_keys = [it["key"] for it in items]
    for idx, ex in enumerate(examples, start=1):
        ex_keys = list(ex.get("scores", {}).keys())
        if ex_keys != cb_keys:
            failures.append(
                f"Few-shot example {idx} score keys disagree with codebook:\n"
                f"    codebook: {cb_keys}\n"
                f"    fewshot:  {ex_keys}"
            )
    return failures


def main() -> int:
    present_uis = [p for p in UI_PATHS if p.exists()]
    for p in UI_PATHS:
        print(f"UI:        {p}{'' if p.exists() else '  (missing — skipped)'}")
    print(f"Codebook:  {CODEBOOK_PATH}")
    print(f"Fewshot:   {FEWSHOT_PATH}")
    print()

    if not present_uis:
        print(f"FAIL — none of the rater UIs exist: {[str(p) for p in UI_PATHS]}")
        return 1

    cb_items = load_codebook_items(CODEBOOK_PATH)
    fewshot_examples = load_fewshot_examples(FEWSHOT_PATH)

    failures: list[str] = []
    for ui_path in present_uis:
        ui_items = extract_rubric_from_ui(ui_path.read_text(encoding="utf-8"), ui_path)
        failures += [f"[{ui_path.name}] {m}" for m in compare_items(ui_items, cb_items)]
    failures += check_n_sequence(cb_items)
    failures += check_against_fewshot(cb_items, fewshot_examples)

    if failures:
        print(f"FAIL — {len(failures)} discrepancy/discrepancies:\n")
        for msg in failures:
            print("  -", msg)
            print()
        return 1

    print(
        f"PASS — codebook items[] is byte-identical to RUBRIC in "
        f"{len(present_uis)} UI(s) ({len(cb_items)} items), "
        "n is 1..13, few-shot score keys match the codebook."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
