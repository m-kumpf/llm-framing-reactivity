#!/usr/bin/env python3
"""
Contract test for build_judge_prompt.build_prompt().

Verifies:
  1. Output shape (list of {role, content} dicts).
  2. All 13 codebook item keys appear in the system message.
  3. The scale anchor and guard appear in the system message.
  4. All few-shot SCORES objects appear; documentation-only fields
     (rationale, flags, ...) do NOT appear anywhere in the prompt.
  5. Blinding: no model identifier, no scenario_id (e.g. "S01_C"),
     no framing_label, no run_number appear anywhere in the prompt,
     using a real row sampled from output/clean_responses/.
  6. The target patient text and response text are present, in the
     user message, untruncated.

Stdlib only. Run from the Comp2 root:

    python scripts/test_build_judge_prompt.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from build_judge_prompt import (  # noqa: E402
    build_prompt,
    build_response_schema,
    load_codebook,
    load_fewshots,
)

COMP2_ROOT = SCRIPTS_DIR.parent
CLEAN_DIR = COMP2_ROOT / "output" / "clean_responses"


def _pick_sample_row() -> dict:
    """Read one row from one cleaned-response CSV for a realistic blinding test."""
    csvs = sorted(CLEAN_DIR.glob("responses_*.csv"))
    if not csvs:
        raise RuntimeError(f"No responses_*.csv files in {CLEAN_DIR}")
    with csvs[0].open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        row = next(reader)
    return row


def main() -> int:
    codebook = load_codebook()
    fewshots = load_fewshots()
    sample = _pick_sample_row()

    patient = sample["prompt"]
    response = sample["response_text"]
    messages = build_prompt(patient, response, codebook, fewshots)

    failures: list[str] = []

    # 1. Shape
    if not (
        isinstance(messages, list)
        and len(messages) == 2
        and all(isinstance(m, dict) and set(m.keys()) == {"role", "content"} for m in messages)
        and [m["role"] for m in messages] == ["system", "user"]
    ):
        failures.append(
            f"Shape wrong: got {[(m.get('role'), set(m.keys()) if isinstance(m, dict) else type(m).__name__) for m in messages]}"
        )

    system_text = messages[0]["content"]
    user_text = messages[1]["content"]
    full_text = system_text + "\n" + user_text

    # 2. All 13 item keys present in system
    item_keys = [it["key"] for it in codebook["items"]]
    missing_keys = [k for k in item_keys if k not in system_text]
    if missing_keys:
        failures.append(f"Item keys missing from system message: {missing_keys}")

    # 3. Scale anchor + guard present
    anchor = codebook["_meta"]["scale_anchor"]
    if anchor not in system_text:
        failures.append("scale_anchor missing from system message")
    if codebook["scale"]["guard"] not in system_text:
        failures.append("scale guard missing from system message")

    # 4a. All few-shot SCORES objects represented (count keys appearing in system)
    fewshot_scores_count = system_text.count('"item_01_validation_concern":')
    if fewshot_scores_count != len(fewshots):
        failures.append(
            f"Expected {len(fewshots)} few-shot scores blocks, "
            f"found {fewshot_scores_count} occurrences of item_01 in system"
        )

    # 4b. Forbidden few-shot fields must NOT appear
    forbidden_fewshot_fields = ["rationale", "flags", "teaching_purpose", "example_id"]
    for fld in forbidden_fewshot_fields:
        if f'"{fld}"' in full_text or f"'{fld}'" in full_text:
            failures.append(f"Forbidden few-shot field '{fld}' leaked into prompt")

    # 4c. Framing labels from few-shots must NOT appear anywhere
    fewshot_framing_labels = sorted({ex.get("framing", "") for ex in fewshots if ex.get("framing")})
    leaked_framings = [lbl for lbl in fewshot_framing_labels if lbl in full_text]
    if leaked_framings:
        failures.append(f"Few-shot framing label(s) leaked into prompt: {leaked_framings}")

    # 5. Blinding against the real-row metadata
    metadata_strings = {
        "model": sample["model"],                     # e.g. "anthropic/claude-sonnet-4.6"
        "scenario_id": sample["scenario_id"],         # e.g. "S01"
        "scenario_label": sample["scenario_label"],   # e.g. "New Type 2 Diabetes Diagnosis"
        "framing_label": sample["framing_label"],     # e.g. "Anxious / Catastrophising"
        "vignette_id": sample["vignette_id"],         # e.g. "S01_A"
    }
    for name, value in metadata_strings.items():
        if value and value in full_text:
            failures.append(
                f"Metadata '{name}' = {value!r} leaked into prompt"
            )

    # 6. Target patient + response present in user message
    if patient not in user_text:
        failures.append("Patient text not present verbatim in user message")
    if response not in user_text:
        failures.append("Response text not present verbatim in user message")

    # 6b. Disambiguating phrase present (guards against future regression
    # that re-introduces ambiguous "score the pair" wording)
    if "Score ONLY the RESPONSE" not in user_text:
        failures.append(
            "Disambiguating phrase 'Score ONLY the RESPONSE' missing from "
            "user message — patient-vs-response scoring scope is unclear"
        )
    if "context" not in user_text.lower():
        failures.append(
            "User message does not name the patient as 'context' — scoring "
            "scope is ambiguous"
        )

    # 7. Bonus: response schema matches codebook keys exactly
    schema = build_response_schema(codebook)
    if sorted(schema["properties"].keys()) != sorted(item_keys):
        failures.append("Response schema keys do not match codebook item keys")
    if schema["required"] != item_keys:
        failures.append("Response schema 'required' is not the codebook key list in order")
    if schema.get("additionalProperties") is not False:
        failures.append("Response schema must have additionalProperties: false")
    for k, prop in schema["properties"].items():
        if prop != {"type": "integer", "enum": [1, 2, 3]}:
            failures.append(f"Schema property for {k} is not integer enum [1,2,3]")

    # Reporting
    print(f"Sample row: model={sample['model']}, vignette_id={sample['vignette_id']}, run={sample['run_number']}")
    print(f"System message length: {len(system_text):,} chars")
    print(f"User message length:   {len(user_text):,} chars")
    print(f"Few-shot scores blocks found: {fewshot_scores_count}")
    print()

    if failures:
        print(f"FAIL — {len(failures)} issue(s):")
        for msg in failures:
            print("  -", msg)
        return 1

    print("PASS — prompt assembled correctly; blinding holds; schema matches codebook.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
