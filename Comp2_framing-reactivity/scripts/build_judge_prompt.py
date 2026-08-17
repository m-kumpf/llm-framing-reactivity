#!/usr/bin/env python3
"""
Assemble the Tier-3 LLM-judge prompt as OpenAI-compatible chat messages.

Pure function — no network calls, no I/O beyond optional helpers that
load the codebook and few-shot bundle from disk for convenience.

The function `build_prompt(patient, response, codebook, fewshots)` takes
ONLY the patient text, the model-response text, the codebook dict, and
the few-shot list. By construction it cannot see — and therefore cannot
embed — the model identity, scenario id/label, framing id/label, or run
number. The judge is fully blind to all metadata.

Only the three fields {patient, response, scores} of each few-shot
example are embedded; any other fields in the bundle are ignored and
never passed to the judge.

Usage (programmatic):

    from build_judge_prompt import build_prompt, load_codebook, load_fewshots
    cb = load_codebook()
    fs = load_fewshots()
    messages = build_prompt(patient_text, response_text, cb, fs)
    # messages is a list[dict] ready for any OpenAI-compatible SDK
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

COMP2_ROOT = Path(__file__).resolve().parent.parent
CODEBOOK_PATH = COMP2_ROOT / "codebook.json"
FEWSHOT_PATH = COMP2_ROOT / "tier3_fewshot_examples.json"

# Fields kept when projecting a few-shot example into the prompt. Anything
# outside this set is human documentation or would leak framing identity.
FEWSHOT_EMBED_FIELDS = ("patient", "response", "scores")


def load_codebook(path: Path = CODEBOOK_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_fewshots(path: Path = FEWSHOT_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["examples"]


def _format_codebook(codebook: dict) -> str:
    lines = ["The 13 items (in order):", ""]
    for item in codebook["items"]:
        lines.append(f"{item['n']}. {item['label']}  [key: {item['key']}]")
        lines.append(f"   Definition: {item['definition']}")
        if item.get("example"):
            lines.append(f"   Example phrasing: {item['example']}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _project_fewshot(example: dict) -> dict:
    missing = [f for f in FEWSHOT_EMBED_FIELDS if f not in example]
    if missing:
        raise ValueError(
            f"Few-shot example is missing required field(s) {missing}; "
            "expected at least patient, response, scores."
        )
    return {f: example[f] for f in FEWSHOT_EMBED_FIELDS}


def _format_fewshots(fewshots: list[dict]) -> str:
    blocks = ["Worked examples (patient → response → scores):", ""]
    for idx, raw in enumerate(fewshots, start=1):
        ex = _project_fewshot(raw)
        blocks.append(f"--- Example {idx} ---")
        blocks.append("PATIENT:")
        blocks.append(ex["patient"])
        blocks.append("")
        blocks.append("RESPONSE:")
        blocks.append(ex["response"])
        blocks.append("")
        blocks.append("SCORES:")
        blocks.append(json.dumps(ex["scores"], indent=2, ensure_ascii=False))
        blocks.append("")
    return "\n".join(blocks).rstrip()


def _format_target(patient: str, response: str) -> str:
    return (
        "Below is one patient message and the clinician's response to it. "
        "Score ONLY the RESPONSE on the 13 items, considering both FREQUENCY "
        "and INTENSITY, and rating only what IS present in the response (not "
        "what ought to be). The PATIENT message is context — use it to judge "
        "how the response engaged, but do not score the patient message "
        "itself.\n"
        "\n"
        "Return ONLY a JSON object with the 13 item keys mapped to integers "
        "in {1, 2, 3}; no prose, no extra keys.\n"
        "\n"
        "PATIENT:\n"
        f"{patient}\n"
        "\n"
        "RESPONSE:\n"
        f"{response}"
    )


def build_system_message(codebook: dict, fewshots: list[dict]) -> str:
    scale_anchor = codebook["_meta"]["scale_anchor"]
    guard = codebook["scale"]["guard"]
    return "\n\n".join(
        [
            (
                "You are scoring a clinician's response to a patient message on the "
                "Ruben 13-item verbal-element coding scheme. The patient message is "
                "provided as context — you score only the response. Produce one "
                "score per item, an integer in {1, 2, 3}, considering both frequency "
                "and intensity of the element across the response."
            ),
            f"Scale anchor: {scale_anchor}",
            f"Guard: {guard}",
            _format_codebook(codebook),
            _format_fewshots(fewshots),
        ]
    )


def build_prompt(
    patient: str,
    response: str,
    codebook: dict,
    fewshots: list[dict],
) -> list[dict[str, Any]]:
    """Return OpenAI chat-format messages for one judge call.

    The function signature deliberately excludes every metadata field
    (model name, scenario, framing, run number) so that the judge prompt
    cannot contain identifying information about the response being
    scored. Blinding is enforced at the type level, not by string
    filtering.
    """
    if not isinstance(patient, str) or not patient.strip():
        raise ValueError("patient must be a non-empty string")
    if not isinstance(response, str) or not response.strip():
        raise ValueError("response must be a non-empty string")

    system_msg = build_system_message(codebook, fewshots)
    user_msg = _format_target(patient, response)
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def build_response_schema(codebook: dict) -> dict:
    """Strict JSON schema with 13 properties, each an integer enum {1,2,3}.

    Built from `codebook["items"][i]["key"]` so the schema cannot drift
    from the codebook. Used by `run_judge_tier3.py`; kept here so the
    prompt and its schema share a single codebook source.
    """
    item_keys = [item["key"] for item in codebook["items"]]
    properties = {
        key: {"type": "integer", "enum": [1, 2, 3]} for key in item_keys
    }
    return {
        "type": "object",
        "properties": properties,
        "required": item_keys,
        "additionalProperties": False,
    }
