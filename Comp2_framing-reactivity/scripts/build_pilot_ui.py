#!/usr/bin/env python3
"""
Build the human rater UI for the Tier 3 pilot or validation set.

Generates a single standalone HTML file that loads the sample, shows the
patient prompt and model response side-by-side, and collects ratings on all
13 Ruben items on the 3-point ordinal scale. The same HTML is distributed to
both raters; rater identity is captured at the top of the UI (typed into a
text field) and embedded in the exported JSON. Includes a sidebar reference
panel with the full rubric, localStorage persistence, keyboard navigation,
and a light/dark toggle.

Two modes (``--set``):
  * pilot (default)      — shows the sample meta bar (#id, scenario, framing).
  * validation (--blind) — FULLY BLINDED: the meta bar is removed entirely so
    raters see only the Patient Message and Model Response, like the LLM judge.
    Uses a separate localStorage key and, on first load, imports any existing
    in-progress pilot ratings once (sample_ids 1-21 are identical across sets).

With no flags the output is byte-identical to the original pilot UI, so the
codebook-drift test (which reads the pilot UI) is unaffected.

Input:  output/sample_pilot/pilot_sample.csv          (--set pilot)
        output/sample_validation/validation_sample.csv (--set validation)
Output: output/<sample_pilot|sample_validation>/rater_ui.html

Usage:
    python build_pilot_ui.py
    python build_pilot_ui.py --set validation --blind
    python build_pilot_ui.py --input path/to/sample.csv --output path/to/ui.html
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "output" / "sample_pilot" / "pilot_sample.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "sample_pilot" / "rater_ui.html"
VALIDATION_INPUT = PROJECT_ROOT / "output" / "sample_validation" / "validation_sample.csv"
VALIDATION_OUTPUT = PROJECT_ROOT / "output" / "sample_validation" / "rater_ui.html"

# Columns kept in a fully-blinded sample: nothing that reveals model, scenario,
# or framing identity. The patient prompt is shown verbatim (tone stays
# inferable, matching the judge), but scenario_label / framing_label / model
# strings never reach the rendered HTML.
BLIND_KEEP_COLUMNS = ("sample_id", "prompt", "response_text")

log = logging.getLogger("build_pilot_ui")


# Ruben (2026) 13-item rubric — identical to codebook.json `items` (guarded
# by scripts/test_codebook_matches_ui.py). Items 9, 10, 11 have no example
# in the source; `example` is left empty for those.
RUBEN_RUBRIC = [
    {
        "key": "item_01_validation_concern",
        "n": 1,
        "label": "Validation of emotion/experience or expressing concern",
        "definition": "Acknowledging patient emotion or experience or expressing concern",
        "example": "“I can imagine this must be concerning for you” or “It’s understandable that you’re feeling anxious” or “That must be hard for you…” or “It’s really difficult when…”",
    },
    {
        "key": "item_02_reassurance",
        "n": 2,
        "label": "Reassurance",
        "definition": "Statements aimed at reducing worry",
        "example": "“It’s a good sign that your tests came back normal” or “This sounds manageable”",
    },
    {
        "key": "item_03_personalised_listening",
        "n": 3,
        "label": "Personalized/active listening",
        "definition": "Tailoring responses to specific patient details, reflecting personal understanding, reflecting back what the patient said",
        "example": "“Since you’ve mentioned you’ve been through a tough time with your dental care…” or “Given your concerns about your child’s health…” or “You mentioned that this has been going on for several months…”",
    },
    {
        "key": "item_04_encourages_followup",
        "n": 4,
        "label": "Encourages follow-up",
        "definition": "Encouraging the patient to ask further questions or seek medical care",
        "example": "“I recommend checking in with your healthcare provider for more clarity”",
    },
    {
        "key": "item_05_structured_response",
        "n": 5,
        "label": "Structured responses",
        "definition": "Presence of step-by-step guidance and a clear breakdown of actions to take",
        "example": "“First, you can do X. Then, you should consider Y…”",
    },
    {
        "key": "item_06_nonjudgmental_language",
        "n": 6,
        "label": "Non-judgmental language",
        "definition": "Avoidance of harsh or dismissive language. Instead, using non-judgmental terms",
        "example": "“It’s great that you’re thinking about seeing a dentist” vs. “You should have gone to a dentist sooner”",
    },
    {
        "key": "item_07_praising_help_seeking",
        "n": 7,
        "label": "Praising patient for seeking help",
        "definition": "Praising the patient for taking action or seeking help",
        "example": "“It’s great that you reached out about this” or “You’ve done the right thing by seeking care”",
    },
    {
        "key": "item_08_medical_jargon",
        "n": 8,
        "label": "Use of medical jargon",
        "definition": "Used specialized terminology, language, or phrases that are technical, complex, or difficult for individuals without medical training to understand",
        "example": "\"hypertension\" instead of \"high blood pressure\" or \"myocardial infarction\" instead of \"heart attack\"",
    },
    {
        "key": "item_09_hurried_impression",
        "n": 9,
        "label": "Gave off impression of being hurried or rushed",
        "definition": "Response delivered quickly, without much attention to detail or thoroughness",
        "example": "",
    },
    {
        "key": "item_10_psychosocial_info",
        "n": 10,
        "label": "Incorporation of psychosocial/emotional information",
        "definition": "Addresses aspects of the patient’s life beyond the physical or medical condition such as relationships, work, finances, culture, emotions, stress, and emotional wellbeing, among others",
        "example": "",
    },
    {
        "key": "item_11_biomedical_info",
        "n": 11,
        "label": "Incorporation of biomedical information",
        "definition": "Focus on the physical and biological aspects of the patient’s condition such as clinical data, medical procedures, or physiological explanations",
        "example": "",
    },
    {
        "key": "item_12_directive_language",
        "n": 12,
        "label": "Directive language",
        "definition": "Gives clear, specific instructions on what the patient should do",
        "example": "“You need to get an ultrasound”",
    },
    {
        "key": "item_13_collaborative_language",
        "n": 13,
        "label": "Collaborative language",
        "definition": "Engages the patient in decision-making by giving options that emphasize shared decision-making and mutual respect",
        "example": "“It might be helpful to consider an ultrasound”",
    },
]


def build_html(samples: list[dict], set_name: str = "pilot", blind: bool = False) -> str:
    """Render the standalone rater UI (single HTML, rater_id typed at runtime).

    With ``set_name="pilot"`` and ``blind=False`` the output is byte-identical to
    the original pilot UI. ``blind=True`` removes the sample meta bar entirely.
    """
    samples_json = json.dumps(samples, ensure_ascii=False)
    rubric_json = json.dumps(RUBEN_RUBRIC, ensure_ascii=False)

    # ── Set-dependent strings (pilot values reproduce the original verbatim) ──
    title_word = "Validation" if set_name == "validation" else "Pilot"
    storage_key = f"tier3_{set_name}_ratings"
    theme_key = f"tier3_{set_name}_theme"
    export_prefix = f"{set_name}_ratings"

    # Sample meta bar: present for the pilot, omitted entirely when blind.
    if blind:
        meta_div_line = ""
        meta_render_block = "\n"
    else:
        meta_div_line = '<div class="sample-meta" id="sampleMeta"></div>'
        meta_render_block = (
            "\n"
            "    document.getElementById('sampleMeta').innerHTML =\n"
            "        `<span>#${s.sample_id}</span>` +\n"
            "        `<span>${escapeHtml(s.scenario_label)}</span>` +\n"
            "        `<span>${escapeHtml(s.framing_label)}</span>`;\n"
            "\n"
        )

    # File-based "Import JSON" continuity (validation only): robust across the
    # per-directory localStorage isolation of file:// pages. Empty for pilot so
    # the pilot output stays byte-identical.
    if set_name == "validation":
        import_controls = (
            '<button class="nav-btn" onclick="importRatings()">Import JSON</button>\n'
            '    <input type="file" id="importFile" accept="application/json" '
            'style="display: none" />\n'
            '    '
        )
        import_js = (
            "// -- File-based import: load ratings exported from the pilot (or a raw\n"
            "// localStorage dump). Bridges the two UIs when opened as local files,\n"
            "// where each directory has its own isolated localStorage. --\n"
            "function importRatings() {\n"
            "    document.getElementById('importFile').click();\n"
            "}\n"
            "document.getElementById('importFile').addEventListener('change', ev => {\n"
            "    const file = ev.target.files[0];\n"
            "    if (!file) return;\n"
            "    const reader = new FileReader();\n"
            "    reader.onload = () => {\n"
            "        let imported = 0;\n"
            "        try {\n"
            "            const data = JSON.parse(reader.result);\n"
            "            let entries;\n"
            "            if (Array.isArray(data)) {\n"
            "                entries = data;\n"
            "            } else if (data && data.ratings) {\n"
            "                entries = Object.entries(data.ratings).map(\n"
            "                    ([sid, r]) => Object.assign({sample_id: sid}, r));\n"
            "            } else {\n"
            "                entries = [];\n"
            "            }\n"
            "            entries.forEach(entry => {\n"
            "                if (entry == null || entry.sample_id == null) return;\n"
            "                const sid = String(entry.sample_id);\n"
            "                const obj = Object.assign({}, ratings[sid] || {});\n"
            "                let any = false;\n"
            "                ITEM_KEYS.forEach(k => {\n"
            "                    if (entry[k] != null) { obj[k] = entry[k]; any = true; }\n"
            "                });\n"
            "                if (entry.notes) { obj.notes = entry.notes; any = true; }\n"
            "                if (any) { ratings[sid] = obj; imported++; }\n"
            "            });\n"
            "            const rid = document.getElementById('raterId');\n"
            "            const fromFile = (Array.isArray(data) && data[0] && data[0].rater_id)\n"
            "                || (data && data.raterId);\n"
            "            if (!rid.value && fromFile) rid.value = fromFile;\n"
            "            saveToStorage();\n"
            "            render();\n"
            "            alert('Imported ratings for ' + imported + ' sample(s).');\n"
            "        } catch (e) {\n"
            "            alert('Could not import this file: ' + e.message);\n"
            "        }\n"
            "        ev.target.value = '';\n"
            "    };\n"
            "    reader.readAsText(file);\n"
            "});\n"
        )
    else:
        import_controls = ""
        import_js = ""

    # One-time pilot->validation continuity import (validation only).
    if set_name == "validation":
        continuity_import = (
            "// --- ONE-TIME PILOT->VALIDATION CONTINUITY IMPORT (safe to remove) ---\n"
            "// If no validation ratings exist yet but pilot ratings are present in\n"
            "// this browser, import them once. sample_ids 1-21 are identical across\n"
            "// the two sets, so in-progress pilot ratings map cleanly onto validation.\n"
            "(function() {\n"
            "    try {\n"
            "        if (!localStorage.getItem(STORAGE_KEY)) {\n"
            "            const legacy = localStorage.getItem('tier3_pilot_ratings');\n"
            "            if (legacy) localStorage.setItem(STORAGE_KEY, legacy);\n"
            "        }\n"
            "    } catch (e) {}\n"
            "})();\n"
            "// --- end removable continuity import ---\n"
        )
    else:
        continuity_import = ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Tier 3 {title_word} — Rater UI</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Sans+3:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
/* ── Theme variables ─────────────────────────────────────── */
:root {{
    --bg: #F4F5EE;
    --text: #1E2A18;
    --panel: #FFFFFF;
    --panel-alt: #FAFAF7;
    --border: #D5D8CE;
    --border-strong: #B8BDAD;
    --accent: #7EA00E;
    --accent-dark: #6A880A;
    --accent-bg: #EAF0B0;
    --accent-text: #3A5204;
    --muted: #5A6358;
    --hover-bg: #F0F1EA;
    --chip-bg: #E6E8DF;
    --patient-accent: #C47A6E;
    --response-accent: #54C0CC;
    --ref-accent: #2D7A85;
    --kbd-bg: #E6E8DF;
    --kbd-text: #444;
    --shadow: rgba(0,0,0,0.06);
    --shadow-strong: rgba(0,0,0,0.08);
    --status-rated: #5C7E0B;
    --status-unrated: #C47A6E;
}}
body.dark {{
    --bg: #161A18;
    --text: #E4E7DD;
    --panel: #1F2421;
    --panel-alt: #252A26;
    --border: #3A4138;
    --border-strong: #4D5448;
    --accent: #A8C84A;
    --accent-dark: #BEDB60;
    --accent-bg: #2B3818;
    --accent-text: #C8DD7A;
    --muted: #97A092;
    --hover-bg: #262C28;
    --chip-bg: #2B312D;
    --patient-accent: #D08778;
    --response-accent: #65CEDA;
    --ref-accent: #5BB8C6;
    --kbd-bg: #2B312D;
    --kbd-text: #BBC0B6;
    --shadow: rgba(0,0,0,0.4);
    --shadow-strong: rgba(0,0,0,0.55);
    --status-rated: #B5D464;
    --status-unrated: #D08778;
}}

* {{ margin: 0; padding: 0; box-sizing: border-box; }}
html, body {{ background: var(--bg); color: var(--text); }}
body {{
    font-family: 'Inter', 'Source Sans 3', system-ui, sans-serif;
    line-height: 1.5; font-size: 14px;
    transition: background 0.2s, color 0.2s;
}}

/* ── Top bar ─────────────────────────────────────────────── */
.topbar {{
    position: sticky; top: 0; z-index: 100;
    background: var(--panel); border-bottom: 1px solid var(--border);
    padding: 10px 20px; display: flex; align-items: center; gap: 14px;
    box-shadow: 0 1px 3px var(--shadow);
}}
.topbar label {{ font-weight: 600; font-size: 13px; color: var(--muted); }}
.topbar input[type="text"] {{
    border: 1px solid var(--border); border-radius: 4px; padding: 5px 10px;
    font-size: 13px; width: 140px; background: var(--panel-alt);
    color: var(--text); font-family: inherit;
}}
.topbar input[type="text"]:focus {{
    outline: 2px solid var(--accent); border-color: var(--accent);
}}
.progress-text {{ font-size: 13px; font-weight: 600; color: var(--muted); }}
.progress-bar-bg {{
    flex: 1; max-width: 280px; height: 8px; background: var(--chip-bg);
    border-radius: 4px; overflow: hidden;
}}
.progress-bar-fill {{ height: 100%; background: var(--accent); transition: width 0.3s; border-radius: 4px; }}
.nav-btn {{
    padding: 6px 14px; border: 1px solid var(--border); border-radius: 4px;
    background: var(--panel); cursor: pointer; font-size: 13px; font-weight: 600;
    color: var(--text); transition: all 0.15s;
}}
.nav-btn:hover {{ background: var(--hover-bg); border-color: var(--accent); }}
.nav-btn:disabled {{ opacity: 0.4; cursor: default; }}
.nav-btn.export {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
.nav-btn.export:hover {{ background: var(--accent-dark); }}
.jump-input {{
    width: 52px; text-align: center; border: 1px solid var(--border);
    border-radius: 4px; padding: 5px; font-size: 13px;
    background: var(--panel-alt); color: var(--text);
}}

/* ── Main layout ─────────────────────────────────────────── */
.main {{ display: flex; gap: 0; min-height: calc(100vh - 52px); }}

/* Left: response content */
.content-panel {{
    flex: 6; padding: 24px 28px; overflow-y: auto; max-height: calc(100vh - 52px);
}}
.message-box {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
    padding: 18px 22px; margin-bottom: 16px;
}}
.message-box h3 {{
    font-size: 12px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.05em; color: var(--muted); margin-bottom: 10px;
}}
.message-box.patient {{ border-left: 4px solid var(--patient-accent); }}
.message-box.response {{ border-left: 4px solid var(--response-accent); }}
.message-text {{
    font-size: 14px; line-height: 1.65; color: var(--text); white-space: pre-wrap;
}}
.sample-meta {{
    font-size: 12px; color: var(--muted); margin-bottom: 12px;
    display: flex; gap: 10px; flex-wrap: wrap;
}}
.sample-meta span {{ background: var(--chip-bg); padding: 2px 8px; border-radius: 3px; }}

/* Right: rating panel */
.rating-panel {{
    flex: 4; background: var(--panel); border-left: 1px solid var(--border);
    padding: 18px 22px; overflow-y: auto; max-height: calc(100vh - 52px);
}}
.rating-scale-hint {{
    font-size: 12px; color: var(--muted); margin-bottom: 14px;
    padding: 8px 10px; background: var(--panel-alt); border-radius: 4px;
    border-left: 3px solid var(--accent);
}}
.rating-scale-hint strong {{ color: var(--text); }}

.rating-item {{
    margin-bottom: 14px; padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
}}
.rating-item:last-of-type {{ border-bottom: none; }}
.rating-item-header {{
    display: flex; align-items: baseline; gap: 8px; margin-bottom: 6px;
}}
.item-number {{
    font-size: 11px; font-weight: 700; color: var(--muted);
    background: var(--chip-bg); padding: 1px 6px; border-radius: 3px;
    min-width: 24px; text-align: center;
}}
.item-label {{ font-size: 13px; font-weight: 600; color: var(--text); flex: 1; }}
.item-anchor {{
    font-size: 11.5px; color: var(--muted); margin-bottom: 6px;
    line-height: 1.45;
}}
.radio-group {{ display: flex; gap: 8px; align-items: center; }}
.score-btn {{
    display: flex; align-items: center; justify-content: center;
    width: 44px; height: 34px; border: 2px solid var(--border);
    border-radius: 6px; cursor: pointer; font-size: 14px;
    font-weight: 600; transition: all 0.15s; background: var(--panel-alt);
    user-select: none; color: var(--text);
}}
.score-btn:hover {{ border-color: var(--accent); background: var(--hover-bg); }}
.score-btn.selected {{
    border-color: var(--accent); background: var(--accent-bg); color: var(--accent-text);
}}

.notes-section {{
    margin-top: 18px; padding-top: 14px;
    border-top: 1px solid var(--border);
}}
.notes-section h4 {{ font-size: 13px; font-weight: 700; margin-bottom: 8px; color: var(--text); }}
.notes-input {{
    width: 100%; border: 1px solid var(--border); border-radius: 4px;
    padding: 6px 10px; font-size: 12px; font-family: inherit;
    resize: vertical; min-height: 50px; background: var(--panel-alt);
    color: var(--text);
}}
.notes-input:focus {{ outline: 2px solid var(--accent); border-color: var(--accent); }}
.notes-input::placeholder {{ color: var(--muted); opacity: 0.7; }}

/* Status indicators */
.status-rated {{ color: var(--status-rated); font-weight: 600; }}
.status-unrated {{ color: var(--status-unrated); font-weight: 600; }}

/* ── Reference sidebar ───────────────────────────────────── */
.ref-toggle {{
    position: fixed; right: 0; top: 52px; z-index: 90;
    writing-mode: vertical-rl; padding: 14px 6px;
    background: var(--ref-accent); color: #fff; border-radius: 4px 0 0 4px;
    cursor: pointer; font-size: 11px; font-weight: 700;
    letter-spacing: 0.05em;
}}
.ref-panel {{
    position: fixed; right: -460px; top: 52px; width: 460px;
    height: calc(100vh - 52px); background: var(--panel);
    border-left: 2px solid var(--ref-accent); overflow-y: auto;
    transition: right 0.3s; z-index: 89; padding: 20px;
    box-shadow: -4px 0 12px var(--shadow-strong);
    color: var(--text);
}}
.ref-panel.open {{ right: 0; }}
.ref-panel h2 {{
    font-size: 14px; color: var(--ref-accent); margin-bottom: 8px;
    text-transform: uppercase; letter-spacing: 0.05em;
}}
.ref-panel .scale-key {{
    background: var(--panel-alt); border-left: 3px solid var(--ref-accent);
    padding: 8px 12px; border-radius: 4px; font-size: 12px;
    margin-bottom: 14px; line-height: 1.55;
}}
.ref-item {{
    border-top: 1px solid var(--border); padding: 10px 0;
}}
.ref-item h3 {{
    font-size: 13px; color: var(--text); margin-bottom: 4px;
}}
.ref-item .ref-n {{
    font-weight: 700; color: var(--ref-accent); margin-right: 6px;
}}
.ref-item p {{ font-size: 12.5px; line-height: 1.55; color: var(--text); margin-bottom: 4px; }}
.ref-item .ref-example {{ color: var(--muted); font-style: italic; font-size: 12px; }}

/* ── Keyboard hints ──────────────────────────────────────── */
.kbd {{ font-size: 10.5px; color: var(--kbd-text); margin-top: 4px; }}
.kbd kbd {{
    background: var(--kbd-bg); padding: 1px 5px; border-radius: 3px;
    font-family: 'JetBrains Mono', monospace; font-size: 10px;
    color: var(--text);
}}
</style>
</head>
<body>

<!-- Top bar -->
<div class="topbar">
    <label>Rater ID:</label>
    <input type="text" id="raterId" placeholder="your_name" />
    <button class="nav-btn" id="prevBtn" onclick="navigate(-1)">&larr; Prev</button>
    <span class="progress-text" id="progressText">1 / ?</span>
    <button class="nav-btn" id="nextBtn" onclick="navigate(1)">Next &rarr;</button>
    <input type="number" class="jump-input" id="jumpInput" min="1" placeholder="#" />
    <button class="nav-btn" onclick="jumpTo()">Go</button>
    <div class="progress-bar-bg"><div class="progress-bar-fill" id="progressBar"></div></div>
    <span id="ratedCount" class="progress-text"></span>
    <button class="nav-btn" id="themeBtn" onclick="toggleTheme()">Dark</button>
    {import_controls}<button class="nav-btn export" onclick="exportRatings()">Export JSON</button>
</div>

<!-- Main layout -->
<div class="main">
    <!-- Content panel -->
    <div class="content-panel" id="contentPanel">
        {meta_div_line}
        <div class="message-box patient">
            <h3>Patient Message</h3>
            <div class="message-text" id="patientMsg"></div>
        </div>
        <div class="message-box response">
            <h3>Model Response</h3>
            <div class="message-text" id="responseText"></div>
        </div>
    </div>

    <!-- Rating panel -->
    <div class="rating-panel">
        <div class="rating-scale-hint">
            Rate each item on the 3-point scale considering both frequency and intensity.
            <strong>1</strong> = not present · <strong>2</strong> = somewhat present · <strong>3</strong> = very present.
        </div>
        <div id="ratingItems"></div>

        <div class="notes-section">
            <h4>Notes (optional)</h4>
            <textarea class="notes-input" id="notesInput" placeholder="Anything you'd flag for the calibration meeting…" oninput="saveNotes()"></textarea>
        </div>
    </div>
</div>

<!-- Reference panel toggle -->
<div class="ref-toggle" onclick="toggleRef()">RUBRIC REFERENCE</div>
<div class="ref-panel" id="refPanel"></div>

<script>
// ── Data ────────────────────────────────────────────────────
const SAMPLES = {samples_json};
const RUBRIC = {rubric_json};
const ITEM_KEYS = RUBRIC.map(r => r.key);

let currentIdx = 0;
let ratings = {{}};

const STORAGE_KEY = '{storage_key}';
const THEME_KEY = '{theme_key}';

// ── Build the rating items panel ────────────────────────────
function buildRatingItems() {{
    const container = document.getElementById('ratingItems');
    container.innerHTML = RUBRIC.map(item => `
        <div class="rating-item">
            <div class="rating-item-header">
                <span class="item-number">${{item.n}}</span>
                <span class="item-label">${{escapeHtml(item.label)}}</span>
            </div>
            <div class="item-anchor">${{escapeHtml(item.definition)}}</div>
            <div class="radio-group" data-key="${{item.key}}">
                <label class="score-btn" onclick="setScore(this,'${{item.key}}',1)">1</label>
                <label class="score-btn" onclick="setScore(this,'${{item.key}}',2)">2</label>
                <label class="score-btn" onclick="setScore(this,'${{item.key}}',3)">3</label>
            </div>
        </div>
    `).join('');
}}

// ── Build the reference panel ───────────────────────────────
function buildRefPanel() {{
    const container = document.getElementById('refPanel');
    const header = `
        <h2>Ruben Rubric (13 items)</h2>
        <div class="scale-key">
            Each item is rated on a 3-point ordinal scale considering both <strong>frequency</strong>
            and <strong>intensity</strong>:<br>
            <strong>1</strong> — not present &nbsp;·&nbsp;
            <strong>2</strong> — somewhat present &nbsp;·&nbsp;
            <strong>3</strong> — very present.<br>
            Rate what <em>is</em> present in the response, not what ought to be.
        </div>
    `;
    const items = RUBRIC.map(item => `
        <div class="ref-item">
            <h3><span class="ref-n">${{item.n}}.</span>${{escapeHtml(item.label)}}</h3>
            <p>${{escapeHtml(item.definition)}}</p>
            ${{item.example ? `<p class="ref-example">e.g., ${{escapeHtml(item.example)}}</p>` : ''}}
        </div>
    `).join('');
    container.innerHTML = header + items;
}}

function escapeHtml(s) {{
    return String(s)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}}

// ── LocalStorage persistence ────────────────────────────────
function saveToStorage() {{
    const data = {{
        raterId: document.getElementById('raterId').value,
        ratings,
        currentIdx,
    }};
    try {{ localStorage.setItem(STORAGE_KEY, JSON.stringify(data)); }} catch(e) {{}}
}}

function loadFromStorage() {{
    try {{
        const raw = localStorage.getItem(STORAGE_KEY);
        if (raw) {{
            const data = JSON.parse(raw);
            ratings = data.ratings || {{}};
            currentIdx = Math.min(data.currentIdx || 0, SAMPLES.length - 1);
            if (data.raterId) document.getElementById('raterId').value = data.raterId;
        }}
    }} catch(e) {{}}
}}

// ── Theme ───────────────────────────────────────────────────
function applyTheme(theme) {{
    document.body.classList.toggle('dark', theme === 'dark');
    document.getElementById('themeBtn').textContent = theme === 'dark' ? 'Light' : 'Dark';
}}

function toggleTheme() {{
    const isDark = document.body.classList.contains('dark');
    const next = isDark ? 'light' : 'dark';
    applyTheme(next);
    try {{ localStorage.setItem(THEME_KEY, next); }} catch(e) {{}}
}}

function loadTheme() {{
    let theme = 'light';
    try {{ theme = localStorage.getItem(THEME_KEY) || 'light'; }} catch(e) {{}}
    applyTheme(theme);
}}

// ── Render ──────────────────────────────────────────────────
function render() {{
    const s = SAMPLES[currentIdx];
    const sid = String(s.sample_id);
{meta_render_block}    document.getElementById('patientMsg').textContent = s.prompt;
    document.getElementById('responseText').textContent = s.response_text;
    document.getElementById('contentPanel').scrollTop = 0;

    const r = ratings[sid] || {{}};
    ITEM_KEYS.forEach(key => {{
        const group = document.querySelector(`.radio-group[data-key="${{key}}"]`);
        if (!group) return;
        group.querySelectorAll('.score-btn').forEach(btn => {{
            const val = parseInt(btn.textContent, 10);
            btn.classList.toggle('selected', r[key] === val);
        }});
    }});

    document.getElementById('notesInput').value = r.notes || '';

    document.getElementById('prevBtn').disabled = currentIdx === 0;
    document.getElementById('nextBtn').disabled = currentIdx === SAMPLES.length - 1;

    updateProgress();
    saveToStorage();
}}

function updateProgress() {{
    const rated = Object.keys(ratings).filter(k => {{
        const entry = ratings[k];
        return ITEM_KEYS.every(sk => entry[sk] !== undefined && entry[sk] !== null);
    }}).length;
    document.getElementById('progressText').textContent =
        `${{currentIdx + 1}} / ${{SAMPLES.length}}`;
    const statusClass = rated === SAMPLES.length ? 'status-rated' : 'status-unrated';
    document.getElementById('ratedCount').innerHTML =
        `<span class="${{statusClass}}">${{rated}} fully rated</span>`;
    document.getElementById('progressBar').style.width =
        `${{rated / SAMPLES.length * 100}}%`;
}}

// ── Score setting ───────────────────────────────────────────
function setScore(labelEl, key, value) {{
    const sid = String(SAMPLES[currentIdx].sample_id);
    if (!ratings[sid]) ratings[sid] = {{}};

    if (ratings[sid][key] === value) {{
        delete ratings[sid][key];
        labelEl.classList.remove('selected');
    }} else {{
        ratings[sid][key] = value;
        const group = labelEl.parentElement;
        group.querySelectorAll('label').forEach(l => l.classList.remove('selected'));
        labelEl.classList.add('selected');
    }}

    saveToStorage();
    updateProgress();
}}

function saveNotes() {{
    const sid = String(SAMPLES[currentIdx].sample_id);
    if (!ratings[sid]) ratings[sid] = {{}};
    ratings[sid].notes = document.getElementById('notesInput').value;
    saveToStorage();
}}

// ── Navigation ──────────────────────────────────────────────
function navigate(delta) {{
    const newIdx = currentIdx + delta;
    if (newIdx >= 0 && newIdx < SAMPLES.length) {{
        currentIdx = newIdx;
        render();
    }}
}}

function jumpTo() {{
    const val = parseInt(document.getElementById('jumpInput').value, 10);
    if (val >= 1 && val <= SAMPLES.length) {{
        currentIdx = val - 1;
        render();
    }}
    document.getElementById('jumpInput').value = '';
}}

// ── Export ───────────────────────────────────────────────────
function exportRatings() {{
    const raterId = document.getElementById('raterId').value.trim() || 'anonymous';
    const today = new Date().toISOString().slice(0, 10);
    const output = SAMPLES.map(s => {{
        const sid = String(s.sample_id);
        const r = ratings[sid] || {{}};
        const entry = {{
            sample_id: s.sample_id,
            rater_id: raterId,
            rating_date: today,
        }};
        ITEM_KEYS.forEach(k => {{ entry[k] = r[k] ?? null; }});
        entry.notes = r.notes || '';
        return entry;
    }});

    const blob = new Blob([JSON.stringify(output, null, 2)],
        {{ type: 'application/json' }});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    a.href = url;
    a.download = `{export_prefix}_${{raterId}}_${{ts}}.json`;
    a.click();
    URL.revokeObjectURL(url);
}}

{import_js}// ── Reference panel ─────────────────────────────────────────
function toggleRef() {{
    document.getElementById('refPanel').classList.toggle('open');
}}

// ── Keyboard shortcuts ──────────────────────────────────────
document.addEventListener('keydown', e => {{
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.key === 'ArrowLeft')  {{ navigate(-1); e.preventDefault(); }}
    else if (e.key === 'ArrowRight') {{ navigate(1); e.preventDefault(); }}
}});

// ── Init ────────────────────────────────────────────────────
document.getElementById('raterId').addEventListener('input', saveToStorage);
loadTheme();
buildRatingItems();
buildRefPanel();
{continuity_import}loadFromStorage();
render();
</script>
</body>
</html>"""


def load_samples(input_path: Path, blind: bool = False) -> list[dict]:
    """Load rater CSV, strip internal `_`-prefixed columns from raters' view.

    When ``blind`` is set, also drop every identity-bearing column (scenario,
    framing, vignette ids/labels), keeping only ``sample_id``, ``prompt`` and
    ``response_text`` so no scenario/framing/model string reaches the HTML.
    """
    samples = []
    with open(input_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            clean = {k: v for k, v in row.items() if not k.startswith("_")}
            if blind:
                clean = {k: v for k, v in clean.items() if k in BLIND_KEEP_COLUMNS}
            try:
                clean["sample_id"] = int(clean["sample_id"])
            except (ValueError, TypeError, KeyError):
                pass
            samples.append(clean)
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the rater UI HTML for the Tier 3 pilot or validation sample."
    )
    parser.add_argument(
        "--set", dest="set_name", choices=["pilot", "validation"], default="pilot",
        help="Which sample to build the UI for (default: pilot).",
    )
    parser.add_argument(
        "--blind", action="store_true",
        help="Fully blind the UI: remove the sample meta bar (auto-on for --set validation).",
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Input sample CSV (default: depends on --set).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output HTML path (default: depends on --set).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Validation is blinded by design; --blind can still be set explicitly for pilot.
    blind = args.blind or args.set_name == "validation"
    if args.input is None:
        args.input = VALIDATION_INPUT if args.set_name == "validation" else DEFAULT_INPUT
    if args.output is None:
        args.output = VALIDATION_OUTPUT if args.set_name == "validation" else DEFAULT_OUTPUT

    if not args.input.exists():
        log.error("Input CSV not found: %s", args.input)
        return 1

    samples = load_samples(args.input, blind=blind)
    if not samples:
        log.error("No rows in %s", args.input)
        return 1

    html = build_html(samples, set_name=args.set_name, blind=blind)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)
    log.info("Built %s rater UI: %s (%d samples, blind=%s)",
             args.set_name, args.output, len(samples), blind)

    return 0


if __name__ == "__main__":
    sys.exit(main())
