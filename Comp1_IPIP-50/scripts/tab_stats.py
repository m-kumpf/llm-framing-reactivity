#!/usr/bin/env python3
"""
Generate an APA-style RTF table of IPIP-50 Big Five descriptive and
inferential statistics across models.

Reads:
  output/analyse-results/summary_statistics.csv
  output/analyse-results/statistical_tests.csv

Writes:
  output/tab-stats/tab_stats.rtf
"""

import csv
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# ── Model mapping (API ID → short display name) ─────────────────────────────
API_TO_SHORT = {
    'anthropic/claude-sonnet-4.6':   'Claude 4.6',
    'openai/gpt-5.3-chat':          'GPT-5.3',
    'google/gemini-3.1-pro-preview': 'Gemini 3.1',
    'moonshotai/kimi-k2.5':         'Kimi K2.5',
    'qwen/qwen3.5-397b-a17b':       'Qwen 3.5',
    'minimax/minimax-m2.7':         'MiniMax M2.7',
    'z-ai/glm-5':                   'GLM 5',
}

MODEL_ORDER = list(API_TO_SHORT.keys())

FACTORS = [
    'Extraversion', 'Agreeableness', 'Conscientiousness',
    'Emotional_Stability', 'Intellect_Openness',
]
FACTOR_HEADERS = [
    'Extraversion', 'Agreeableness', 'Conscientiousness',
    'Emotional Stability', 'Intellect / Openness',
]

# ── Load data ────────────────────────────────────────────────────────────────
summary_path = BASE / 'output' / 'analyse-results' / 'summary_statistics.csv'
tests_path = BASE / 'output' / 'analyse-results' / 'statistical_tests.csv'

summary_rows = list(csv.DictReader(open(summary_path)))
test_rows = list(csv.DictReader(open(tests_path)))

# Index summary: (api_id, factor) → {mean, sd}
stats = {}
for r in summary_rows:
    stats[(r['model'], r['factor'])] = (float(r['mean']), float(r['sd']))

# Extract KW results per factor
kw = {}
for r in test_rows:
    if r['test'] == 'Kruskal-Wallis':
        kw[r['factor']] = {
            'H': float(r['statistic']),
            'p': float(r['p_corrected']),
        }

# Count significant pairwise comparisons per factor
n_pairs = len(MODEL_ORDER) * (len(MODEL_ORDER) - 1) // 2
sig_counts = {f: 0 for f in FACTORS}
for r in test_rows:
    if r['test'] == 'Mann-Whitney U' and r['significant_005'] == 'True':
        sig_counts[r['factor']] += 1


# ── RTF helpers ──────────────────────────────────────────────────────────────
def fmt_p(p):
    """Format p-value APA-style (no leading zero)."""
    if p < 0.001:
        return '< .001'
    return f'{p:.3f}'.lstrip('0')  # e.g. 0.032 → .032


def rtf_escape(text):
    """Escape special RTF characters."""
    return text.replace('\\', '\\\\').replace('{', '\\{').replace('}', '\\}')


# ── Build RTF ────────────────────────────────────────────────────────────────
# Column widths in twips (1 inch = 1440 twips)
COL_MODEL = 2200   # ~1.5 in for model name
COL_FACTOR = 1700  # ~1.2 in per factor column

# Cell border definitions (APA: horizontal rules only)
BORDER_TOP = '\\clbrdrt\\brdrs\\brdrw10'
BORDER_BOTTOM = '\\clbrdrb\\brdrs\\brdrw10'


def cell_defs(borders=''):
    """Generate \\cellx definitions for all columns."""
    defs = []
    x = COL_MODEL
    defs.append(f'{borders}\\cellx{x}')
    for _ in FACTORS:
        x += COL_FACTOR
        defs.append(f'{borders}\\cellx{x}')
    return '\n'.join(defs)


def row_left_first(cells, bold_first=False, italic_first=False, borders='',
                   spacing_after=0):
    """Build a row where the first cell is left-aligned, rest centred."""
    sa = f'\\sa{spacing_after}' if spacing_after else ''

    parts = [f'\\trowd\\trqc\n{cell_defs(borders)}']
    for i, c in enumerate(cells):
        text = rtf_escape(str(c))
        align = '\\ql' if i == 0 else '\\qc'
        bf = '\\b' if (i == 0 and bold_first) else ''
        bf_end = '\\b0' if (i == 0 and bold_first) else ''
        it = '\\i' if (i == 0 and italic_first) else ''
        it_end = '\\i0' if (i == 0 and italic_first) else ''
        parts.append(
            f'\\pard\\intbl{align}{sa} {bf}{it} {text}{it_end}{bf_end}\\cell'
        )
    parts.append('\\row')
    return '\n'.join(parts)


# Assemble RTF document
lines = []
lines.append('{\\rtf1\\ansi\\deff0')
lines.append('{\\fonttbl{\\f0 Times New Roman;}}')
lines.append('\\f0\\fs20')  # 10pt

# Title line
lines.append('\\pard\\qc\\b Table 1\\b0\\par')
lines.append('\\pard\\qc\\i Mean (SD) IPIP-50 Factor Scores by Model, '
             'Omnibus Kruskal\\endash Wallis Tests, '
             'and Significant Pairwise Comparisons\\i0\\par')
lines.append('\\par')

# Header row (top + bottom border)
header_borders = f'{BORDER_TOP}{BORDER_BOTTOM}'
header = ['Model'] + FACTOR_HEADERS
lines.append(row_left_first(header, bold_first=True, borders=header_borders))

# Model rows — mean line then (SD) line
for idx, api_id in enumerate(MODEL_ORDER):
    short = API_TO_SHORT[api_id]
    mean_cells = [short]
    sd_cells = ['']

    for f in FACTORS:
        m, s = stats[(api_id, f)]
        mean_cells.append(f'{m:.1f}')
        sd_cells.append(f'({s:.1f})')

    # No borders on model rows, except bottom border on last model's SD row
    is_last = (idx == len(MODEL_ORDER) - 1)
    lines.append(row_left_first(mean_cells, bold_first=False))
    sd_border = BORDER_BOTTOM if is_last else ''
    lines.append(row_left_first(sd_cells, borders=sd_border,
                                spacing_after=40 if not is_last else 0))

# Footer: H row (italic label, manual RTF for first cell)
h_cells = ['H']
for f in FACTORS:
    h_cells.append(f'{kw[f]["H"]:.1f}')
lines.append(row_left_first(h_cells, italic_first=True))

# Footer: p row (italic label)
p_cells = ['p']
for f in FACTORS:
    p_cells.append(fmt_p(kw[f]['p']))
lines.append(row_left_first(p_cells, italic_first=True))

# Footer: sig pairs row (with bottom border)
sig_cells = ['Sig. pairs']
for f in FACTORS:
    sig_cells.append(f'{sig_counts[f]}/{n_pairs}')
lines.append(row_left_first(sig_cells, borders=BORDER_BOTTOM))

# Table note
lines.append('\\pard\\sa0\\par')
n_models = len(MODEL_ORDER)
lines.append(
    '\\pard\\fs18 {\\i Note.} '
    '{\\i N} = 10 independent repetitions per model (temperature = 1.0). '
    'Scores range from 10 to 50. '
    'Kruskal\\endash Wallis {\\i H} tests omnibus differences across '
    f'all {n_models} models per factor. '
    'Sig. pairs = number of pairwise Mann\\endash Whitney {\\i U} comparisons '
    f'reaching significance at Holm-corrected \\u945? = .05 '
    f'(out of {n_pairs} possible pairs).\\par'
)

lines.append('}')

# ── Write output ─────────────────────────────────────────────────────────────
out_dir = BASE / 'output' / 'tab-stats'
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / 'tab_stats.rtf'
out_path.write_text('\n'.join(lines), encoding='utf-8')
print(f'Saved {out_path}')
