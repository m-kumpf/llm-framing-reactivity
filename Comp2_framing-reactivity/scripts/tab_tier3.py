#!/usr/bin/env python3
"""
Generate the three APA-style RTF main-text tables for the manuscript
(Comp2 / Tier-3), following the RTF conventions of Comp1's tab_stats.py.

Reads:
  comp2 model roster (static, below)
  output/analyse_tier3/permutation_tests.csv
  output/analyse_tier3_judge/agreement_per_item.csv
  output/analyse_tier3_judge/gate_verdicts.json

Writes (output/tab_tier3/):
  tab1_models.rtf            — the 7 models
  tab2_tier3_inference.rtf   — pooled permutation ANOVA, 9 endpoints x 3 effects
  tab3_validation_gate.rtf   — condensed judge-validation gate + composite ICC
"""

import csv
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / 'output' / 'tab_tier3'

MODELS = [
    ('Anthropic',   'Claude Sonnet 4.6', 'anthropic/claude-sonnet-4.6'),
    ('OpenAI',      'GPT-5.3',           'openai/gpt-5.3-chat'),
    ('Google',      'Gemini 3.1 Pro',    'google/gemini-3.1-pro-preview'),
    ('Moonshot AI', 'Kimi K2.5',         'moonshotai/kimi-k2.5'),
    ('Alibaba',     'Qwen 3.5',          'qwen/qwen3.5-397b-a17b'),
    ('MiniMax',     'MiniMax M2.7',      'minimax/minimax-m2.7'),
    ('Z-AI',        'GLM 5',             'z-ai/glm-5'),
]

ITEM_LABELS = {
    'item_01_validation_concern':    'Validation of emotion / concern',
    'item_02_reassurance':           'Reassurance',
    'item_03_personalised_listening': 'Personalized / active listening',
    'item_04_encourages_followup':   'Encourages follow-up',
    'item_05_structured_response':   'Structured responses',
    'item_06_nonjudgmental_language': 'Non-judgmental language',
    'item_07_praising_help_seeking': 'Praising help-seeking',
    'item_08_medical_jargon':        'Use of medical jargon',
    'item_09_hurried_impression':    'Hurried impression',
    'item_10_psychosocial_info':     'Psychosocial information',
    'item_11_biomedical_info':       'Biomedical information',
    'item_12_directive_language':    'Directive language',
    'item_13_collaborative_language': 'Collaborative language',
}
ENDPOINT_LABELS = dict(ITEM_LABELS)
ENDPOINT_LABELS['composite_relationship'] = 'Relationship-oriented composite'
ENDPOINT_LABELS['composite_relationship_5item'] = \
    'Relationship composite, 5-item'
ENDPOINT_LABELS['composite_conscientious'] = 'Conscientious composite'
ENDPOINT_LABELS['composite_guiding'] = 'Guiding composite'
ENDPOINT_LABELS['composite_technical'] = 'Technical composite'

ROLE_SUFFIX = {
    'primary': ' (primary)',
    'secondary': '',
    'sensitivity': ' (sensitivity)',
    'secondary_ruben': ' (Ruben)',
    'exploratory_ruben': ' (Ruben, exploratory)',
    'exploratory_fallback': ' (exploratory)',
}

STATUS_DISPLAY = {
    'confirmatory': 'Confirmatory',
    'descriptive_saturated': 'Saturated (descriptive)',
    'exploratory': 'Exploratory',
}


# ── RTF helpers (Comp1 tab_stats.py conventions) ─────────────────────────────

def fmt_p(p):
    p = float(p)
    if p < 0.001:
        return '< .001'
    return f'{p:.3f}'.lstrip('0')


def fmt_r(v, nd=2):
    """APA correlation-like value, no leading zero."""
    s = f'{float(v):.{nd}f}'
    return s.replace('0.', '.', 1) if abs(float(v)) < 1 else s


def rtf_escape(text):
    s = (str(text).replace('\\', '\\\\').replace('{', '\\{')
         .replace('}', '\\}'))
    # RTF is ANSI: encode non-ASCII as \uN? so Word renders κ, η, ×, – etc.
    return ''.join(ch if ord(ch) < 128 else f'\\u{ord(ch)}?' for ch in s)


BORDER_TOP = '\\clbrdrt\\brdrs\\brdrw10'
BORDER_BOTTOM = '\\clbrdrb\\brdrs\\brdrw10'


class RtfTable:
    """Minimal APA RTF table builder with per-table column widths (twips)."""

    def __init__(self, number, caption, col_widths):
        self.col_widths = col_widths
        self.lines = [
            '{\\rtf1\\ansi\\deff0',
            '{\\fonttbl{\\f0 Times New Roman;}}',
            '\\f0\\fs20',
            f'\\pard\\ql\\b Table {number}\\b0\\par',
            f'\\pard\\ql\\i {rtf_escape(caption)}\\i0\\par',
            '\\par',
        ]

    def _cell_defs(self, borders=''):
        defs, x = [], 0
        for w in self.col_widths:
            x += w
            defs.append(f'{borders}\\cellx{x}')
        return '\n'.join(defs)

    def row(self, cells, borders='', bold=False, italic_first=False,
            left_cols=1):
        parts = [f'\\trowd\\trqc\n{self._cell_defs(borders)}']
        for i, c in enumerate(cells):
            align = '\\ql' if i < left_cols else '\\qc'
            b, b0 = ('\\b', '\\b0') if bold else ('', '')
            it, it0 = ('\\i', '\\i0') if (italic_first and i == 0) else ('', '')
            parts.append(
                f'\\pard\\intbl{align} {b}{it} {rtf_escape(c)}{it0}{b0}\\cell')
        parts.append('\\row')
        self.lines.append('\n'.join(parts))

    def note(self, text_rtf):
        self.lines.append('\\pard\\sa0\\par')
        self.lines.append(f'\\pard\\fs18 {{\\i Note.}} {text_rtf}\\par')

    def save(self, path):
        self.lines.append('}')
        path.write_text('\n'.join(self.lines), encoding='utf-8')
        print(f'Saved {path}')


# ── Table 1: models ──────────────────────────────────────────────────────────

def build_table1():
    t = RtfTable(1, 'Large Language Models Evaluated', [2200, 2600, 4200])
    t.row(['Provider', 'Model', 'API identifier (OpenRouter)'],
          borders=f'{BORDER_TOP}{BORDER_BOTTOM}', bold=True, left_cols=3)
    for i, (prov, name, api) in enumerate(MODELS):
        b = BORDER_BOTTOM if i == len(MODELS) - 1 else ''
        t.row([prov, name, api], borders=b, left_cols=3)
    t.note('All models accessed via the OpenRouter API '
           '(temperature = 1.0, top-{\\i p} = 1.0, no system prompt); '
           '10 repetitions per scenario \\u215? framing cell.')
    t.save(OUT_DIR / 'tab1_models.rtf')


# ── Tables 2 / S1: pooled Tier-3 inference (two Holm families) ──────────────

def _emit_inference_table(number, caption, rows, note, filename):
    t = RtfTable(number, caption, [3300, 1750, 1500, 1100, 1250, 1250])
    t.row(['Endpoint', 'Effect', 'F', 'η²p', 'p (perm.)',
           'p (Holm)'],
          borders=f'{BORDER_TOP}{BORDER_BOTTOM}', bold=True, left_cols=2)

    effect_display = {'model': 'Model', 'framing': 'Framing',
                      'interaction': 'Model × Framing'}
    endpoints = []
    for r in rows:
        if r['endpoint'] not in endpoints:
            endpoints.append(r['endpoint'])

    for ei, ep in enumerate(endpoints):
        ep_rows = [r for r in rows if r['endpoint'] == ep]
        label = ENDPOINT_LABELS.get(ep, ep)
        label += ROLE_SUFFIX.get(ep_rows[0]['role'], '')
        for j, r in enumerate(ep_rows):
            first = label if j == 0 else ''
            p_holm = fmt_p(r['p_holm'])
            if float(r['p_holm']) < 0.05:
                p_holm += '*'
            is_last = (ei == len(endpoints) - 1 and j == len(ep_rows) - 1)
            t.row([first, effect_display[r['effect']], f"{float(r['F']):.1f}",
                   fmt_r(r['eta2_partial']), fmt_p(r['p_perm']), p_holm],
                  borders=BORDER_BOTTOM if is_last else '', left_cols=2)
    t.note(note)
    t.save(OUT_DIR / filename)


_NOTE_DESIGN = (
    'Balanced three-factor decomposition with scenario as crossed blocking '
    'factor; error term pools all scenario-involving interactions. Degrees '
    'of freedom: Model (6, 369), Framing (5, 369), Model \\u215? Framing '
    '(30, 369). {\\i p} (perm.) from 10,000 restricted permutations (main '
    'effects: labels permuted within strata; interaction: Freedman\\endash '
    'Lane residual permutation within scenarios). ')


def build_table2():
    rows = list(csv.DictReader(open(
        BASE / 'output' / 'analyse_tier3' / 'permutation_tests.csv')))
    conf = [r for r in rows if r.get('holm_family') == 'confirmatory']
    supp = [r for r in rows if r.get('holm_family') == 'sensitivity_ruben']

    _emit_inference_table(
        2,
        'Pooled Permutation ANOVA on the Confirmatory Judge-Scored Tier-3 '
        'Endpoints (420 Model × Framing × Scenario Cell Means)',
        conf,
        _NOTE_DESIGN
        + f'{{\\i p}} (Holm) corrected across the {len(conf)} confirmatory '
          'endpoint \\u215? effect tests (one family: the primary composite '
          'and the eight agreement-gate-confirmed items). *{\\i p} < .05 '
          'after correction. The primary relationship-oriented composite is '
          'the component validated by Ruben, Blanch\\endash Hartigan & Hall '
          '(2026; mean of items 1, 2, 3, 6, 7, 10 \\u8212? item 6 retained '
          'per the validated construct). Sensitivity and secondary Ruben '
          'composites are corrected in a separate family (Table S1).',
        'tab2_tier3_inference.rtf')

    _emit_inference_table(
        'S1',
        'Pooled Permutation ANOVA on the Supplementary Tier-3 Composites: '
        '5-Item Sensitivity Variant and Ruben Secondary Composites',
        supp,
        _NOTE_DESIGN
        + f'{{\\i p}} (Holm) corrected across the {len(supp)} supplementary '
          'endpoint \\u215? effect tests as a SEPARATE family from the '
          'confirmatory endpoints of Table 2, so these supplementary tests '
          'do not affect the confirmatory corrections. *{\\i p} < .05 after '
          'correction. Relationship composite, 5-item = items 1, 2, 3, 7, '
          '10 (item 6 removed; application-specific sensitivity check). '
          'Ruben\\rquote s remaining chatbot composites: conscientious '
          '(item 9 reverse-scored, item 13), guiding (items 5, 12), '
          'technical (items 8, 11) \\u8212? labelled exploratory where they '
          'contain non-gate-confirmed items.',
        'tab_s1_tier3_sensitivity.rtf')


# ── Table 3: validation gate ─────────────────────────────────────────────────

def build_table3():
    rows = list(csv.DictReader(open(
        BASE / 'output' / 'analyse_tier3_judge' / 'agreement_per_item.csv')))
    gate = json.loads((BASE / 'output' / 'analyse_tier3_judge'
                       / 'gate_verdicts.json').read_text())

    t = RtfTable(
        3,
        'Human Validation of the LLM Judge: Agreement per Rubric Item and '
        'Gate Verdict (n = 210 Double-Coded Responses)',
        [3300, 2450, 1300, 1300, 2250])
    t.row(['Item', 'κw judge–median [95% CI]',
           'κw H–H', 'AC2', 'Verdict'],
          borders=f'{BORDER_TOP}{BORDER_BOTTOM}', bold=True)

    for r in rows:
        n = int(r['item'].split('_')[1])
        label = f"{n}. {ITEM_LABELS[r['item']]}"
        ci = (f"{fmt_r(r['kappa_w_med'])} "
              f"[{fmt_r(r['kappa_w_med_lo'])}, {fmt_r(r['kappa_w_med_hi'])}]")
        ac2 = (float(r['ac2_JM']) + float(r['ac2_JN'])) / 2
        t.row([label, ci, fmt_r(r['kappa_w_HH']), fmt_r(ac2),
               STATUS_DISPLAY[r['status']]])

    comp = gate['composite']
    icc_ci = (f"{fmt_r(comp['icc_judge_vs_humanmean'])} "
              f"[{fmt_r(comp['icc_judge_vs_humanmean_ci'][0])}, "
              f"{fmt_r(comp['icc_judge_vs_humanmean_ci'][1])}]")
    t.row(['Relationship-oriented composite (ICC)', icc_ci,
           fmt_r(comp['icc_human_human']), '—',
           STATUS_DISPLAY[comp['status']]],
          borders=BORDER_TOP)
    sens = comp['sensitivity_excl_item06']
    sens_ci = (f"{fmt_r(sens['icc_judge_vs_humanmean'])} "
               f"[{fmt_r(sens['icc_judge_vs_humanmean_ci'][0])}, "
               f"{fmt_r(sens['icc_judge_vs_humanmean_ci'][1])}]")
    t.row(['  5-item variant, item 6 removed (ICC)', sens_ci,
           '—', '—', 'Sensitivity'],
          borders=BORDER_BOTTOM)

    t.note('\\u954?w = quadratic-weighted Cohen\\rquote s kappa between the '
           'DeepSeek-V4-Flash judge and the two-rater median, with 95% '
           'bootstrap CI (10,000 iterations); H\\endash H = human\\endash '
           'human \\u954?w (upper reference); AC2 = Gwet\\rquote s AC2 '
           '(quadratic weights, mean of judge vs. each rater), robust to '
           'skewed marginals. Verdicts (pre-specified): Confirmatory = '
           'CI lower bound \\u8805? .40 and \\u954?w H\\endash H \\u8805? '
           '.40; Saturated (descriptive) = near-ceiling item with almost no '
           'variance \\u8212? high raw agreement (AC2 \\u8805? .90) but '
           'kappa uninformative; Exploratory = agreement insufficient '
           '(item 6: the human criterion itself was unreliable, '
           '\\u954?w H\\endash H = .10). Composite rows: ICC(A,1), judge vs. '
           'mean of both raters, vs. the human\\endash human ceiling of '
           '.86; gate floor .50. The 6-item composite (items 1, 2, 3, 6, 7, '
           '10) is the relationship-oriented component validated by Ruben, '
           'Blanch\\endash Hartigan & Hall (2026); item 6 is retained per '
           'that construct despite its low human\\endash human reliability '
           'in this application. The 5-item variant (item 6 removed) is an '
           'application-specific sensitivity check; conclusions are '
           'unchanged.')
    t.save(OUT_DIR / 'tab3_validation_gate.rtf')


if __name__ == '__main__':
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_table1()
    build_table2()
    build_table3()
