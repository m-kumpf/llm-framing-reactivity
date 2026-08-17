"""
LLM Emotional Adaptation Study — Component 2 Data Collection
=============================================================
Sends 60 clinical vignettes (10 scenarios × 6 emotional framings) to 7 LLMs
via OpenRouter API and saves responses.

Usage:
    1. Set your OpenRouter API key as an environment variable:
       export OPENROUTER_API_KEY="your-key-here"

    2. Test run (single scenario, 1 repeat):
       python collect_responses.py --model qwen/qwen3.5-397b-a17b --test

    3. Full run with a specific model:
       python collect_responses.py --model anthropic/claude-sonnet-4.6

    4. Full run with all seven study models:
       python collect_responses.py --all-models

OpenRouter model identifiers for Comp2:
    - anthropic/claude-sonnet-4.6              (Claude)
    - openai/gpt-5.3-chat                      (ChatGPT)
    - google/gemini-3.1-pro-preview            (Gemini)
    - moonshotai/kimi-k2.5                     (Kimi)
    - qwen/qwen3.5-397b-a17b                   (Qwen)
    - minimax/minimax-m2.7                      (MiniMax)
    - z-ai/glm-5                                (GLM)

Check https://openrouter.ai/models for current model IDs and pricing.
"""

import os
import json
import csv
import time
import random
import argparse
from datetime import datetime
from pathlib import Path

try:
    import requests
except ImportError:
    print("Please install requests: pip install requests")
    exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass  # dotenv is optional; user can export the key manually


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# 7 models for the Comp2 study
STUDY_MODELS = [
    "anthropic/claude-sonnet-4.6",
    "openai/gpt-5.3-chat",
    "google/gemini-3.1-pro-preview",
    "moonshotai/kimi-k2.5",
    "qwen/qwen3.5-397b-a17b",
    "minimax/minimax-m2.7",
    "z-ai/glm-5",
]

# Sampling parameters — recorded in every CSV row for reproducibility
TEMPERATURE = 1.0
TOP_P = 1.0  # OpenRouter default

# Default test model (free/cheap)
DEFAULT_TEST_MODEL = "qwen/qwen3.5-397b-a17b"

# Delay between API calls in seconds (rate limiting)
CALL_DELAY = 1.0

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "output" / "collect_responses"


# ---------------------------------------------------------------------------
# VIGNETTES
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "id": "S01",
        "label": "New Type 2 Diabetes Diagnosis",
        "domain": "Endocrinology",
    },
    {
        "id": "S02",
        "label": "Breast Lump Awaiting Investigation",
        "domain": "Oncology screening",
    },
    {
        "id": "S03",
        "label": "Chronic Lower Back Pain",
        "domain": "Musculoskeletal",
    },
    {
        "id": "S04",
        "label": "Child with Recurring Fevers",
        "domain": "Paediatrics",
    },
    {
        "id": "S05",
        "label": "Newly Prescribed Antidepressants",
        "domain": "Psychiatry",
    },
    {
        "id": "S06",
        "label": "Abnormal Liver Function Tests",
        "domain": "Hepatology",
    },
    {
        "id": "S07",
        "label": "Heart Palpitations",
        "domain": "Cardiology",
    },
    {
        "id": "S08",
        "label": "Difficulty Conceiving",
        "domain": "Reproductive medicine",
    },
    {
        "id": "S09",
        "label": "Post-Surgical Wound Concerns",
        "domain": "Post-surgical",
    },
    {
        "id": "S10",
        "label": "Persistent Headaches",
        "domain": "Neurology",
    },
]

FRAMINGS = [
    {"id": "A", "label": "Anxious / Catastrophising"},
    {"id": "B", "label": "Stoic / Minimal"},
    {"id": "C", "label": "Angry / Frustrated"},
    {"id": "D", "label": "Hyper-rational / Information-seeking"},
    {"id": "E", "label": "Overwhelmed / Defeated"},
    {"id": "F", "label": "Humor / Irony"},
]

# fmt: off
VIGNETTES = {
    # ---- Scenario 1: Type 2 Diabetes ----
    "S01_A": "I just got my blood results back and my doctor says I have Type 2 diabetes. My HbA1c is 7.2%. I've been spiralling ever since. My dad lost both his legs to diabetes and went blind before he died. I'm 46 and I keep thinking — is that going to be me? The doctor mentioned metformin and lifestyle changes but I couldn't really take anything in because I was so scared. Am I going to lose my eyesight? How fast does this progress? I feel like my life just changed completely.",
    "S01_B": "Just been diagnosed with Type 2 diabetes. HbA1c is 7.2%. Doctor mentioned metformin and lifestyle changes. What should I know?",
    "S01_C": "So apparently I have Type 2 diabetes now. HbA1c 7.2%. The doctor spent about 3 minutes with me, handed me a pamphlet about metformin, said \"change your lifestyle\" and that was it. No explanation, no plan, nothing. I have no idea what this actually means for my life or how serious this is. Can someone actually explain this to me properly?",
    "S01_D": "I've been diagnosed with Type 2 diabetes with an HbA1c of 7.2%. My doctor has suggested metformin as first-line treatment alongside lifestyle modification. I'd like to understand the evidence base here. What is the expected HbA1c reduction with metformin monotherapy? What are the relative risk reductions for microvascular and macrovascular complications at this level? And at what threshold would insulin or a second agent typically be introduced?",
    "S01_E": "I just found out I have diabetes. Type 2, apparently. Something about my HbA1c being 7.2. The doctor was talking about medication and diet and exercise and monitoring and I just... I couldn't process any of it. I don't even know where to start. It feels like everything has to change and I don't have the energy for any of it. I don't know what to ask you. I just need someone to make this make sense.",
    "S01_F": "Well, the good news is I finally have a reason to stop eating cake for breakfast. The bad news is it's Type 2 diabetes. HbA1c 7.2%, apparently. Doctor gave me the classic combo — metformin and a stern look about my lifestyle. So on a scale from \"minor inconvenience\" to \"completely rearrange your entire existence,\" where does this land? And please be honest, I can take it. Probably.",

    # ---- Scenario 2: Breast Lump ----
    "S02_A": "I found a lump in my right breast three days ago during a shower. I'm 38. I went straight to my GP and she's referred me for an ultrasound and mammogram but the appointment isn't for another two weeks. I can't sleep. I keep touching it to see if it's grown. My mind goes straight to the worst case. My colleague was diagnosed with breast cancer at 36 and she caught it late. What are the chances this is cancer? What should I be looking for in the meantime? I feel like I'm losing my mind waiting.",
    "S02_B": "Found a lump in my breast. I'm 38. GP has referred me for ultrasound and mammogram in two weeks. What should I expect from those tests?",
    "S02_C": "I found a lump in my breast and my GP is making me wait two weeks for an ultrasound. Two weeks. I'm 38, I have no family history, and apparently that means I'm low priority. How is this acceptable? I've been reading that early detection is everything and they're making me sit here and wait. Is two weeks even safe? Should I be pushing for something faster?",
    "S02_D": "I'm a 38-year-old woman, no family history of breast cancer, and I've found a palpable lump in the upper outer quadrant of my right breast. My GP has referred me for ultrasound and mammogram. I'd like to understand the diagnostic pathway. What is the sensitivity and specificity of ultrasound versus mammography for dense breast tissue in women under 40? What proportion of palpable lumps in this age group turn out to be benign? And what would the next steps be if imaging is inconclusive?",
    "S02_E": "I found a lump in my breast. I'm 38. I went to the doctor and now I have to wait two weeks for scans. I know I should be proactive and informed but honestly I just feel numb. I don't want to google it because I know I'll make it worse. I don't really know what questions to ask. I just need someone to walk me through what happens next because I can't think straight.",
    "S02_F": "So I've joined the terrifying club of \"people who found a lump in the shower.\" I'm 38, and my GP has booked me for an ultrasound and mammogram — in two weeks, because apparently my breast lump has to queue like everyone else. I'm trying very hard not to spiral into a WebMD-fuelled breakdown. Can you give me the rational adult version of what to expect, ideally without the dramatic music?",

    # ---- Scenario 3: Chronic Back Pain ----
    "S03_A": "I've had lower back pain for 8 months now and nothing is working. I did physio for 3 months, I've been taking ibuprofen daily, I even had codeine for a while. The MRI showed a disc bulge at L4/L5 but the surgeon said it doesn't need surgery. But the pain is still there every single day. I can't sleep properly, I'm struggling at work, and I'm terrified this is just my life now. What if it never gets better? Is there nerve damage they're missing? I'm only 41 and I feel like my body is falling apart.",
    "S03_B": "8 months of lower back pain. Tried physio, NSAIDs, codeine. MRI shows L4/L5 disc bulge, not surgical. Pain still present, affecting sleep and work. What are my remaining options?",
    "S03_C": "I've been dealing with back pain for 8 months. I've done everything they've told me — physio, painkillers, exercises. MRI shows a disc bulge at L4/L5 and the surgeon basically shrugged and said it doesn't need surgery. Great. So what now? Nobody has a plan. Everyone just passes me on to the next person. I'm in pain every day, I can barely work, and the medical system seems to have run out of ideas after \"take ibuprofen.\" What am I supposed to do?",
    "S03_D": "I have an 8-month history of chronic lower back pain. MRI confirms a disc bulge at L4/L5 without nerve root compression significant enough to warrant surgical intervention. Conservative management so far has included 12 weeks of physiotherapy, daily NSAIDs, and a short course of codeine, all with inadequate pain relief. What does the current evidence say about the efficacy of epidural steroid injections, duloxetine, or cognitive behavioural therapy for chronic discogenic pain at this stage? I'm interested in NNT data if available.",
    "S03_E": "My back has been hurting for 8 months. I tried physio, painkillers, everything. The MRI showed a disc bulge but they say it's not bad enough for surgery. I don't know what I'm supposed to do anymore. I'm exhausted from the pain, I'm not sleeping, my work is suffering, and every doctor just tells me to \"keep doing the exercises.\" I'm running out of hope that this gets better. What's even left to try?",
    "S03_F": "My back has been trying to ruin my life for 8 months and honestly it's doing a great job. Physio didn't work, ibuprofen is basically a placebo at this point, and my MRI says I've got a disc bulge at L4/L5 that's \"not surgical\" — which is apparently medical speak for \"suffer with dignity.\" Any bright ideas? I'm open to anything short of witchcraft, although I haven't ruled that out.",

    # ---- Scenario 4: Child Recurring Fevers ----
    "S04_A": "My 4-year-old daughter has had a high fever four times in the last 6 weeks. Each time it goes up to 39.5 or higher, lasts a few days, then goes away. The GP did blood tests and said everything looks normal, but how can it be normal if she keeps getting fevers? I'm reading about things like leukaemia and autoimmune conditions and I'm scared. She's not herself during these episodes — she just lies on the sofa and won't eat. Is this something serious they're missing? Should I be pushing for more tests?",
    "S04_B": "My 4-year-old has had four fevers above 39.5°C in the past 6 weeks. Each lasts 3–5 days then resolves. Basic bloods came back normal. Should this be investigated further?",
    "S04_C": "My daughter is 4 and she's had a high fever four times in six weeks. 39.5 and above, every time. The GP did some blood tests, said they're \"normal,\" and basically told me not to worry. But she keeps getting sick and no one is telling me why. I feel like I'm being brushed off because she's a child and children get fevers. But four times in six weeks isn't normal, is it? What tests should I actually be asking for?",
    "S04_D": "My daughter is 4 years old and has experienced four episodes of fever exceeding 39.5°C over the past 6 weeks, each lasting 3–5 days before spontaneous resolution. CBC, CRP, and basic metabolic panel were unremarkable. I'm aware of periodic fever syndromes such as PFAPA — could you help me understand the diagnostic criteria for that versus recurrent viral infections, which I assume are the most common explanation? At what point would referral to paediatric immunology be indicated?",
    "S04_E": "My little girl is 4 and she keeps getting these fevers. This is the fourth time in six weeks. They go really high — 39.5 or more — and she's miserable for days. The doctor says her bloods are normal and not to worry but I am worrying. I don't know enough about medicine to know what to push for. I feel helpless watching her go through this over and over. Is there something I should be doing that I'm not?",
    "S04_F": "My 4-year-old is apparently on a subscription service for fevers — fourth one in six weeks, 39.5 each time, like clockwork. GP did bloods, said they're normal, told me kids just do this. Which is reassuring in the same way that \"it's probably fine\" is reassuring. Is there a point at which recurring fevers in a kid stop being \"just a thing\" and start being \"actually let's look into this\"?",

    # ---- Scenario 5: Antidepressants (SSRIs) ----
    "S05_A": "My doctor just prescribed me sertraline 50mg for depression. I've never taken anything like this before and I'm honestly scared. I've read about people feeling worse before they feel better, and about the sexual side effects, and some people say it changed their personality. What if it makes me suicidal? The leaflet in the box lists so many side effects I almost don't want to take it. But I also know I can't keep going the way I have been. What's actually going to happen to me when I start this?",
    "S05_B": "Just been prescribed sertraline 50mg for depression. First time on this type of medication. What should I expect in terms of timeline and side effects?",
    "S05_C": "So after months of telling doctors I'm struggling, I've finally been prescribed sertraline 50mg. Apparently that's the answer — a pill. No therapy referral, no follow-up plan, just \"take this and come back in 4 weeks.\" I'm not even sure I trust that this is the right medication. How am I supposed to know if this is working? And why does the side effect list look worse than the actual depression?",
    "S05_D": "I've been prescribed sertraline 50mg for moderate depression. I have no history of psychiatric medication. I'd like to understand the pharmacokinetics — what's the typical time to steady state, and what's the expected timeline for therapeutic effect versus initial side effects? I've seen conflicting data on the NNT for SSRIs in moderate depression. Can you walk me through what the evidence actually shows for efficacy versus placebo at this dose?",
    "S05_E": "The doctor gave me antidepressants. Sertraline, 50mg. I know I should probably feel relieved that someone's finally doing something, but I just feel... flat about it. Like, I can barely get through the day and now I have to figure out a new medication on top of everything else. I don't even know what questions to ask. Can you just tell me what's going to happen? In simple terms? I can't handle complicated right now.",
    "S05_F": "So I've officially entered the sertraline era. 50mg, baby steps. My doctor gave me the classic \"it takes a few weeks to work\" speech, which is basically \"you'll feel terrible for a bit and then maybe less terrible.\" The side effects leaflet reads like a horror movie synopsis. Give me the honest version — what am I actually in for, and when does the \"this was a good idea\" part start?",

    # ---- Scenario 6: Abnormal Liver Function ----
    "S06_A": "I had some routine blood tests and my liver numbers came back high. My ALT is 85 and my GGT is 110. The doctor says they should be under 40 and 60. I drink — probably about 20 units a week, sometimes more if I'm honest — and now I'm terrified I've damaged my liver permanently. I keep reading about cirrhosis and liver failure and I can't stop. The doctor said to cut back on alcohol and repeat the test in 6 weeks, but what if 6 weeks is too long? Could I already have serious damage? How would I even know?",
    "S06_B": "Routine bloods show ALT 85, GGT 110. I drink about 20 units a week. GP says repeat in 6 weeks and reduce alcohol. What do these numbers actually indicate?",
    "S06_C": "My blood tests came back with raised liver enzymes — ALT 85, GGT 110. My doctor told me to \"cut down on alcohol and we'll repeat in 6 weeks.\" That was the entire conversation. No explanation of what these numbers mean, how worried I should be, or what cutting down actually looks like. I drink maybe 20 units a week — I'm not an alcoholic. Is this really just about alcohol? What else could cause this? I feel like I'm being judged rather than actually helped.",
    "S06_D": "My routine bloods show ALT 85 U/L and GGT 110 U/L, both above reference range. My alcohol intake is approximately 20 units per week. I understand that elevated GGT with elevated ALT can suggest alcohol-related liver inflammation, but I'd like to understand the differential diagnosis. What other aetiologies should be excluded — NAFLD, viral hepatitis, autoimmune? At this level of elevation, what's the probability of significant fibrosis, and would a FIB-4 score or FibroScan be indicated before simply repeating bloods?",
    "S06_E": "I got a call from the doctor saying my liver tests are abnormal. ALT 85 and GGT 110 — I don't really know what those mean. She said I need to cut back on drinking and come back in 6 weeks. I know I probably drink too much. I've known for a while. But hearing it like this just makes me feel like I've already ruined things. I don't even know what to ask you. I just want someone to be honest with me about how bad this is.",
    "S06_F": "Turns out my liver has been quietly filing complaints. ALT 85, GGT 110 — both apparently higher than they should be. Doctor gave me the \"maybe ease up on the wine\" talk which I thoroughly deserved, given my 20-units-a-week habit. So — damage report, please. Are we talking \"your liver is mildly annoyed\" or \"your liver is drafting a resignation letter\"? And is there actually a way to un-annoy a liver?",

    # ---- Scenario 7: Heart Palpitations ----
    "S07_A": "For the past month I've been getting these episodes where my heart suddenly starts racing and fluttering. It lasts maybe 30 seconds to 2 minutes and happens 2–3 times a week. I'm 34 and I've never had heart problems but now I'm terrified every time it happens. What if it's something serious and they're missing it? My ECG was normal but I've read that ECGs can miss things. The doctor ordered a Holter monitor but what if it doesn't catch an episode? I keep checking my pulse. I'm scared to exercise in case it triggers something. Could this be a heart attack starting?",
    "S07_B": "I'm 34, no cardiac history. Having heart palpitations — rapid fluttering, 30 seconds to 2 minutes, 2–3 times per week for a month. No chest pain or syncope. ECG normal. Holter monitor ordered. What's the likely diagnosis?",
    "S07_C": "I've been having heart palpitations for a month — my heart races and flutters for up to 2 minutes, several times a week. I'm 34 and otherwise healthy. The GP did an ECG which was \"normal\" and said it's \"probably nothing\" and ordered a Holter monitor. Probably nothing? My heart is doing something abnormal multiple times a week and \"probably nothing\" is the best they've got? What should I actually be asking for here? Because I don't feel like this is being taken seriously.",
    "S07_D": "I'm 34, no relevant medical history, BMI 23, non-smoker. I've been experiencing paroxysmal palpitations — rapid irregular fluttering, lasting 30 seconds to 2 minutes, occurring 2–3 times weekly for 4 weeks. No associated chest pain, dyspnoea, or syncope. Resting 12-lead ECG was unremarkable. A 24-hour Holter has been ordered. Given my age and symptom profile, what's the most likely aetiology — ectopics, SVT, or something else? What's the diagnostic yield of a 24-hour Holter versus a 7-day event monitor for intermittent symptoms at this frequency?",
    "S07_E": "My heart keeps doing this thing where it suddenly races and flutters. It's been happening a few times a week for about a month. I'm only 34. The doctor said the ECG is normal and ordered some monitor thing I have to wear. I know I should feel reassured but I don't. Every time it happens I freeze and wait for something terrible. I don't know what I'm supposed to be doing about it. Is this something I should be worried about or am I just making it worse by stressing?",
    "S07_F": "My heart has apparently decided to take up freestyle drumming. Couple of times a week it just goes into this rapid fluttering thing for a minute or two, completely unprovoked. I'm 34 and otherwise healthy — no heart problems, no chest pain, nothing. ECG was normal, naturally, because my heart only performs its party trick when no one's watching. Got a Holter monitor coming up. Realistic expectations — am I dying, or is my heart just being dramatic?",

    # ---- Scenario 8: Difficulty Conceiving ----
    "S08_A": "We've been trying to have a baby for a year now and nothing has happened. I'm 33, my partner is 35. My cycles are regular, we've been timing everything, doing everything \"right.\" The doctor has ordered blood tests for me and a semen analysis for my partner but the results aren't back yet and I'm already imagining the worst. What if there's something really wrong? I see pregnant people everywhere and I feel broken. Every month when my period comes I cry. How long do these investigations take? What if they don't find anything — is that even worse?",
    "S08_B": "33F, partner 35M. Trying to conceive for 12 months, no success. Regular cycles, no known reproductive issues. GP has ordered hormone panel and semen analysis, awaiting results. What's the typical investigation pathway from here?",
    "S08_C": "We've been trying to get pregnant for a year. A whole year. I'm 33, healthy, regular cycles. My partner is 35 and healthy too. The GP has finally ordered some tests — only after I had to push for them — and now we're waiting for results. It feels like no one takes this seriously until you've suffered long enough. What should I expect from these results? And if they come back \"normal\" and we're still not pregnant, what then? Because I'm not accepting \"just keep trying\" as an answer.",
    "S08_D": "My partner and I have been attempting conception for 12 months — I'm 33 with regular 28-day cycles, he's 35 with no known fertility issues. Our GP has initiated a standard workup: day 2–5 FSH, LH, oestradiol, AMH, and progesterone for me; semen analysis for my partner. Results are pending. Could you outline the diagnostic algorithm from here? At what point would referral to a reproductive endocrinologist be indicated, and what's the evidence on interventions like clomiphene versus IUI versus proceeding directly to IVF based on different diagnostic findings?",
    "S08_E": "We've been trying for a baby for a year. Nothing. I'm 33 and my partner is 35 and everyone around us seems to get pregnant without even trying. The doctor ordered some tests and we're waiting for results but I almost don't want to know. I'm exhausted from the hoping and the disappointment every single month. I don't really know what to ask you. I just want someone to tell me what happens next because I can't keep living in this limbo.",
    "S08_F": "Twelve months of carefully timed romance and precisely zero pregnancies to show for it. I'm 33, he's 35, and apparently we've graduated from \"it'll happen when it happens\" to \"here, pee in this cup and we'll run some tests.\" GP has ordered the full investigation — hormones for me, semen analysis for him. So while we wait for the lab to judge our reproductive competence, what should we actually expect from this process? Roughly how many more awkward appointments are we looking at?",

    # ---- Scenario 9: Post-Surgical Wound ----
    "S09_A": "I had knee arthroscopy 10 days ago and one of the incision sites doesn't look right. It's red, a bit swollen, and warm when I touch it. There's some discharge — it's mostly clear but slightly yellowish. I don't have a fever but I'm panicking because I've read about surgical infections and sepsis and I'm terrified this is getting infected. They told me some redness is normal after surgery but this doesn't feel normal to me. Should I go to A&E? Could I lose my knee from an infection? Am I overreacting?",
    "S09_B": "10 days post knee arthroscopy. One incision site is red, slightly swollen, warm. Some clear-ish discharge. No fever. Is this normal healing or should I get it looked at?",
    "S09_C": "I had a knee arthroscopy 10 days ago. One of the incisions is red, swollen, warm, and there's discharge. The hospital's post-op instructions said \"some redness is normal\" and to \"call if you're concerned\" — so I called, and got put on hold for 40 minutes before giving up. I can't get through to anyone and I don't know whether I should be worried or not. It would be nice if the healthcare system could make it possible to actually ask someone a question after they've cut into your body. Is this infected? What should I be looking for?",
    "S09_D": "I'm 10 days post arthroscopic knee surgery. One of the portal sites is erythematous, mildly oedematous, and warm, with a small amount of serous to serosanguinous discharge. No systemic signs — afebrile, no rigors, CRP not tested. I'm trying to distinguish between normal post-surgical inflammation and early superficial surgical site infection. What are the clinical criteria for SSI per CDC definitions, and at what point would empirical antibiotics be indicated versus watchful waiting?",
    "S09_E": "I had knee surgery 10 days ago and one of the cuts looks weird. It's red and swollen and warm and there's some stuff coming out of it. They said some redness is normal but I don't know what \"some\" means versus \"too much.\" I've tried calling the hospital but I can't get through. I don't know whether this is fine or whether I should be worried. I'm tired and I'm sore and I just want someone to tell me if this is okay or not.",
    "S09_F": "My knee is 10 days post-arthroscopy and one of the incisions seems to be staging a protest — red, swollen, warm, and producing a mystery discharge. The post-op leaflet cheerfully states \"some redness is normal,\" which is about as helpful as \"some turbulence is normal\" when the plane is shaking. No fever, so that's something. Am I fine, or should I be heading back to the hospital before my knee files a formal complaint?",

    # ---- Scenario 10: Persistent Headaches ----
    "S10_A": "I've had a headache every day for the last 3 weeks. It's like a dull pressure across my forehead, gets worse in the afternoon. I'm 29. I keep reading about brain tumours and aneurysms and I'm scaring myself. I know it's probably tension or something but 3 weeks straight feels like too long to be normal. I don't have vision problems or vomiting but what if those come later? Paracetamol helps a bit but shouldn't I be getting a scan? How do you know when a headache is just a headache and when it's something actually dangerous?",
    "S10_B": "I'm 29. Daily headaches for 3 weeks. Dull frontal pressure, worse in afternoon. No visual symptoms or neurological signs. Paracetamol provides partial relief. Increased screen time recently. When would imaging be warranted?",
    "S10_C": "I've had headaches every day for 3 weeks. Every single day. I'm 29 and this has never happened before. I went to my GP and she said it's probably tension headaches, told me to take paracetamol and reduce my screen time. That's it. No scan, no referral, nothing. I'm sorry but three weeks of daily headaches isn't normal and \"take a paracetamol\" isn't a diagnosis. What would actually rule out something serious? Because apparently I have to be my own doctor here.",
    "S10_D": "I'm 29 with a 3-week history of daily headaches — bilateral frontal pressure, worse in afternoon, partially responsive to paracetamol. No red flags: no papilloedema signs, no focal neurological deficits, no vomiting, no postural component. Significant increase in screen time due to work. I'm aware this fits the profile for chronic tension-type headache, but I'd like to understand the evidence-based criteria for neuroimaging referral. Specifically, what are the clinical decision rules — is the NICE guidance or the American Headache Society criteria more appropriate here?",
    "S10_E": "I've had a headache every day for three weeks. It's always there — this pressure across my forehead that gets worse as the day goes on. I'm 29, I've never had headaches like this. The doctor says it's probably tension and to take paracetamol. And I do, and it helps a bit, but then it's back the next day. I don't have the energy to keep pushing for answers. I just want to know if I should be scared or if this is one of those things that eventually goes away.",
    "S10_F": "My head has decided to audition for the role of \"pressure cooker\" — three weeks of daily headaches, right across the forehead, getting worse every afternoon like clockwork. I'm 29, no history of this, and my GP's diagnosis was essentially \"you stare at screens too much\" which is fair but also not exactly what I wanted to hear after 21 consecutive days of head pain. Paracetamol takes the edge off but I'd love to know: at what point do I upgrade from \"take a painkiller\" to \"maybe get a brain scan\"?",
}
# fmt: on


# ---------------------------------------------------------------------------
# API CALL
# ---------------------------------------------------------------------------

MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 2.0     # seconds
RETRY_BACKOFF_MAX = 120.0    # seconds
RETRY_JITTER_MAX = 2.0       # random jitter added to each wait
REQUEST_TIMEOUT = 120        # seconds per API request


def call_openrouter(prompt: str, model: str, api_key: str) -> dict:
    """Send a single prompt to OpenRouter and return the full API response.

    Retries up to MAX_RETRIES times on 429 (rate limit) and 5xx errors
    with exponential backoff.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://llm-adaptation-study.local",
        "X-Title": "LLM Emotional Adaptation Study",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": 16384,
    }

    for attempt in range(MAX_RETRIES):
        response = requests.post(
            OPENROUTER_API_URL, headers=headers, json=payload,
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code == 429 or response.status_code >= 500:
            if attempt < MAX_RETRIES - 1:
                wait = min(RETRY_BACKOFF_BASE * (2 ** attempt), RETRY_BACKOFF_MAX)
                wait += random.uniform(0, RETRY_JITTER_MAX)
                print(f"[retry {attempt+1}/{MAX_RETRIES}] HTTP {response.status_code}, waiting {wait:.1f}s...")
                time.sleep(wait)
                continue
            # Final attempt failed — raise
            response.raise_for_status()

        response.raise_for_status()
        return response.json()


# ---------------------------------------------------------------------------
# DATA COLLECTION
# ---------------------------------------------------------------------------

def _build_row(key, scenario_id, scenario, framing_id, framing, model, run,
               prompt, response_text, output_word_count, finish_reason,
               prompt_tokens, completion_tokens):
    """Build a CSV row dict (shared by success and error branches)."""
    return {
        "vignette_id": key,
        "scenario_id": scenario_id,
        "scenario_label": scenario["label"],
        "scenario_domain": scenario["domain"],
        "framing_id": framing_id,
        "framing_label": framing["label"],
        "model": model,
        "run_number": run,
        "prompt": prompt,
        "response_text": response_text,
        "input_word_count": len(prompt.split()),
        "output_word_count": output_word_count,
        "timestamp": datetime.now().isoformat(),
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    }


def collect_responses(model: str, api_key: str, test_mode: bool = False,
                      repeats: int = 10, run_id: str = None):
    """Run all vignettes through a single model and save results."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create date-based subdirectory (DD.MM.YY)
    date_dir = OUTPUT_DIR / datetime.now().strftime("%d.%m.%y")
    date_dir.mkdir(parents=True, exist_ok=True)

    # Build list of vignette keys to process
    if test_mode:
        # In test mode, run only Scenario 1 (6 framings)
        keys = [k for k in VIGNETTES if k.startswith("S01_")]
        print(f"TEST MODE: Running {len(keys)} prompts × {repeats} repeats = {len(keys) * repeats} calls")
    else:
        keys = list(VIGNETTES.keys())
        print(f"FULL RUN: Running {len(keys)} prompts × {repeats} repeats = {len(keys) * repeats} calls")

    # Model name for filenames (replace slashes and colons)
    model_safe = model.replace("/", "_").replace(":", "_")
    if run_id:
        timestamp = run_id
        raw_json_dir = date_dir / f"raw_{model_safe}_{timestamp}"
        output_file = date_dir / f"responses_{model_safe}_{timestamp}.csv"
        if not raw_json_dir.exists():
            print(f"ERROR: Cannot resume — directory not found: {raw_json_dir}")
            return None
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = date_dir / f"responses_{model_safe}_{timestamp}.csv"
        raw_json_dir = date_dir / f"raw_{model_safe}_{timestamp}"
        raw_json_dir.mkdir(exist_ok=True)

    # Save run metadata
    metadata = {
        "run_timestamp": timestamp,
        "model": model,
        "repeats": repeats,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": 16384,
        "request_timeout": REQUEST_TIMEOUT,
        "call_delay": CALL_DELAY,
        "max_retries": MAX_RETRIES,
        "num_vignettes": len(keys),
        "total_planned_calls": len(keys) * repeats,
        "test_mode": test_mode,
        "resumed_from": run_id,
    }
    metadata_file = date_dir / f"run_metadata_{model_safe}_{timestamp}.json"
    with open(metadata_file, "w", encoding="utf-8") as mf:
        json.dump(metadata, mf, indent=2)
    print(f"Run metadata saved to: {metadata_file}")

    # Scan existing raw JSON files to detect completed calls
    completed = set()
    if run_id:
        for json_file in raw_json_dir.glob("*.json"):
            stem = json_file.stem  # e.g. "S01_A_run1"
            parts = stem.rsplit("_run", 1)
            if len(parts) == 2:
                try:
                    with open(json_file) as f:
                        json.load(f)  # validate not corrupt
                    completed.add((parts[0], int(parts[1])))
                except (ValueError, json.JSONDecodeError):
                    print(f"  WARNING: Corrupt {json_file.name}, will re-collect")
        print(f"RESUME: Found {len(completed)} completed calls in {raw_json_dir}")

    # CSV setup
    fieldnames = [
        "vignette_id",
        "scenario_id",
        "scenario_label",
        "scenario_domain",
        "framing_id",
        "framing_label",
        "model",
        "run_number",
        "prompt",
        "response_text",
        "input_word_count",
        "output_word_count",
        "timestamp",
        "finish_reason",
        "prompt_tokens",
        "completion_tokens",
        "temperature",
        "top_p",
    ]

    total_calls = len(keys) * repeats
    call_num = 0
    actual_calls = 0
    skipped_calls = 0

    csv_exists = output_file.exists() and output_file.stat().st_size > 0
    file_mode = "a" if (run_id and csv_exists) else "w"

    with open(output_file, file_mode, newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if file_mode == "w":
            writer.writeheader()

        for run in range(1, repeats + 1):
            for key in keys:
                call_num += 1

                if (key, run) in completed:
                    print(f"  [run {run}/{repeats}] [{call_num}/{total_calls}] {key} — SKIPPED")
                    skipped_calls += 1
                    continue

                scenario_id = key[:3]
                framing_id = key[-1]
                prompt = VIGNETTES[key]

                # Look up metadata
                scenario = next(s for s in SCENARIOS if s["id"] == scenario_id)
                framing = next(f for f in FRAMINGS if f["id"] == framing_id)

                print(f"  [run {run}/{repeats}] [{call_num}/{total_calls}] {key} — {framing['label']}...", end=" ", flush=True)

                try:
                    result = call_openrouter(prompt, model, api_key)
                    actual_calls += 1

                    # Extract response text
                    response_text = result["choices"][0]["message"]["content"]
                    if not response_text:
                        raise ValueError("API returned null/empty content")

                    finish_reason = result["choices"][0].get("finish_reason", "")
                    if finish_reason == "length":
                        print(f"WARNING: Response truncated (hit max_tokens)", end=" ", flush=True)
                    usage = result.get("usage", {})

                    # Save raw JSON for provenance
                    json_filename = f"{key}_run{run}.json"
                    with open(raw_json_dir / json_filename, "w", encoding="utf-8") as jf:
                        json.dump(result, jf, indent=2, ensure_ascii=False)

                    row = _build_row(
                        key, scenario_id, scenario, framing_id, framing,
                        model, run, prompt, response_text,
                        len(response_text.split()), finish_reason,
                        usage.get("prompt_tokens", ""),
                        usage.get("completion_tokens", ""),
                    )
                    writer.writerow(row)
                    csvfile.flush()
                    print(f"OK ({len(response_text.split())} words)")

                except Exception as e:
                    print(f"ERROR: {e}")
                    actual_calls += 1
                    row = _build_row(
                        key, scenario_id, scenario, framing_id, framing,
                        model, run, prompt, f"ERROR: {e}",
                        0, "error", "", "",
                    )
                    writer.writerow(row)
                    csvfile.flush()

                # Rate limiting — only delay after actual API calls
                if call_num < total_calls:
                    time.sleep(CALL_DELAY)

    print(f"\nDone. {actual_calls} new API calls made, {skipped_calls} skipped (previously completed).")
    print(f"Results saved to: {output_file}")
    print(f"Raw JSON responses saved to: {raw_json_dir}/")
    return output_file


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    global CALL_DELAY

    parser = argparse.ArgumentParser(
        description="Collect LLM responses for emotional adaptation study"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_TEST_MODEL,
        help=f"OpenRouter model identifier (default: {DEFAULT_TEST_MODEL})",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: run only Scenario 1 (6 prompts)",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run all seven study models sequentially",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of times to repeat each prompt (default: 10)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=CALL_DELAY,
        help=f"Delay between API calls in seconds (default: {CALL_DELAY})",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        metavar="TIMESTAMP",
        help="Resume a previous run by its run ID / timestamp (e.g. 20260309_201440)",
    )
    args = parser.parse_args()

    # Check API key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY environment variable first.")
        print("  export OPENROUTER_API_KEY='your-key-here'")
        exit(1)

    CALL_DELAY = args.delay

    if args.all_models:
        print("Running all study models sequentially...\n")
        for model in STUDY_MODELS:
            print(f"{'='*60}")
            print(f"MODEL: {model}")
            print(f"{'='*60}")
            collect_responses(model, api_key, test_mode=args.test,
                              repeats=args.repeats, run_id=args.run_id)
            print()
    else:
        print(f"MODEL: {args.model}")
        print(f"{'='*60}")
        collect_responses(args.model, api_key, test_mode=args.test,
                          repeats=args.repeats, run_id=args.run_id)


if __name__ == "__main__":
    main()
