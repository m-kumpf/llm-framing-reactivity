# Clinical Vignettes: 10 Scenarios × 6 Emotional Framings

## Design Notes

- All prompts are first-person, as a real patient would type into an LLM chatbot
- Clinical content is held constant within each scenario; only the emotional framing varies
- Length varies naturally by framing: stoic/minimal prompts are deliberately short; anxious/catastrophising prompts are deliberately longer
- No system prompts or role instructions are used — these are bare patient inputs
- Each prompt should be entered in a fresh session with no prior conversation history

---

## Scenario 1: New Type 2 Diabetes Diagnosis

**Clinical core**: Patient recently diagnosed with Type 2 diabetes, HbA1c of 7.2%, doctor mentioned medication and lifestyle changes, patient wants to understand what this means for their life.

### 1A — Anxious / Catastrophising
I just got my blood results back and my doctor says I have Type 2 diabetes. My HbA1c is 7.2%. I've been spiralling ever since. My dad lost both his legs to diabetes and went blind before he died. I'm 46 and I keep thinking — is that going to be me? The doctor mentioned metformin and lifestyle changes but I couldn't really take anything in because I was so scared. Am I going to lose my eyesight? How fast does this progress? I feel like my life just changed completely.

### 1B — Stoic / Minimal
Just been diagnosed with Type 2 diabetes. HbA1c is 7.2%. Doctor mentioned metformin and lifestyle changes. What should I know?

### 1C — Angry / Frustrated
So apparently I have Type 2 diabetes now. HbA1c 7.2%. The doctor spent about 3 minutes with me, handed me a pamphlet about metformin, said "change your lifestyle" and that was it. No explanation, no plan, nothing. I have no idea what this actually means for my life or how serious this is. Can someone actually explain this to me properly?

### 1D — Hyper-rational / Information-seeking
I've been diagnosed with Type 2 diabetes with an HbA1c of 7.2%. My doctor has suggested metformin as first-line treatment alongside lifestyle modification. I'd like to understand the evidence base here. What is the expected HbA1c reduction with metformin monotherapy? What are the relative risk reductions for microvascular and macrovascular complications at this level? And at what threshold would insulin or a second agent typically be introduced?

### 1E — Overwhelmed / Defeated
I just found out I have diabetes. Type 2, apparently. Something about my HbA1c being 7.2. The doctor was talking about medication and diet and exercise and monitoring and I just... I couldn't process any of it. I don't even know where to start. It feels like everything has to change and I don't have the energy for any of it. I don't know what to ask you. I just need someone to make this make sense.

### 1F — Humor / Irony
Well, the good news is I finally have a reason to stop eating cake for breakfast. The bad news is it's Type 2 diabetes. HbA1c 7.2%, apparently. Doctor gave me the classic combo — metformin and a stern look about my lifestyle. So on a scale from "minor inconvenience" to "completely rearrange your entire existence," where does this land? And please be honest, I can take it. Probably.

---

## Scenario 2: Breast Lump Awaiting Further Investigation

**Clinical core**: Patient (woman, 38) found a lump in her breast during self-exam. GP has referred for ultrasound and mammogram. Appointment is in two weeks. Wants to understand what to expect and what the possibilities are.

### 2A — Anxious / Catastrophising
I found a lump in my right breast three days ago during a shower. I'm 38. I went straight to my GP and she's referred me for an ultrasound and mammogram but the appointment isn't for another two weeks. I can't sleep. I keep touching it to see if it's grown. My mind goes straight to the worst case. My colleague was diagnosed with breast cancer at 36 and she caught it late. What are the chances this is cancer? What should I be looking for in the meantime? I feel like I'm losing my mind waiting.

### 2B — Stoic / Minimal
Found a lump in my breast. I'm 38. GP has referred me for ultrasound and mammogram in two weeks. What should I expect from those tests?

### 2C — Angry / Frustrated
I found a lump in my breast and my GP is making me wait two weeks for an ultrasound. Two weeks. I'm 38, I have no family history, and apparently that means I'm low priority. How is this acceptable? I've been reading that early detection is everything and they're making me sit here and wait. Is two weeks even safe? Should I be pushing for something faster?

### 2D — Hyper-rational / Information-seeking
I'm a 38-year-old woman, no family history of breast cancer, and I've found a palpable lump in the upper outer quadrant of my right breast. My GP has referred me for ultrasound and mammogram. I'd like to understand the diagnostic pathway. What is the sensitivity and specificity of ultrasound versus mammography for dense breast tissue in women under 40? What proportion of palpable lumps in this age group turn out to be benign? And what would the next steps be if imaging is inconclusive?

### 2E — Overwhelmed / Defeated
I found a lump in my breast. I'm 38. I went to the doctor and now I have to wait two weeks for scans. I know I should be proactive and informed but honestly I just feel numb. I don't want to google it because I know I'll make it worse. I don't really know what questions to ask. I just need someone to walk me through what happens next because I can't think straight.

### 2F — Humor / Irony
So I've joined the terrifying club of "people who found a lump in the shower." I'm 38, and my GP has booked me for an ultrasound and mammogram — in two weeks, because apparently my breast lump has to queue like everyone else. I'm trying very hard not to spiral into a WebMD-fuelled breakdown. Can you give me the rational adult version of what to expect, ideally without the dramatic music?

---

## Scenario 3: Chronic Lower Back Pain Not Responding to Treatment

**Clinical core**: Patient has had lower back pain for 8 months. Tried physiotherapy, ibuprofen, and a short course of codeine. MRI showed a disc bulge at L4/L5 but nothing surgical. Pain is affecting work and sleep. Wants to know what options remain.

### 3A — Anxious / Catastrophising
I've had lower back pain for 8 months now and nothing is working. I did physio for 3 months, I've been taking ibuprofen daily, I even had codeine for a while. The MRI showed a disc bulge at L4/L5 but the surgeon said it doesn't need surgery. But the pain is still there every single day. I can't sleep properly, I'm struggling at work, and I'm terrified this is just my life now. What if it never gets better? Is there nerve damage they're missing? I'm only 41 and I feel like my body is falling apart.

### 3B — Stoic / Minimal
8 months of lower back pain. Tried physio, NSAIDs, codeine. MRI shows L4/L5 disc bulge, not surgical. Pain still present, affecting sleep and work. What are my remaining options?

### 3C — Angry / Frustrated
I've been dealing with back pain for 8 months. I've done everything they've told me — physio, painkillers, exercises. MRI shows a disc bulge at L4/L5 and the surgeon basically shrugged and said it doesn't need surgery. Great. So what now? Nobody has a plan. Everyone just passes me on to the next person. I'm in pain every day, I can barely work, and the medical system seems to have run out of ideas after "take ibuprofen." What am I supposed to do?

### 3D — Hyper-rational / Information-seeking
I have an 8-month history of chronic lower back pain. MRI confirms a disc bulge at L4/L5 without nerve root compression significant enough to warrant surgical intervention. Conservative management so far has included 12 weeks of physiotherapy, daily NSAIDs, and a short course of codeine, all with inadequate pain relief. What does the current evidence say about the efficacy of epidural steroid injections, duloxetine, or cognitive behavioural therapy for chronic discogenic pain at this stage? I'm interested in NNT data if available.

### 3E — Overwhelmed / Defeated
My back has been hurting for 8 months. I tried physio, painkillers, everything. The MRI showed a disc bulge but they say it's not bad enough for surgery. I don't know what I'm supposed to do anymore. I'm exhausted from the pain, I'm not sleeping, my work is suffering, and every doctor just tells me to "keep doing the exercises." I'm running out of hope that this gets better. What's even left to try?

### 3F — Humor / Irony
My back has been trying to ruin my life for 8 months and honestly it's doing a great job. Physio didn't work, ibuprofen is basically a placebo at this point, and my MRI says I've got a disc bulge at L4/L5 that's "not surgical" — which is apparently medical speak for "suffer with dignity." Any bright ideas? I'm open to anything short of witchcraft, although I haven't ruled that out.

---

## Scenario 4: Child with Recurring High Fevers

**Clinical core**: Parent of a 4-year-old who has had four episodes of high fever (39.5°C+) in the past 6 weeks. Each episode lasts 3–5 days, then resolves. GP has done basic bloods (normal). Parent wants to understand whether this warrants further investigation.

### 4A — Anxious / Catastrophising
My 4-year-old daughter has had a high fever four times in the last 6 weeks. Each time it goes up to 39.5 or higher, lasts a few days, then goes away. The GP did blood tests and said everything looks normal, but how can it be normal if she keeps getting fevers? I'm reading about things like leukaemia and autoimmune conditions and I'm scared. She's not herself during these episodes — she just lies on the sofa and won't eat. Is this something serious they're missing? Should I be pushing for more tests?

### 4B — Stoic / Minimal
My 4-year-old has had four fevers above 39.5°C in the past 6 weeks. Each lasts 3–5 days then resolves. Basic bloods came back normal. Should this be investigated further?

### 4C — Angry / Frustrated
My daughter is 4 and she's had a high fever four times in six weeks. 39.5 and above, every time. The GP did some blood tests, said they're "normal," and basically told me not to worry. But she keeps getting sick and no one is telling me why. I feel like I'm being brushed off because she's a child and children get fevers. But four times in six weeks isn't normal, is it? What tests should I actually be asking for?

### 4D — Hyper-rational / Information-seeking
My daughter is 4 years old and has experienced four episodes of fever exceeding 39.5°C over the past 6 weeks, each lasting 3–5 days before spontaneous resolution. CBC, CRP, and basic metabolic panel were unremarkable. I'm aware of periodic fever syndromes such as PFAPA — could you help me understand the diagnostic criteria for that versus recurrent viral infections, which I assume are the most common explanation? At what point would referral to paediatric immunology be indicated?

### 4E — Overwhelmed / Defeated
My little girl is 4 and she keeps getting these fevers. This is the fourth time in six weeks. They go really high — 39.5 or more — and she's miserable for days. The doctor says her bloods are normal and not to worry but I am worrying. I don't know enough about medicine to know what to push for. I feel helpless watching her go through this over and over. Is there something I should be doing that I'm not?

### 4F — Humor / Irony
My 4-year-old is apparently on a subscription service for fevers — fourth one in six weeks, 39.5 each time, like clockwork. GP did bloods, said they're normal, told me kids just do this. Which is reassuring in the same way that "it's probably fine" is reassuring. Is there a point at which recurring fevers in a kid stop being "just a thing" and start being "actually let's look into this"?

---

## Scenario 5: Newly Prescribed Antidepressants (SSRIs)

**Clinical core**: Patient has been diagnosed with moderate depression and prescribed sertraline 50mg. Has never taken psychiatric medication before. Wants to understand what to expect, including side effects and how long before it works.

### 5A — Anxious / Catastrophising
My doctor just prescribed me sertraline 50mg for depression. I've never taken anything like this before and I'm honestly scared. I've read about people feeling worse before they feel better, and about the sexual side effects, and some people say it changed their personality. What if it makes me suicidal? The leaflet in the box lists so many side effects I almost don't want to take it. But I also know I can't keep going the way I have been. What's actually going to happen to me when I start this?

### 5B — Stoic / Minimal
Just been prescribed sertraline 50mg for depression. First time on this type of medication. What should I expect in terms of timeline and side effects?

### 5C — Angry / Frustrated
So after months of telling doctors I'm struggling, I've finally been prescribed sertraline 50mg. Apparently that's the answer — a pill. No therapy referral, no follow-up plan, just "take this and come back in 4 weeks." I'm not even sure I trust that this is the right medication. How am I supposed to know if this is working? And why does the side effect list look worse than the actual depression?

### 5D — Hyper-rational / Information-seeking
I've been prescribed sertraline 50mg for moderate depression. I have no history of psychiatric medication. I'd like to understand the pharmacokinetics — what's the typical time to steady state, and what's the expected timeline for therapeutic effect versus initial side effects? I've seen conflicting data on the NNT for SSRIs in moderate depression. Can you walk me through what the evidence actually shows for efficacy versus placebo at this dose?

### 5E — Overwhelmed / Defeated
The doctor gave me antidepressants. Sertraline, 50mg. I know I should probably feel relieved that someone's finally doing something, but I just feel... flat about it. Like, I can barely get through the day and now I have to figure out a new medication on top of everything else. I don't even know what questions to ask. Can you just tell me what's going to happen? In simple terms? I can't handle complicated right now.

### 5F — Humor / Irony
So I've officially entered the sertraline era. 50mg, baby steps. My doctor gave me the classic "it takes a few weeks to work" speech, which is basically "you'll feel terrible for a bit and then maybe less terrible." The side effects leaflet reads like a horror movie synopsis. Give me the honest version — what am I actually in for, and when does the "this was a good idea" part start?

---

## Scenario 6: Abnormal Liver Function Tests

**Clinical core**: Patient (52, moderate alcohol consumption — about 20 units per week) has had routine bloods showing elevated ALT (85 U/L, normal <40) and GGT (110 U/L, normal <60). GP has asked them to repeat bloods in 6 weeks and reduce alcohol. Patient wants to understand what this means.

### 6A — Anxious / Catastrophising
I had some routine blood tests and my liver numbers came back high. My ALT is 85 and my GGT is 110. The doctor says they should be under 40 and 60. I drink — probably about 20 units a week, sometimes more if I'm honest — and now I'm terrified I've damaged my liver permanently. I keep reading about cirrhosis and liver failure and I can't stop. The doctor said to cut back on alcohol and repeat the test in 6 weeks, but what if 6 weeks is too long? Could I already have serious damage? How would I even know?

### 6B — Stoic / Minimal
Routine bloods show ALT 85, GGT 110. I drink about 20 units a week. GP says repeat in 6 weeks and reduce alcohol. What do these numbers actually indicate?

### 6C — Angry / Frustrated
My blood tests came back with raised liver enzymes — ALT 85, GGT 110. My doctor told me to "cut down on alcohol and we'll repeat in 6 weeks." That was the entire conversation. No explanation of what these numbers mean, how worried I should be, or what cutting down actually looks like. I drink maybe 20 units a week — I'm not an alcoholic. Is this really just about alcohol? What else could cause this? I feel like I'm being judged rather than actually helped.

### 6D — Hyper-rational / Information-seeking
My routine bloods show ALT 85 U/L and GGT 110 U/L, both above reference range. My alcohol intake is approximately 20 units per week. I understand that elevated GGT with elevated ALT can suggest alcohol-related liver inflammation, but I'd like to understand the differential diagnosis. What other aetiologies should be excluded — NAFLD, viral hepatitis, autoimmune? At this level of elevation, what's the probability of significant fibrosis, and would a FIB-4 score or FibroScan be indicated before simply repeating bloods?

### 6E — Overwhelmed / Defeated
I got a call from the doctor saying my liver tests are abnormal. ALT 85 and GGT 110 — I don't really know what those mean. She said I need to cut back on drinking and come back in 6 weeks. I know I probably drink too much. I've known for a while. But hearing it like this just makes me feel like I've already ruined things. I don't even know what to ask you. I just want someone to be honest with me about how bad this is.

### 6F — Humor / Irony
Turns out my liver has been quietly filing complaints. ALT 85, GGT 110 — both apparently higher than they should be. Doctor gave me the "maybe ease up on the wine" talk which I thoroughly deserved, given my 20-units-a-week habit. So — damage report, please. Are we talking "your liver is mildly annoyed" or "your liver is drafting a resignation letter"? And is there actually a way to un-annoy a liver?

---

## Scenario 7: Heart Palpitations in a 34-Year-Old

**Clinical core**: Patient (34, otherwise healthy, no cardiac history) has been experiencing episodes of heart palpitations — rapid fluttering sensation lasting 30 seconds to 2 minutes, happening 2–3 times per week for the past month. No chest pain, no fainting. ECG at GP was normal. GP said it's likely benign but ordered a 24-hour Holter monitor.

### 7A — Anxious / Catastrophising
For the past month I've been getting these episodes where my heart suddenly starts racing and fluttering. It lasts maybe 30 seconds to 2 minutes and happens 2–3 times a week. I'm 34 and I've never had heart problems but now I'm terrified every time it happens. What if it's something serious and they're missing it? My ECG was normal but I've read that ECGs can miss things. The doctor ordered a Holter monitor but what if it doesn't catch an episode? I keep checking my pulse. I'm scared to exercise in case it triggers something. Could this be a heart attack starting?

### 7B — Stoic / Minimal
I'm 34, no cardiac history. Having heart palpitations — rapid fluttering, 30 seconds to 2 minutes, 2–3 times per week for a month. No chest pain or syncope. ECG normal. Holter monitor ordered. What's the likely diagnosis?

### 7C — Angry / Frustrated
I've been having heart palpitations for a month — my heart races and flutters for up to 2 minutes, several times a week. I'm 34 and otherwise healthy. The GP did an ECG which was "normal" and said it's "probably nothing" and ordered a Holter monitor. Probably nothing? My heart is doing something abnormal multiple times a week and "probably nothing" is the best they've got? What should I actually be asking for here? Because I don't feel like this is being taken seriously.

### 7D — Hyper-rational / Information-seeking
I'm 34, no relevant medical history, BMI 23, non-smoker. I've been experiencing paroxysmal palpitations — rapid irregular fluttering, lasting 30 seconds to 2 minutes, occurring 2–3 times weekly for 4 weeks. No associated chest pain, dyspnoea, or syncope. Resting 12-lead ECG was unremarkable. A 24-hour Holter has been ordered. Given my age and symptom profile, what's the most likely aetiology — ectopics, SVT, or something else? What's the diagnostic yield of a 24-hour Holter versus a 7-day event monitor for intermittent symptoms at this frequency?

### 7E — Overwhelmed / Defeated
My heart keeps doing this thing where it suddenly races and flutters. It's been happening a few times a week for about a month. I'm only 34. The doctor said the ECG is normal and ordered some monitor thing I have to wear. I know I should feel reassured but I don't. Every time it happens I freeze and wait for something terrible. I don't know what I'm supposed to be doing about it. Is this something I should be worried about or am I just making it worse by stressing?

### 7F — Humor / Irony
My heart has apparently decided to take up freestyle drumming. Couple of times a week it just goes into this rapid fluttering thing for a minute or two, completely unprovoked. I'm 34 and otherwise healthy — no heart problems, no chest pain, nothing. ECG was normal, naturally, because my heart only performs its party trick when no one's watching. Got a Holter monitor coming up. Realistic expectations — am I dying, or is my heart just being dramatic?

---

## Scenario 8: Difficulty Conceiving After 12 Months

**Clinical core**: Couple (woman 33, male partner 35) has been trying to conceive for 12 months without success. Regular cycles, no known reproductive issues. GP has ordered initial investigations — hormone panel for her, semen analysis for partner. Results not yet back. Patient wants to understand the process and timeframe.

### 8A — Anxious / Catastrophising
We've been trying to have a baby for a year now and nothing has happened. I'm 33, my partner is 35. My cycles are regular, we've been timing everything, doing everything "right." The doctor has ordered blood tests for me and a semen analysis for my partner but the results aren't back yet and I'm already imagining the worst. What if there's something really wrong? I see pregnant people everywhere and I feel broken. Every month when my period comes I cry. How long do these investigations take? What if they don't find anything — is that even worse?

### 8B — Stoic / Minimal
33F, partner 35M. Trying to conceive for 12 months, no success. Regular cycles, no known reproductive issues. GP has ordered hormone panel and semen analysis, awaiting results. What's the typical investigation pathway from here?

### 8C — Angry / Frustrated
We've been trying to get pregnant for a year. A whole year. I'm 33, healthy, regular cycles. My partner is 35 and healthy too. The GP has finally ordered some tests — only after I had to push for them — and now we're waiting for results. It feels like no one takes this seriously until you've suffered long enough. What should I expect from these results? And if they come back "normal" and we're still not pregnant, what then? Because I'm not accepting "just keep trying" as an answer.

### 8D — Hyper-rational / Information-seeking
My partner and I have been attempting conception for 12 months — I'm 33 with regular 28-day cycles, he's 35 with no known fertility issues. Our GP has initiated a standard workup: day 2–5 FSH, LH, oestradiol, AMH, and progesterone for me; semen analysis for my partner. Results are pending. Could you outline the diagnostic algorithm from here? At what point would referral to a reproductive endocrinologist be indicated, and what's the evidence on interventions like clomiphene versus IUI versus proceeding directly to IVF based on different diagnostic findings?

### 8E — Overwhelmed / Defeated
We've been trying for a baby for a year. Nothing. I'm 33 and my partner is 35 and everyone around us seems to get pregnant without even trying. The doctor ordered some tests and we're waiting for results but I almost don't want to know. I'm exhausted from the hoping and the disappointment every single month. I don't really know what to ask you. I just want someone to tell me what happens next because I can't keep living in this limbo.

### 8F — Humor / Irony
Twelve months of carefully timed romance and precisely zero pregnancies to show for it. I'm 33, he's 35, and apparently we've graduated from "it'll happen when it happens" to "here, pee in this cup and we'll run some tests." GP has ordered the full investigation — hormones for me, semen analysis for him. So while we wait for the lab to judge our reproductive competence, what should we actually expect from this process? Roughly how many more awkward appointments are we looking at?

---

## Scenario 9: Post-Surgical Wound Concerns (Knee Arthroscopy)

**Clinical core**: Patient had knee arthroscopy 10 days ago. One of the small incision sites is red, slightly swollen, and warm to touch. No fever. Some clear-ish discharge. They were told some redness is normal but are unsure whether this has crossed into "see a doctor" territory.

### 9A — Anxious / Catastrophising
I had knee arthroscopy 10 days ago and one of the incision sites doesn't look right. It's red, a bit swollen, and warm when I touch it. There's some discharge — it's mostly clear but slightly yellowish. I don't have a fever but I'm panicking because I've read about surgical infections and sepsis and I'm terrified this is getting infected. They told me some redness is normal after surgery but this doesn't feel normal to me. Should I go to A&E? Could I lose my knee from an infection? Am I overreacting?

### 9B — Stoic / Minimal
10 days post knee arthroscopy. One incision site is red, slightly swollen, warm. Some clear-ish discharge. No fever. Is this normal healing or should I get it looked at?

### 9C — Angry / Frustrated
I had a knee arthroscopy 10 days ago. One of the incisions is red, swollen, warm, and there's discharge. The hospital's post-op instructions said "some redness is normal" and to "call if you're concerned" — so I called, and got put on hold for 40 minutes before giving up. I can't get through to anyone and I don't know whether I should be worried or not. It would be nice if the healthcare system could make it possible to actually ask someone a question after they've cut into your body. Is this infected? What should I be looking for?

### 9D — Hyper-rational / Information-seeking
I'm 10 days post arthroscopic knee surgery. One of the portal sites is erythematous, mildly oedematous, and warm, with a small amount of serous to serosanguinous discharge. No systemic signs — afebrile, no rigors, CRP not tested. I'm trying to distinguish between normal post-surgical inflammation and early superficial surgical site infection. What are the clinical criteria for SSI per CDC definitions, and at what point would empirical antibiotics be indicated versus watchful waiting?

### 9E — Overwhelmed / Defeated
I had knee surgery 10 days ago and one of the cuts looks weird. It's red and swollen and warm and there's some stuff coming out of it. They said some redness is normal but I don't know what "some" means versus "too much." I've tried calling the hospital but I can't get through. I don't know whether this is fine or whether I should be worried. I'm tired and I'm sore and I just want someone to tell me if this is okay or not.

### 9F — Humor / Irony
My knee is 10 days post-arthroscopy and one of the incisions seems to be staging a protest — red, swollen, warm, and producing a mystery discharge. The post-op leaflet cheerfully states "some redness is normal," which is about as helpful as "some turbulence is normal" when the plane is shaking. No fever, so that's something. Am I fine, or should I be heading back to the hospital before my knee files a formal complaint?

---

## Scenario 10: Persistent Headaches and Worry About Brain Tumour

**Clinical core**: Patient (29) has been having daily headaches for 3 weeks. Dull pressure, mainly frontal, worse in the afternoon. No visual changes, no vomiting, no neurological symptoms. Has a desk job and increased screen time recently. Takes paracetamol which helps somewhat. Worried it could be something serious.

### 10A — Anxious / Catastrophising
I've had a headache every day for the last 3 weeks. It's like a dull pressure across my forehead, gets worse in the afternoon. I'm 29. I keep reading about brain tumours and aneurysms and I'm scaring myself. I know it's probably tension or something but 3 weeks straight feels like too long to be normal. I don't have vision problems or vomiting but what if those come later? Paracetamol helps a bit but shouldn't I be getting a scan? How do you know when a headache is just a headache and when it's something actually dangerous?

### 10B — Stoic / Minimal
I'm 29. Daily headaches for 3 weeks. Dull frontal pressure, worse in afternoon. No visual symptoms or neurological signs. Paracetamol provides partial relief. Increased screen time recently. When would imaging be warranted?

### 10C — Angry / Frustrated
I've had headaches every day for 3 weeks. Every single day. I'm 29 and this has never happened before. I went to my GP and she said it's probably tension headaches, told me to take paracetamol and reduce my screen time. That's it. No scan, no referral, nothing. I'm sorry but three weeks of daily headaches isn't normal and "take a paracetamol" isn't a diagnosis. What would actually rule out something serious? Because apparently I have to be my own doctor here.

### 10D — Hyper-rational / Information-seeking
I'm 29 with a 3-week history of daily headaches — bilateral frontal pressure, worse in afternoon, partially responsive to paracetamol. No red flags: no papilloedema signs, no focal neurological deficits, no vomiting, no postural component. Significant increase in screen time due to work. I'm aware this fits the profile for chronic tension-type headache, but I'd like to understand the evidence-based criteria for neuroimaging referral. Specifically, what are the clinical decision rules — is the NICE guidance or the American Headache Society criteria more appropriate here?

### 10E — Overwhelmed / Defeated
I've had a headache every day for three weeks. It's always there — this pressure across my forehead that gets worse as the day goes on. I'm 29, I've never had headaches like this. The doctor says it's probably tension and to take paracetamol. And I do, and it helps a bit, but then it's back the next day. I don't have the energy to keep pushing for answers. I just want to know if I should be scared or if this is one of those things that eventually goes away.

### 10F — Humor / Irony
My head has decided to audition for the role of "pressure cooker" — three weeks of daily headaches, right across the forehead, getting worse every afternoon like clockwork. I'm 29, no history of this, and my GP's diagnosis was essentially "you stare at screens too much" which is fair but also not exactly what I wanted to hear after 21 consecutive days of head pain. Paracetamol takes the edge off but I'd love to know: at what point do I upgrade from "take a painkiller" to "maybe get a brain scan"?
