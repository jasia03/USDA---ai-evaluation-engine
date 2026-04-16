# USDA AI Evaluation Engine

A modular, end-to-end evaluation framework for measuring whether a deployed USDA helpdesk chatbot delivers genuine user and business value and not just technical performance. Built as part of the George Mason University Challenge X project in collaboration with the USDA.

This framework answers the question : **"Does this AI product help our employees?"** by combining structural metrics, north star metrics, LLM-as-judge semantic scoring, drift detection, and automated alignment reporting into a single repeatable pipeline.

---

## The Problem

Large language models behave probabilistically, not like traditional software. Single-run tests and static benchmarks cannot capture how a deployed chatbot performs across hundreds of real conversations over time. Without a structured evaluation engine, USDA teams have no reliable way to:

- Know whether the chatbot is actually resolving employee problems
- Identify which failure patterns are most common and why
- Detect when chatbot performance degrades over time
- Make data-driven decisions about where to invest improvement efforts
 
---

## The Solution

A nine-step evaluation pipeline that processes raw chatbot conversation logs and produces a full suite of metrics, a cross-layer correlation analysis, drift detection against a baseline, and a periodic alignment report — all from this single command.

```
python run_evaluation.py --data your_chatbot_data.xlsx
```

---

## Pipeline Architecture

```
Raw conversation data (.xlsx)
          │
          ▼
01_trace_parser.py          →  Structured trace objects (one per conversation)
          │
          ▼
02_data_health_check.py     →  Data quality validation (stability, coverage)
          │
          ▼
03_structural_metrics.py    →  9 deterministic metrics (length, latency, KBAs, etc.)
          │
          ▼
04_north_star_metrics.py    →  4 outcome metrics (resolution, escalation, deflection)
          │
          ▼
05_golden_set_selector.py   →  Stratified sample for human annotation
          │
     [Manual annotation]    →  Ideal responses + quality scores (1–5)
          │
          ▼
06_llm_judge.py             →  Semantic scoring via Claude API (accuracy, helpfulness, tone, brevity)
          │                 
          ▼
07_unified_merger.py        →  Master metrics table (all layers combined)
          │
          ▼
08_analysis_and_drift.py    →  Cross-layer correlations + drift detection
          │
          ▼
09_alignment_report.py      →  Periodic alignment report (primary deliverable)
```
Additional analysis scripts (run independently after the core pipeline):
10_failure_classifier.py    →  Automatic failure pattern labeling (all 588 conversations)
11_rouge_scores.py          →  ROUGE similarity scores vs ideal responses (golden set)
12_latency_and_deflection.py→  Latency per turn + explicit deflection rate
---

## Metrics

### Structural metrics (all 588 conversations)
Deterministic signals computed directly from the conversation data.

| Metric | Description |
|---|---|
| Response length | Average words per AI response |
| Conversation length | Total message turns |
| Human turns | Number of times the user sent a message |
| Response latency | Seconds between user message and AI reply |
| KBAs retrieved | Unique knowledge base documents used |
| IO ratio | Total AI words / total user words |
| Session duration | Seconds from first to last message |
| Restart rate | Whether the user restarted the conversation |
| Vocabulary complexity | Flesch-Kincaid grade level of AI responses |

### North star metrics (all 588 conversations)
Outcome signals that measure whether users are succeeding.

| Metric | Description |
|---|---|
| Resolution rate | % of conversations fully resolved by chatbot |
| Escalation rate | % of conversations handed to a human agent |
| Self-containment rate | % of conversations handled without escalation |
| Deflection value | Weighted estimate of human effort saved (0–1) |

### Semantic metrics (37 annotated conversations)
LLM-as-judge scoring calibrated against human expert annotations.

| Dimension | Description |
|---|---|
| Accuracy | Was the information correct and grounded in fact? |
| Helpfulness | Did the response address what the user actually needed? |
| Tone | Was it professional and appropriate for a government helpdesk? |
| Brevity | Was the length appropriate for the complexity of the question? |

---

## Key Findings

Analysis of 588 real conversations from the USDA helpdesk chatbot (January–February 2026):

**Resolution rate: 17.0%** - only 1 in 6 conversations was fully resolved by the chatbot without human intervention.

**Abandonment rate: 43.0%** - the most common outcome was users giving up entirely.

**The 63-point self-containment gap** - the chatbot handles 80.1% of conversations without escalating but resolves only 17.0%. This gap represents conversations the chatbot attempted but failed to resolve, where users abandoned or restarted rather than getting help.

**Zero-KBA conversations resolve at 0.0%** - when the knowledge base retrieval system returns no documents, not a single conversation in the dataset ended in resolution. 51 conversations had zero KBAs retrieved.

**Vocabulary complexity predicts failure** - resolved conversations average a Flesch-Kincaid grade of 5.8 (plain English), while abandoned conversations average grade 8.6 (high school complexity). This is the strongest structural predictor of resolution outcome (r = -0.324).

**LLM judge composite score: 3.21 / 5.00** - tone scores highest (3.86) while helpfulness scores lowest (2.89), meaning the chatbot sounds professional but often fails to actually help.

**Category performance spread:**
- Best: LincPass Issues - 47.8% resolution rate
- Worst: Software Issue - 14.5% resolution rate, 61.3% escalation rate

**Monthly improvement** - resolution rate improved from 13.7% in January to 20.9% in February, a 7.2 percentage point gain.

**False resolution flags detected** - some conversations are marked resolved by the system without user confirmation, meaning the 17.0% figure may be slightly inflated.

**ROUGE-L F1 scores validate LLM judge findings** — resolved conversations average ROUGE-L F1 of 0.68 against ideal responses, while escalated conversations average only 0.18. This classical NLP measurement independently confirms the semantic scoring results.

**First-turn latency averages 11.9 seconds** — significantly higher than the overall average of 5.8 seconds. Latency decreases as conversations progress, averaging 4.1 seconds by turn 3, suggesting the system works harder on initial knowledge retrieval.

**Deflection rate: 9.2% explicit, 8.7% silent** — 9.2% of conversations had the chatbot explicitly acknowledge it could not help. An additional 8.7% were silent stalls where the chatbot produced no response at all. Silent stalls are more harmful because users wait without knowing help is unavailable.
---

## Failure Taxonomy

Five distinct failure patterns identified through manual annotation of 37 conversations:

1. **Zero KBA retrieved** — the knowledge base had no relevant content; the chatbot produced no response and users left immediately
2. **Wrong KBA retrieved** — the chatbot confidently answered using an irrelevant document (e.g., voicemail instructions given for an SMTP relay question)
3. **Right KBA, wrong section** — the correct document was found but the wrong part was used (e.g., document-signing steps given for a certificate update question)
4. **Right KBA, user context ignored** — relevant documents were retrieved but the chatbot ignored user-provided details and gave a generic response
5. **False resolution flagging** — conversations marked resolved by the system despite users abandoning

---

## Recommendations

**1. Implement zero-KBA fallback**
When knowledge base retrieval returns no documents, the chatbot should immediately acknowledge the gap and offer escalation rather than making the user wait indefinitely.

**2. Mandatory clarifying question for short queries**
Queries under 6 words should trigger a clarifying question before any knowledge base search is attempted. This prevents the confident-wrong-answer failure pattern that appeared across multiple annotated conversations.

**3. Agency-aware document routing**
KBA documents should be tagged by USDA agency. Forest Service employees should not receive ERS-specific instructions. This was directly observed in annotation where a Forest Service user received instructions referencing an ERS Info Link folder that does not exist on their desktop.

**4. Fix resolution detection mechanism**
Resolution should require explicit user confirmation rather than being triggered automatically by a system event. Current false positives inflate the reported resolution rate.

**5. Reduce vocabulary complexity**
Target a Flesch-Kincaid grade of 6–8 for all responses. Plain language responses resolve at nearly 3× the rate of complex responses.

---

## Setup

### Requirements

Python 3.10 or higher.

```bash
pip install -r requirements.txt
```

Dependencies: `pandas`, `openpyxl`

No other external libraries are required. The Flesch-Kincaid vocabulary complexity metric uses a pure Python implementation — no NLTK download needed.

### API key (LLM judge only)

The LLM judge requires an Anthropic API key. Get one at [console.anthropic.com](https://console.anthropic.com). Scoring 37 conversations costs under $1 using `claude-sonnet-4-6`.

```bash
# Windows
set ANTHROPIC_API_KEY=your_key_here

# Mac / Linux
export ANTHROPIC_API_KEY=your_key_here
```

---

## Running the Engine

### Full pipeline (recommended)

```bash
python run_evaluation.py --data messages_data_student_extract.xlsx
```

### Skip LLM judge (if no API key)

```bash
python run_evaluation.py --data messages_data_student_extract.xlsx --skip-llm
```

### With custom baseline for drift comparison

```bash
python run_evaluation.py --data new_data.xlsx --baseline data/baseline_metrics.json
```

### Run individual steps

```bash
python 01_trace_parser.py
python 02_data_health_check.py
python 03_structural_metrics.py
python 04_north_star_metrics.py
python 05_golden_set_selector.py
# Complete annotation of golden_set_for_annotation.txt manually
python 06_llm_judge.py
python 07_unified_merger.py
python 08_analysis_and_drift.py
python 09_alignment_report.py
# Additional analysis (run after core pipeline)
python 10_failure_classifier.py
python 11_rouge_scores.py
python 12_latency_and_deflection.py
```

---

## Output Files

All outputs are saved to the `data/` folder:

| File | Description |
|---|---|
| `traces.json` | 588 structured conversation objects |
| `health_check_report.txt` | Data quality validation results |
| `structural_metrics.csv` | 9 structural metrics per conversation |
| `structural_summary.txt` | Structural metrics summary with breakdowns |
| `north_star_metrics.csv` | 4 north star metrics per conversation |
| `north_star_summary.txt` | North star summary with category breakdowns |
| `golden_set_ids.json` | Thread IDs selected for annotation |
| `golden_set_for_annotation.txt` | Formatted file for human annotation |
| `semantic_scores.csv` | LLM judge scores for 37 annotated conversations |
| `master_metrics.csv` | All metrics combined in one table |
| `correlation_analysis.txt` | Cross-layer correlation findings |
| `drift_report.txt` | Month-over-month drift detection results |
| `baseline_metrics.json` | Baseline values for future drift comparison |
| `alignment_report.txt` | **Primary deliverable — periodic alignment report** |

---

## Adapting for Other USDA AI Products

This engine was designed to be modular and reusable. To evaluate a different USDA AI application:

1. Ensure your conversation data includes: a thread/conversation ID, message content, message role (user/AI), termination type, and timestamps
2. Update the column name mapping in `01_trace_parser.py` to match your data schema
3. Update the category labels in `03_structural_metrics.py` and `04_north_star_metrics.py` if your application uses different issue categories
4. Run `run_evaluation.py` pointing at your new data file
5. The baseline from a previous run can be used for drift comparison with `--baseline`

---

## Data Privacy

Raw conversation data and pipeline outputs are not included in this repository. The dataset contains real USDA employee helpdesk interactions and is kept private. The `data/` folder and annotation file are excluded via `.gitignore`.

To run this pipeline you will need access to the original conversation dataset.

---

## Project Context

Built for the George Mason University Challenge X - a project-based learning initiative in collaboration with federal government partners. The challenge brief asked for a modular AI evaluation framework for a deployed USDA helpdesk chatbot that could validate business impact, ensure user success, and provide actionable insights for leadership.

**Team:** Infinite Loop 
**Member:** Jasia Sanjana
**Partner:** United States Department of Agriculture (USDA)  
**Data period:** January 2, 2026 – February 27, 2026  
**Conversations analyzed:** 588
