"""
USDA AI Evaluation Engine
Step 5: LLM Judge (Semantic Metrics)

Uses Claude as a judge to score chatbot responses across 4 semantic
dimensions for each conversation in the golden set.

Dimensions scored (each 1-5):
  - accuracy    : Was the information correct and grounded in fact?
  - helpfulness : Did it actually address what the user needed?
  - tone        : Was it professional and appropriate for a government helpdesk?
  - brevity     : Was the length appropriate — not too long, not too short?

For golden set conversations, the judge also receives the human-annotated
ideal response and quality score as calibration context.

Outputs:
  data/semantic_scores.csv      — one row per conversation with all 4 scores
  data/semantic_summary.txt     — summary with breakdowns and key findings
"""

import json
import csv
import re
import time
import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
# Update ANNOTATION_FILE to point to wherever you saved the annotation file
ANNOTATION_FILE = Path("golden_set_for_annotation.txt")
GOLDEN_IDS_FILE = Path("data/golden_set_ids.json")
TRACES_FILE = Path("data/traces.json")
SCORES_CSV = Path("data/semantic_scores.csv")
SUMMARY_FILE = Path("data/semantic_summary.txt")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 500
RATE_LIMIT_SECS = 1.0    # pause between API calls to avoid rate limiting


# ── Annotation parser ─────────────────────────────────────────────────────────
def parse_annotations(filepath: Path) -> dict:
    """
    Parse the human-annotated golden set file.
    Returns a dict keyed by thread_id with ideal_response, quality_score, notes.
    """
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    # Split into per-conversation blocks
    blocks = re.split(
        r"={6,}\s*\nCONVERSATION \d+ of \d+\s*\n={6,}",
        content
    )

    annotations = {}

    for block in blocks:
        # Must contain a thread ID to be a real conversation block
        thread_match = re.search(r"Thread ID\s*:\s*(\S+)", block)
        if not thread_match:
            continue

        thread_id = thread_match.group(1)

        # Quality score — handles formats: 4, _4_, _4__
        score_match = re.search(r"QUALITY_SCORE:\s*_?(\d)_?", block)
        quality_score = int(score_match.group(1)) if score_match else None

        # All >>> <<< blocks — first is ideal response, second is notes
        all_blocks = re.findall(r">>>(.*?)<<<", block, re.DOTALL)
        ideal_response = all_blocks[0].strip() if len(all_blocks) > 0 else ""
        notes = all_blocks[1].strip() if len(all_blocks) > 1 else ""

        # Clean up the [Write 2-4 sentences...] placeholder if not filled in
        if "[Write 2-4 sentences" in ideal_response:
            ideal_response = ""

        annotations[thread_id] = {
            "ideal_response": ideal_response,
            "quality_score":  quality_score,
            "notes":          notes,
        }

    return annotations


# ── Conversation formatter ────────────────────────────────────────────────────
def format_conversation(trace: dict, max_turns: int = 8) -> str:
    """
    Format a trace into a clean conversation string for the judge.
    Only includes human and real AI text messages — no debug or button UI.
    Caps at max_turns to keep prompt size reasonable.
    """
    lines = []
    turn_count = 0

    for msg in trace["messages"]:
        if msg["role"] not in ["human", "ai"]:
            continue
        content = msg["content"].strip()
        # Skip button UI messages
        if content.startswith("{") and "buttons" in content:
            continue
        if len(content.split()) < 2:
            continue

        role_label = "USER" if msg["role"] == "human" else "CHATBOT"
        lines.append(f"[{role_label}]: {content}")
        turn_count += 1
        if turn_count >= max_turns:
            lines.append("[... conversation continues ...]")
            break

    return "\n\n".join(lines)


# ── Judge prompt builder ──────────────────────────────────────────────────────
def build_judge_prompt(
    trace: dict,
    conversation_text: str,
    annotation: dict | None,
) -> str:
    """
    Build the scoring prompt for the LLM judge.

    If annotation is provided (golden set), includes the human ideal response
    and quality score as calibration context.
    """
    category = trace.get("template_category") or "Unknown"
    outcome = trace.get("termination_type", "unknown")
    kba_count = len(trace.get("kba_docs_retrieved", []))

    context_section = f"""You are evaluating a USDA government IT helpdesk chatbot.

Conversation context:
- Issue category: {category}
- Conversation outcome: {outcome}
- Knowledge base documents retrieved: {kba_count}"""

    if annotation and annotation.get("ideal_response"):
        context_section += f"""

Human expert annotation:
- Ideal response: {annotation['ideal_response']}
- Human quality score: {annotation['quality_score']}/5"""
        if annotation.get("notes"):
            context_section += f"""
- Reviewer notes: {annotation['notes'][:300]}"""

    prompt = f"""{context_section}

Conversation to evaluate:
{conversation_text}

Score the CHATBOT's FIRST substantive response on these 4 dimensions.
If the chatbot gave no response or only a button UI response, score all dimensions 1.

Scoring scale:
5 = Excellent  — performs this dimension perfectly
4 = Good       — mostly correct with minor gaps
3 = Adequate   — partial, missing something important
2 = Poor       — largely fails this dimension
1 = Failing    — completely wrong or absent

Dimensions:
- accuracy   : Was the information correct and grounded in fact? Did it address the right problem?
- helpfulness: Did the response actually help the user move forward toward a solution?
- tone       : Was it professional, clear, and appropriate for a government IT helpdesk?
- brevity    : Was the length appropriate — not overwhelming or too sparse for the complexity?

Respond ONLY with a JSON object in this exact format, no other text:
{{
  "accuracy": <1-5>,
  "helpfulness": <1-5>,
  "tone": <1-5>,
  "brevity": <1-5>,
  "reasoning": "<one sentence explaining the most important factor in your scores>"
}}"""

    return prompt


# ── API call ──────────────────────────────────────────────────────────────────
def call_judge(prompt: str) -> dict | None:
    """
    Call Claude API and parse the JSON response.
    Returns dict with scores or None if parsing fails.
    """
    import urllib.request

    payload = json.dumps({
        "model":      MODEL,
        "max_tokens": MAX_TOKENS,
        "messages":   [{"role": "user", "content": prompt}],
    }).encode("utf-8")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        print("Set it with: set ANTHROPIC_API_KEY=your_key_here  (Windows)")
        print("         or: export ANTHROPIC_API_KEY=your_key_here  (Mac/Linux)")
        return None

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type":      "application/json",
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        # Extract text content
        text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")

        # Parse JSON — strip markdown code fences if present
        text = re.sub(r"```(?:json)?", "", text).strip()
        scores = json.loads(text)

        # Validate all required fields present
        required = ["accuracy", "helpfulness", "tone", "brevity"]
        for field in required:
            if field not in scores:
                return None
            scores[field] = max(1, min(5, int(scores[field])))

        return scores

    except Exception as e:
        print(f"    API error: {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("USDA AI Evaluation Engine — LLM Judge (Semantic Metrics)")
    print("=" * 60)

    # Load data
    with open(TRACES_FILE, encoding="utf-8") as f:
        all_traces = json.load(f)
    traces_by_id = {t["thread_id"]: t for t in all_traces}

    with open(GOLDEN_IDS_FILE, encoding="utf-8") as f:
        golden_ids = json.load(f)

    annotations = parse_annotations(ANNOTATION_FILE)

    print(f"\nGolden set: {len(golden_ids)} threads")
    print(f"Annotations parsed: {len(annotations)}")
    print(f"Model: {MODEL}")

    # Validate annotations parsed correctly
    missing = [tid for tid in golden_ids if tid not in annotations]
    if missing:
        print(
            f"\nWARNING: {len(missing)} threads in golden set have no annotation")
        for m in missing:
            print(f"  {m}")

    # Score each conversation
    print(f"\nScoring {len(golden_ids)} conversations...")
    print("(This will take a few minutes)\n")

    rows = []
    failed = []

    for i, thread_id in enumerate(golden_ids, 1):
        trace = traces_by_id.get(thread_id)
        if not trace:
            print(
                f"  [{i:02d}/{len(golden_ids)}] SKIP — trace not found: {thread_id[:8]}")
            continue

        annotation = annotations.get(thread_id)
        conversation_text = format_conversation(trace)
        prompt = build_judge_prompt(trace, conversation_text, annotation)

        print(
            f"  [{i:02d}/{len(golden_ids)}] Scoring {thread_id[:8]}... ", end="", flush=True)

        scores = call_judge(prompt)

        if scores:
            # Composite score — average of all 4 dimensions
            composite = round(
                (scores["accuracy"] + scores["helpfulness"] +
                 scores["tone"] + scores["brevity"]) / 4, 2
            )

            row = {
                "thread_id":        thread_id,
                "termination_type": trace["termination_type"],
                "template_category": trace.get("template_category") or "Unknown",
                "human_score":      annotation["quality_score"] if annotation else None,
                "llm_accuracy":     scores["accuracy"],
                "llm_helpfulness":  scores["helpfulness"],
                "llm_tone":         scores["tone"],
                "llm_brevity":      scores["brevity"],
                "llm_composite":    composite,
                "llm_reasoning":    scores.get("reasoning", ""),
            }
            rows.append(row)
            print(f"acc={scores['accuracy']} help={scores['helpfulness']} "
                  f"tone={scores['tone']} brev={scores['brevity']} → {composite}")
        else:
            print("FAILED — will retry once")
            failed.append((i, thread_id, trace, annotation))

        time.sleep(RATE_LIMIT_SECS)

    # Retry failed calls once
    if failed:
        print(f"\nRetrying {len(failed)} failed calls...")
        for i, thread_id, trace, annotation in failed:
            conversation_text = format_conversation(trace)
            prompt = build_judge_prompt(trace, conversation_text, annotation)
            print(f"  [{i:02d}] Retry {thread_id[:8]}... ", end="", flush=True)
            scores = call_judge(prompt)
            if scores:
                composite = round(
                    (scores["accuracy"] + scores["helpfulness"] +
                     scores["tone"] + scores["brevity"]) / 4, 2
                )
                row = {
                    "thread_id":        thread_id,
                    "termination_type": trace["termination_type"],
                    "template_category": trace.get("template_category") or "Unknown",
                    "human_score":      annotation["quality_score"] if annotation else None,
                    "llm_accuracy":     scores["accuracy"],
                    "llm_helpfulness":  scores["helpfulness"],
                    "llm_tone":         scores["tone"],
                    "llm_brevity":      scores["brevity"],
                    "llm_composite":    composite,
                    "llm_reasoning":    scores.get("reasoning", ""),
                }
                rows.append(row)
                print(f"OK → {composite}")
            else:
                print("FAILED again — skipping")
            time.sleep(RATE_LIMIT_SECS)

    # Save CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(SCORES_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved: {SCORES_CSV}")

    # Build summary
    summary = build_summary(rows)
    SUMMARY_FILE.write_text(summary, encoding="utf-8")
    print(f"Saved: {SUMMARY_FILE}")
    print()
    print(summary)

    return rows


# ── Summary builder ───────────────────────────────────────────────────────────
def build_summary(rows: list) -> str:
    from collections import defaultdict

    def avg(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    lines = []
    lines.append("=" * 65)
    lines.append("USDA AI Evaluation Engine — Semantic Metrics Summary")
    lines.append("=" * 65)
    lines.append(f"Conversations scored: {len(rows)}")
    lines.append("")

    # ── Overall averages ──────────────────────────────────────────────────────
    lines.append(
        "── Overall LLM judge scores ──────────────────────────────────")
    for dim in ["llm_accuracy", "llm_helpfulness", "llm_tone", "llm_brevity", "llm_composite"]:
        label = dim.replace("llm_", "").title()
        lines.append(f"  {label:<20} {avg([r[dim] for r in rows]):.2f} / 5.00")
    lines.append("")

    # ── Human vs LLM agreement ────────────────────────────────────────────────
    paired = [(r["human_score"], r["llm_composite"])
              for r in rows if r["human_score"] is not None]
    if paired:
        lines.append(
            "── Human score vs LLM composite agreement ────────────────────")
        lines.append(f"  Conversations with human scores : {len(paired)}")
        diffs = [abs(h - l) for h, l in paired]
        lines.append(
            f"  Average absolute difference    : {avg(diffs):.2f} points")
        within_1 = sum(1 for d in diffs if d <= 1.0)
        lines.append(f"  Within 1 point                 : {within_1}/{len(paired)} "
                     f"({within_1/len(paired)*100:.0f}%)")
        lines.append("")

    # ── By termination type ───────────────────────────────────────────────────
    by_term = defaultdict(list)
    for r in rows:
        by_term[r["termination_type"]].append(r)

    lines.append(
        "── By termination type ───────────────────────────────────────")
    lines.append(f"  {'Type':<15} {'N':>4} {'Accuracy':>9} {'Helpful':>8} "
                 f"{'Tone':>6} {'Brevity':>8} {'Composite':>10}")
    lines.append("  " + "-" * 65)
    for term, group in sorted(by_term.items()):
        lines.append(
            f"  {term:<15} {len(group):>4} "
            f"{avg([r['llm_accuracy'] for r in group]):>9.2f} "
            f"{avg([r['llm_helpfulness'] for r in group]):>8.2f} "
            f"{avg([r['llm_tone'] for r in group]):>6.2f} "
            f"{avg([r['llm_brevity'] for r in group]):>8.2f} "
            f"{avg([r['llm_composite'] for r in group]):>10.2f}"
        )
    lines.append("")

    # ── By category ───────────────────────────────────────────────────────────
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["template_category"]].append(r)

    lines.append(
        "── By category ───────────────────────────────────────────────")
    lines.append(f"  {'Category':<45} {'N':>4} {'Composite':>10}")
    lines.append("  " + "-" * 63)
    for cat, group in sorted(by_cat.items(),
                             key=lambda x: avg([r["llm_composite"]
                                               for r in x[1]]),
                             reverse=True):
        lines.append(
            f"  {cat:<45} {len(group):>4} "
            f"{avg([r['llm_composite'] for r in group]):>10.2f}"
        )
    lines.append("")

    # ── Score distribution ────────────────────────────────────────────────────
    lines.append(
        "── Human score distribution ──────────────────────────────────")
    score_dist = defaultdict(int)
    for r in rows:
        if r["human_score"] is not None:
            score_dist[r["human_score"]] += 1
    for score in sorted(score_dist.keys()):
        bar = "█" * score_dist[score]
        lines.append(f"  {score} stars: {score_dist[score]:>3}  {bar}")
    lines.append("")

    # ── Lowest scoring conversations ──────────────────────────────────────────
    lines.append(
        "── Lowest scoring conversations (LLM composite) ──────────────")
    worst = sorted(rows, key=lambda r: r["llm_composite"])[:5]
    for r in worst:
        lines.append(
            f"  {r['thread_id'][:8]}  "
            f"{r['termination_type']:<12} "
            f"{r['template_category'][:30]:<30}  "
            f"composite={r['llm_composite']}"
        )
    lines.append("")

    # ── Highest scoring conversations ─────────────────────────────────────────
    lines.append(
        "── Highest scoring conversations (LLM composite) ─────────────")
    best = sorted(rows, key=lambda r: r["llm_composite"], reverse=True)[:5]
    for r in best:
        lines.append(
            f"  {r['thread_id'][:8]}  "
            f"{r['termination_type']:<12} "
            f"{r['template_category'][:30]:<30}  "
            f"composite={r['llm_composite']}"
        )
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
