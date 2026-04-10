"""
USDA AI Evaluation Engine
Step 4: Golden Set Selection

Selects 50 representative conversations from the 588 traces for manual
annotation. The selection is stratified — it ensures coverage across
termination types, categories, and conversation lengths so the golden
set reflects the full range of chatbot behavior.

After running this script:
  1. Open data/golden_set_for_annotation.txt
  2. For each conversation, read the user's question and the chatbot's response
  3. Write your ideal response in the IDEAL_RESPONSE field
  4. Write your quality score (1-5) in the QUALITY_SCORE field
  5. Save the file — the LLM judge will use it in the semantic phase

Scoring guide:
  5 = Excellent  — accurate, helpful, clear, appropriately concise
  4 = Good       — mostly correct with minor gaps or slight verbosity
  3 = Adequate   — answered partially but missed key information
  2 = Poor       — largely unhelpful, confusing, or off-topic
  1 = Failing    — wrong, harmful, or completely irrelevant

Output:
  data/golden_set_ids.json           — thread IDs selected for golden set
  data/golden_set_for_annotation.txt — human-readable file for manual annotation
"""

import json
import random
from pathlib import Path
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
TRACES_FILE  = Path("data/traces.json")
OUTPUT_DIR   = Path("data")
GOLDEN_IDS_FILE = Path("data/golden_set_ids.json")
ANNOTATION_FILE = Path("data/golden_set_for_annotation.txt")

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_SEED      = 42
TARGET_TOTAL     = 37   # 37 threads with known categories available
                        # (288 Unknown-category threads excluded —
                        #  almost all are abandoned with no useful metadata)

# How many threads to pick per termination type
# Weighted toward the most common types
TERMINATION_TARGETS = {
    "abandoned":  15,   # largest group — 43%
    "escalated":  13,   # 20%
    "restarted":  12,   # 20%
    "resolved":   10,   # 17% — smallest but most important
}

# Priority categories — ensure these are always represented
PRIORITY_CATEGORIES = [
    "Request for Information/assistance IT",
    "Software Issue",
    "Microsoft 365",
    "LincPass Issues",
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_first_human_message(trace: dict) -> str:
    """Return the first human message content in a trace."""
    for msg in trace["messages"]:
        if msg["role"] == "human":
            return msg["content"]
    return ""


def get_real_ai_responses(trace: dict) -> list[str]:
    """
    Return list of real AI text responses (not button UI messages).
    Filters out responses that are clearly system UI elements.
    """
    responses = []
    for msg in trace["messages"]:
        if msg["role"] != "ai":
            continue
        content = msg["content"].strip()
        if content.startswith("{") and "buttons" in content:
            continue
        if len(content.split()) < 5:
            continue
        responses.append(content)
    return responses


def format_conversation_for_annotation(trace: dict, index: int) -> str:
    """
    Format a single trace into a readable block for manual annotation.
    Shows the full human/AI exchange clearly.
    """
    lines = []
    lines.append("=" * 70)
    lines.append(f"CONVERSATION {index:02d} of {TARGET_TOTAL}")
    lines.append("=" * 70)
    lines.append(f"Thread ID  : {trace['thread_id']}")
    lines.append(f"Category   : {trace['template_category'] or 'Unknown'}")
    lines.append(f"Outcome    : {trace['termination_type'].upper()}")
    lines.append(f"Date       : {trace['conversation_date']}")
    lines.append(f"KBAs used  : {len(trace['kba_docs_retrieved'])}")
    lines.append("")

    # Show the conversation — human and AI turns only
    lines.append("── Conversation ─────────────────────────────────────────────")
    turn_num = 0
    for msg in trace["messages"]:
        if msg["role"] not in ["human", "ai"]:
            continue
        content = msg["content"].strip()
        # Skip button UI messages
        if content.startswith("{") and "buttons" in content:
            continue
        if len(content.split()) < 2:
            continue

        turn_num += 1
        role_label = "USER" if msg["role"] == "human" else "CHATBOT"
        lines.append(f"\n  [{role_label}]")
        # Wrap long content at 65 chars for readability
        words = content.split()
        line_buf = "  "
        for word in words:
            if len(line_buf) + len(word) + 1 > 67:
                lines.append(line_buf)
                line_buf = "  " + word
            else:
                line_buf += (" " if line_buf.strip() else "") + word
        if line_buf.strip():
            lines.append(line_buf)

    lines.append("")
    lines.append("── Your annotation ──────────────────────────────────────────")
    lines.append("")
    lines.append("  IDEAL_RESPONSE:")
    lines.append("  [Write 2-4 sentences describing the ideal chatbot answer")
    lines.append("   to the user's first question. Focus on: correctness,")
    lines.append("   clarity, actionable steps, and appropriate length.]")
    lines.append("")
    lines.append("  >>>")
    lines.append("")
    lines.append("")
    lines.append("")
    lines.append("  <<<")
    lines.append("")
    lines.append("  QUALITY_SCORE: ___  (1=Failing, 2=Poor, 3=Adequate, 4=Good, 5=Excellent)")
    lines.append("")
    lines.append("  NOTES (optional):")
    lines.append("  >>>")
    lines.append("")
    lines.append("  <<<")
    lines.append("")

    return "\n".join(lines)


# ── Selection strategy ────────────────────────────────────────────────────────
def select_golden_set(traces: list[dict]) -> list[dict]:
    """
    Stratified selection of 50 threads.

    Strategy:
    1. Group by termination type
    2. Within each group, prioritize threads from priority categories
    3. Fill remaining slots with random selection from the group
    4. Ensure no 'Unknown' category threads are selected
       (they have no template_category and are less useful to annotate)
    """
    random.seed(RANDOM_SEED)

    # Exclude Unknown category — not useful for annotation
    eligible = [t for t in traces if t["template_category"] is not None]

    # Group by termination type
    by_term = defaultdict(list)
    for t in eligible:
        by_term[t["termination_type"]].append(t)

    selected = []

    for term_type, target_n in TERMINATION_TARGETS.items():
        pool = by_term[term_type]
        if not pool:
            continue

        # Split pool into priority category threads and the rest
        priority = [t for t in pool if t["template_category"] in PRIORITY_CATEGORIES]
        other    = [t for t in pool if t["template_category"] not in PRIORITY_CATEGORIES]

        random.shuffle(priority)
        random.shuffle(other)

        # Take from priority first, then fill from other (all categories welcome)
        picks = priority[:target_n]
        remaining_slots = target_n - len(picks)
        if remaining_slots > 0:
            picks += other[:remaining_slots]

        # If still short (small termination group), take all available
        selected.extend(picks if len(picks) >= target_n else pool[:target_n])

    # Sort by category then termination for cleaner annotation experience
    selected.sort(key=lambda t: (
        t["template_category"] or "",
        t["termination_type"]
    ))

    return selected


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("USDA AI Evaluation Engine — Golden Set Selection")
    print("=" * 60)

    # Load traces
    with open(TRACES_FILE, encoding="utf-8") as f:
        traces = json.load(f)
    print(f"\nLoaded {len(traces)} traces")

    # Select golden set
    golden = select_golden_set(traces)
    print(f"Selected {len(golden)} threads for golden set")

    # Print selection breakdown
    print("\n── Selection breakdown ───────────────────────────────")
    by_term = defaultdict(int)
    by_cat  = defaultdict(int)
    for t in golden:
        by_term[t["termination_type"]] += 1
        by_cat[t["template_category"]] += 1

    print("  By termination type:")
    for term, count in sorted(by_term.items()):
        print(f"    {term:<15} {count}")

    print("  By category:")
    for cat, count in sorted(by_cat.items()):
        print(f"    {cat:<45} {count}")

    # Save thread IDs
    golden_ids = [t["thread_id"] for t in golden]
    with open(GOLDEN_IDS_FILE, "w", encoding="utf-8") as f:
        json.dump(golden_ids, f, indent=2)
    print(f"\nSaved thread IDs: {GOLDEN_IDS_FILE}")

    # Build annotation file
    print("Building annotation file...")
    blocks = []

    # Header instructions
    header = "\n".join([
        "=" * 70,
        "USDA AI Evaluation Engine — Golden Set Annotation File",
        "=" * 70,
        "",
        "INSTRUCTIONS:",
        "  For each conversation below:",
        "",
        "  1. Read the full USER / CHATBOT exchange carefully",
        "  2. In the IDEAL_RESPONSE section, write 2-4 sentences describing",
        "     what a perfect helpdesk response would look like for the user's",
        "     first question. You do not need to be a USDA IT expert —",
        "     focus on whether the response is clear, accurate, actionable,",
        "     and appropriately concise.",
        "  3. In QUALITY_SCORE, write a single number 1-5:",
        "       5 = Excellent  — accurate, helpful, clear, concise",
        "       4 = Good       — mostly correct, minor gaps",
        "       3 = Adequate   — partial answer, missing key info",
        "       2 = Poor       — largely unhelpful or confusing",
        "       1 = Failing    — wrong, harmful, or completely off-topic",
        "  4. NOTES is optional — use it to flag anything unusual",
        "",
        f"  Total conversations to annotate: {len(golden)}",
        "  Estimated time: 3-4 hours",
        "",
        "  Save this file when done. The LLM judge will read it.",
        "",
    ])
    blocks.append(header)

    for i, trace in enumerate(golden, 1):
        blocks.append(format_conversation_for_annotation(trace, i))

    full_text = "\n".join(blocks)
    ANNOTATION_FILE.write_text(full_text, encoding="utf-8")
    print(f"Saved annotation file: {ANNOTATION_FILE}")
    print(f"\nNext step: open {ANNOTATION_FILE.name} and complete the annotations manually.")
    print("Estimated time: 3-4 hours")

    return golden


if __name__ == "__main__":
    main()
