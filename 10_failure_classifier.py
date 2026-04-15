"""
USDA AI Evaluation Engine
Step 10: Failure Pattern Classifier

Automatically labels all 588 conversations with one of six classes
based on rules derived from the structural metrics, semantic scores,
and conversation content.

Failure patterns (from manual annotation of golden set):
  0 - SUCCESS             — conversation resolved successfully
  1 - ZERO_KBA            — no documents retrieved; chatbot stalled
  2 - WRONG_KBA           — wrong documents retrieved; wrong answer given
  3 - WRONG_KBA_SECTION   — right document, wrong section used
  4 - CONTEXT_IGNORED     — right docs, but user context not used
  5 - FALSE_RESOLUTION    — marked resolved without user confirmation
  6 - ESCALATED_CLEAN     — correct escalation (admin task, out of scope)
  7 - UNKNOWN             — insufficient signals to classify

Classification rules:
  - SUCCESS       : termination=resolved AND (llm_accuracy>=4 OR no LLM score)
  - ZERO_KBA      : kbas_retrieved=0 AND termination!=resolved
  - FALSE_RESOL.  : termination=resolved AND llm_accuracy<=2 (scored conversations)
  - WRONG_KBA     : kbas_retrieved>0 AND llm_accuracy=1 AND avg_response_length>20
  - CONTEXT_IGN.  : kbas_retrieved>0 AND llm_accuracy=2 AND human_turns>=2
  - WRONG_SECTION : kbas_retrieved>0 AND llm_accuracy=2 AND human_turns<2
  - CLEAN_ESC.    : termination=escalated AND kbas_retrieved=0 AND human_turns>=2
  - UNKNOWN       : anything not caught by above rules

For the 551 non-golden-set conversations, rules use only structural
signals (no LLM scores). For the 37 golden set conversations,
rules also use semantic signals for higher precision.

Outputs:
  data/failure_patterns.csv     — one row per trace with pattern label
  data/failure_pattern_summary.txt — distribution and breakdown
"""

import json
import csv
import re
import pandas as pd
from pathlib import Path
from collections import Counter

# ── Paths ─────────────────────────────────────────────────────────────────────
MASTER_FILE   = Path("data/master_metrics.csv")
TRACES_FILE   = Path("data/traces.json")
PATTERNS_CSV  = Path("data/failure_patterns.csv")
SUMMARY_FILE  = Path("data/failure_pattern_summary.txt")

# ── Pattern labels ────────────────────────────────────────────────────────────
PATTERNS = {
    0: "SUCCESS",
    1: "ZERO_KBA_FAILURE",
    2: "WRONG_KBA_RETRIEVED",
    3: "WRONG_KBA_SECTION",
    4: "CONTEXT_IGNORED",
    5: "FALSE_RESOLUTION",
    6: "CLEAN_ESCALATION",
    7: "UNKNOWN",
}

PATTERN_DESCRIPTIONS = {
    0: "Resolved successfully — chatbot answered correctly",
    1: "Zero KBA retrieved — chatbot stalled with no knowledge base content",
    2: "Wrong KBA retrieved — chatbot answered confidently using wrong documents",
    3: "Right KBA, wrong section — correct document but wrong part used",
    4: "User context ignored — relevant docs retrieved but user details not used",
    5: "False resolution — marked resolved without genuine user confirmation",
    6: "Clean escalation — correctly routed admin/out-of-scope task to human",
    7: "Unknown — insufficient signals to classify with confidence",
}


# ── Content-based signals ─────────────────────────────────────────────────────
def get_first_ai_response(trace: dict) -> str:
    """Return the first substantive AI text response in a trace."""
    for msg in trace["messages"]:
        if msg["role"] != "ai":
            continue
        content = msg["content"].strip()
        if content.startswith("{") and "buttons" in content:
            continue
        if len(content.split()) >= 10:
            return content.lower()
    return ""


def is_deflection_response(ai_text: str) -> bool:
    """Check if the AI acknowledged it could not help."""
    phrases = [
        "not contain", "do not contain", "does not contain",
        "not available in", "no information", "unable to find",
        "cannot find", "not found in", "not provided in",
        "consult additional", "consult relevant",
        "please consult", "contact.*support", "contact.*helpdesk",
    ]
    return any(re.search(p, ai_text) for p in phrases)


def has_repeated_user_message(trace: dict) -> bool:
    """Check if user sent the same message twice — signal of frustration."""
    human_msgs = [
        m["content"].strip().lower()
        for m in trace["messages"]
        if m["role"] == "human"
    ]
    return len(human_msgs) != len(set(human_msgs)) and len(human_msgs) >= 2


def user_explicitly_corrected(trace: dict) -> bool:
    """
    Check if user corrected the chatbot's assumption in a follow-up.
    Signal that user context was ignored in the first response.
    """
    human_msgs = [
        m["content"].strip().lower()
        for m in trace["messages"]
        if m["role"] == "human"
    ]
    if len(human_msgs) < 2:
        return False
    # Look for correction signals in second or later human messages
    correction_phrases = [
        "not ers", "not forest", "not that", "that's not",
        "that is not", "wrong", "incorrect", "i already",
        "i have already", "already tried", "already restarted",
        "i said", "as i mentioned", "i meant",
    ]
    for msg in human_msgs[1:]:
        if any(p in msg for p in correction_phrases):
            return True
    return False


# ── Main classifier ───────────────────────────────────────────────────────────
def classify_trace(row: pd.Series, trace: dict) -> tuple[int, str]:
    """
    Classify one trace into a failure pattern.
    Returns (pattern_id, confidence: 'high'|'medium'|'low')
    """
    term         = row["termination_type"]
    kbas         = row["kbas_retrieved"]
    human_turns  = row["human_turns"]
    resp_length  = row["avg_response_length"]
    has_semantic = pd.notna(row.get("llm_accuracy"))

    # Get semantic signals if available
    llm_acc  = float(row["llm_accuracy"])  if has_semantic else None
    llm_help = float(row["llm_helpfulness"]) if has_semantic else None

    # Get content signals
    first_ai   = get_first_ai_response(trace)
    repeated   = has_repeated_user_message(trace)
    corrected  = user_explicitly_corrected(trace)
    deflection = is_deflection_response(first_ai)

    # ── Rule 1: SUCCESS ───────────────────────────────────────────────────────
    if term == "resolved":
        if has_semantic and llm_acc <= 2:
            return 5, "high"   # Resolved but LLM says response was wrong → false resolution
        return 0, "high"

    # ── Rule 2: ZERO KBA FAILURE ──────────────────────────────────────────────
    if kbas == 0 and term != "resolved":
        # Sub-case: was this a clean escalation (admin task, user knew it needed IT)?
        if term == "escalated" and human_turns >= 2 and resp_length == 0:
            return 6, "medium"
        return 1, "high"

    # ── Rule 3: FALSE RESOLUTION (semantic signal only) ───────────────────────
    if has_semantic and term == "resolved" and llm_acc is not None and llm_acc <= 2:
        return 5, "high"

    # ── Rule 4: WRONG KBA RETRIEVED ───────────────────────────────────────────
    if has_semantic and llm_acc == 1 and kbas > 0 and resp_length > 20:
        # Chatbot produced a response but it was completely wrong
        return 2, "high"

    if not has_semantic and kbas > 0:
        # Without semantic signal, use structural proxies
        if term in ["abandoned", "restarted"] and resp_length > 80:
            # Long response that user immediately left → likely wrong/useless answer
            if repeated:
                return 2, "medium"  # Repeated message = user saw it was wrong

    # ── Rule 5: CONTEXT IGNORED ───────────────────────────────────────────────
    if has_semantic and llm_acc == 2 and kbas > 0:
        if corrected or human_turns >= 3:
            return 4, "high"

    if not has_semantic and corrected:
        return 4, "medium"

    # ── Rule 6: WRONG KBA SECTION ─────────────────────────────────────────────
    if has_semantic and llm_acc == 2 and kbas > 0 and human_turns < 3:
        return 3, "medium"

    # ── Rule 7: CLEAN ESCALATION ─────────────────────────────────────────────
    if term == "escalated" and human_turns >= 2:
        if kbas == 0 or deflection:
            return 6, "medium"

    # ── Rule 8: UNKNOWN ───────────────────────────────────────────────────────
    return 7, "low"


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("USDA AI Evaluation Engine — Failure Pattern Classifier")
    print("=" * 60)

    # Load data
    df = pd.read_csv(MASTER_FILE)
    with open(TRACES_FILE, encoding="utf-8") as f:
        traces = json.load(f)
    traces_by_id = {t["thread_id"]: t for t in traces}

    print(f"\nClassifying {len(df)} conversations...")

    rows = []
    for _, row in df.iterrows():
        trace = traces_by_id.get(row["thread_id"])
        if not trace:
            pattern_id, confidence = 7, "low"
        else:
            pattern_id, confidence = classify_trace(row, trace)

        rows.append({
            "thread_id":        row["thread_id"],
            "termination_type": row["termination_type"],
            "template_category":row["template_category"],
            "kbas_retrieved":   int(row["kbas_retrieved"]),
            "human_turns":      int(row["human_turns"]),
            "has_semantic":     pd.notna(row.get("llm_accuracy")),
            "pattern_id":       pattern_id,
            "pattern_label":    PATTERNS[pattern_id],
            "confidence":       confidence,
        })

    # Save CSV
    result_df = pd.DataFrame(rows)
    result_df.to_csv(PATTERNS_CSV, index=False)
    print(f"Saved: {PATTERNS_CSV}")

    # Build summary
    summary = build_summary(result_df)
    SUMMARY_FILE.write_text(summary, encoding="utf-8")
    print(f"Saved: {SUMMARY_FILE}")
    print()
    print(summary)

    return result_df


def build_summary(df: pd.DataFrame) -> str:
    lines = []
    lines.append("=" * 65)
    lines.append("USDA AI Evaluation Engine — Failure Pattern Summary")
    lines.append("=" * 65)
    lines.append(f"Total conversations classified: {len(df)}")
    lines.append("")

    # ── Overall distribution ──────────────────────────────────────────────────
    lines.append("── Pattern distribution (all 588 conversations) ──────────────")
    lines.append("")
    lines.append(f"  {'Pattern':<30} {'N':>5} {'%':>7}  Description")
    lines.append("  " + "-" * 85)

    counts = df.groupby(["pattern_id", "pattern_label"]).size().reset_index(name="n")
    counts = counts.sort_values("pattern_id")

    for _, row in counts.iterrows():
        pct = row["n"] / len(df) * 100
        desc = PATTERN_DESCRIPTIONS[row["pattern_id"]][:45]
        lines.append(
            f"  {row['pattern_label']:<30} {row['n']:>5} {pct:>6.1f}%  {desc}"
        )
    lines.append("")

    # ── Confidence breakdown ──────────────────────────────────────────────────
    lines.append("── Classification confidence ─────────────────────────────────")
    conf_counts = df["confidence"].value_counts()
    for conf, n in conf_counts.items():
        lines.append(f"  {conf:<10} {n:>5} ({n/len(df)*100:.1f}%)")
    lines.append("")

    # ── By category ───────────────────────────────────────────────────────────
    lines.append("── Pattern distribution by category ──────────────────────────")
    lines.append("")

    by_cat = df.groupby(["template_category", "pattern_label"]).size().unstack(fill_value=0)
    pattern_cols = [PATTERNS[i] for i in sorted(PATTERNS.keys()) if PATTERNS[i] in by_cat.columns]
    by_cat = by_cat.reindex(columns=pattern_cols, fill_value=0)

    # Short column headers
    short_headers = {
        "SUCCESS":            "SUCC",
        "ZERO_KBA_FAILURE":   "ZERO",
        "WRONG_KBA_RETRIEVED":"WKBA",
        "WRONG_KBA_SECTION":  "WSEC",
        "CONTEXT_IGNORED":    "CTIG",
        "FALSE_RESOLUTION":   "FRES",
        "CLEAN_ESCALATION":   "CESC",
        "UNKNOWN":            "UNKN",
    }

    header_row = f"  {'Category':<45}"
    for col in pattern_cols:
        header_row += f" {short_headers.get(col, col[:4]):>5}"
    lines.append(header_row)
    lines.append("  " + "-" * (45 + 6 * len(pattern_cols)))

    for cat, cat_row in by_cat.iterrows():
        row_str = f"  {cat:<45}"
        for col in pattern_cols:
            row_str += f" {int(cat_row.get(col, 0)):>5}"
        lines.append(row_str)
    lines.append("")

    lines.append("  Column key:")
    for col in pattern_cols:
        lines.append(f"  {short_headers.get(col, col[:4])} = {col}")
    lines.append("")

    # ── Key insights ──────────────────────────────────────────────────────────
    lines.append("── Key insights ──────────────────────────────────────────────")
    total = len(df)
    success_n  = len(df[df["pattern_label"] == "SUCCESS"])
    zero_kba_n = len(df[df["pattern_label"] == "ZERO_KBA_FAILURE"])
    wrong_kba_n= len(df[df["pattern_label"] == "WRONG_KBA_RETRIEVED"])
    unknown_n  = len(df[df["pattern_label"] == "UNKNOWN"])

    lines.append(f"  Conversations that succeeded              : {success_n} ({success_n/total*100:.1f}%)")
    lines.append(f"  Zero-KBA failures                        : {zero_kba_n} ({zero_kba_n/total*100:.1f}%)")
    lines.append(f"  Wrong KBA retrieved                      : {wrong_kba_n} ({wrong_kba_n/total*100:.1f}%)")
    lines.append(f"  Unclassified (unknown)                   : {unknown_n} ({unknown_n/total*100:.1f}%)")
    lines.append("")
    lines.append(f"  Note: {len(df[df['has_semantic']==True])} conversations had semantic scores (golden set).")
    lines.append(f"  Classification confidence is higher for these {len(df[df['has_semantic']==True])} conversations.")
    lines.append(f"  For the remaining {len(df[df['has_semantic']==False])} conversations, rules use structural signals only.")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
