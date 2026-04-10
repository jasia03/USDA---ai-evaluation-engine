"""
USDA AI Evaluation Engine
Step 2: Structural Metrics

Computes 9 structural metrics on every trace. These are deterministic,
high-frequency signals — no AI judgment required.

Metrics computed per trace:
  1.  avg_response_length       — average words per real AI text response
  2.  total_turns               — total number of messages in the conversation
  3.  human_turns               — number of times the user sent a message
  4.  avg_response_latency_secs — average seconds between human msg and next AI reply
  5.  kbas_retrieved            — number of unique KBA documents retrieved
  6.  io_ratio                  — total AI words / total human words
  7.  session_duration_secs     — seconds from first to last message
  8.  restart_flag              — 1 if termination_type == 'restarted', else 0
  9.  avg_vocab_complexity      — average Flesch-Kincaid grade level of AI responses

Outputs:
  data/structural_metrics.csv   — one row per trace, all 9 metrics
  data/structural_summary.txt   — human-readable summary with breakdowns
"""

import json
import re
import csv
import datetime
from pathlib import Path
from collections import defaultdict


# ── Pure Python Flesch-Kincaid implementation ─────────────────────────────────
def count_syllables(word: str) -> int:
    """
    Count syllables in a word using vowel-group rules.
    This is the standard approximation used by readability tools.
    Accurate enough for grade-level scoring across large text samples.
    """
    word = word.lower().strip(".,!?;:'\"()-")
    if not word:
        return 0
    # Count vowel groups (consecutive vowels = one syllable)
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    # Silent 'e' at end
    if word.endswith("e") and len(word) > 2 and word[-2] not in vowels:
        count = max(1, count - 1)
    return max(1, count)


def flesch_kincaid_grade(text: str) -> float:
    """
    Compute Flesch-Kincaid Grade Level for a piece of text.

    Formula: 0.39 × (words/sentences) + 11.8 × (syllables/words) − 15.59

    Returns a grade level (e.g. 8.0 = 8th grade reading level).
    Lower = easier to read. Typical target for plain English: 8–10.
    """
    # Split into sentences on . ! ?
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    num_sentences = max(1, len(sentences))

    # Split into words (ignore punctuation tokens)
    words = [w for w in re.findall(r"[a-zA-Z']+", text) if w]
    num_words = max(1, len(words))

    num_syllables = sum(count_syllables(w) for w in words)

    grade = (
        0.39 * (num_words / num_sentences)
        + 11.8 * (num_syllables / num_words)
        - 15.59
    )
    return round(grade, 2)

# ── Paths ─────────────────────────────────────────────────────────────────────
TRACES_FILE  = Path("data/traces.json")
OUTPUT_DIR   = Path("data")
METRICS_CSV  = Path("data/structural_metrics.csv")
SUMMARY_FILE = Path("data/structural_summary.txt")

# ── Constants ─────────────────────────────────────────────────────────────────
# Minimum words for an AI response to count as a real text response
# (filters out button UI messages and one-word system replies)
MIN_WORDS_FOR_REAL_RESPONSE = 5


# ── Helpers ───────────────────────────────────────────────────────────────────
def is_real_ai_response(content: str) -> bool:
    """
    Return True if this AI message is a genuine text response,
    not a UI button message or system notice.

    Button messages look like: {'buttons': ['Yes', 'No'], 'pre_text': '...'}
    We filter these out before computing length and vocabulary metrics.
    """
    if not content:
        return False
    # Button/UI responses start with { and contain 'buttons'
    stripped = content.strip()
    if stripped.startswith("{") and "buttons" in stripped:
        return False
    # Must have enough words to be meaningful
    if len(stripped.split()) < MIN_WORDS_FOR_REAL_RESPONSE:
        return False
    return True


def count_words(text: str) -> int:
    """Count words in a string."""
    return len(text.split()) if text else 0


def parse_time(time_str: str):
    """
    Parse a time string like '00:40:39.700000' into a datetime.time object.
    Returns None if parsing fails.
    """
    if not time_str or time_str == "None":
        return None
    try:
        # Handle microseconds
        return datetime.time.fromisoformat(time_str)
    except (ValueError, AttributeError):
        return None


def time_to_seconds(t: datetime.time) -> float:
    """Convert a datetime.time to total seconds since midnight."""
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1_000_000


def time_diff_seconds(t1: datetime.time, t2: datetime.time) -> float:
    """
    Compute seconds between two time objects.
    Handles the edge case where t2 is before t1 (conversation crosses midnight).
    Always returns a positive number.
    """
    s1 = time_to_seconds(t1)
    s2 = time_to_seconds(t2)
    diff = s2 - s1
    # If negative, conversation crossed midnight — add 24 hours
    if diff < 0:
        diff += 86400
    return diff


def safe_avg(values: list) -> float:
    """Return average of a list, or 0.0 if the list is empty."""
    return round(sum(values) / len(values), 2) if values else 0.0


# ── Metric computations ───────────────────────────────────────────────────────
def compute_metrics(trace: dict) -> dict:
    """
    Compute all 9 structural metrics for a single trace.
    Returns a flat dictionary suitable for one CSV row.
    """
    messages = trace["messages"]

    # Separate messages by role for convenience
    human_msgs = [m for m in messages if m["role"] == "human"]
    ai_msgs    = [m for m in messages if m["role"] == "ai"]

    # Real AI text responses only (filters out button UI messages)
    real_ai_msgs = [m for m in ai_msgs if is_real_ai_response(m["content"])]

    # ── Metric 1: Average response length ────────────────────────────────────
    ai_word_counts = [count_words(m["content"]) for m in real_ai_msgs]
    avg_response_length = safe_avg(ai_word_counts)

    # ── Metric 2: Total turns ────────────────────────────────────────────────
    total_turns = len(messages)

    # ── Metric 3: Human turns ────────────────────────────────────────────────
    human_turns = len(human_msgs)

    # ── Metric 4: Average response latency ───────────────────────────────────
    # For each human message, find the NEXT ai message and compute the gap
    latencies = []
    all_msgs_sorted = sorted(messages, key=lambda m: m["sequence"])

    for i, msg in enumerate(all_msgs_sorted):
        if msg["role"] != "human":
            continue
        t_human = parse_time(msg["message_time"])
        if t_human is None:
            continue
        # Find next ai message after this human message
        for next_msg in all_msgs_sorted[i + 1:]:
            if next_msg["role"] == "ai":
                t_ai = parse_time(next_msg["message_time"])
                if t_ai is not None:
                    latency = time_diff_seconds(t_human, t_ai)
                    # Sanity cap: ignore gaps over 30 minutes
                    # (user likely walked away — not chatbot latency)
                    if latency <= 1800:
                        latencies.append(latency)
                break  # Only count the FIRST ai reply per human message

    avg_response_latency_secs = safe_avg(latencies)

    # ── Metric 5: KBAs retrieved ─────────────────────────────────────────────
    kbas_retrieved = len(trace["kba_docs_retrieved"])

    # ── Metric 6: IO ratio ───────────────────────────────────────────────────
    total_ai_words    = sum(count_words(m["content"]) for m in real_ai_msgs)
    total_human_words = sum(count_words(m["content"]) for m in human_msgs)
    io_ratio = round(total_ai_words / total_human_words, 2) if total_human_words > 0 else 0.0

    # ── Metric 7: Session duration ───────────────────────────────────────────
    all_times = [
        parse_time(m["message_time"])
        for m in messages
        if parse_time(m["message_time"]) is not None
    ]
    if len(all_times) >= 2:
        t_first = min(all_times, key=time_to_seconds)
        t_last  = max(all_times, key=time_to_seconds)
        session_duration_secs = round(time_diff_seconds(t_first, t_last), 2)
    else:
        session_duration_secs = 0.0

    # ── Metric 8: Restart flag ───────────────────────────────────────────────
    restart_flag = 1 if trace["termination_type"] == "restarted" else 0

    # ── Metric 9: Average vocabulary complexity ──────────────────────────────
    fk_scores = []
    for m in real_ai_msgs:
        content = m["content"].strip()
        if len(content.split()) >= 10:   # Need enough words for a reliable score
            score = flesch_kincaid_grade(content)
            # Cap at reasonable bounds — textstat can return extreme values
            # for very short or unusual text
            if -5 <= score <= 25:
                fk_scores.append(score)

    avg_vocab_complexity = safe_avg(fk_scores)

    return {
        "thread_id":                 trace["thread_id"],
        "termination_type":          trace["termination_type"],
        "template_category":         trace["template_category"] or "Unknown",
        "feedback_rating":           trace["feedback_rating"],
        "conversation_date":         trace["conversation_date"],
        # ── The 9 metrics ──
        "avg_response_length":       avg_response_length,
        "total_turns":               total_turns,
        "human_turns":               human_turns,
        "avg_response_latency_secs": avg_response_latency_secs,
        "kbas_retrieved":            kbas_retrieved,
        "io_ratio":                  io_ratio,
        "session_duration_secs":     session_duration_secs,
        "restart_flag":              restart_flag,
        "avg_vocab_complexity":      avg_vocab_complexity,
    }


# ── Summary report ────────────────────────────────────────────────────────────
def build_summary(rows: list[dict]) -> str:
    """
    Build a human-readable summary of all metrics broken down by
    termination type and by category.
    """
    metric_cols = [
        "avg_response_length",
        "total_turns",
        "human_turns",
        "avg_response_latency_secs",
        "kbas_retrieved",
        "io_ratio",
        "session_duration_secs",
        "avg_vocab_complexity",
    ]

    def group_avg(group: list[dict], col: str) -> float:
        vals = [r[col] for r in group if r[col] is not None]
        return safe_avg([v for v in vals if isinstance(v, (int, float))])

    def section(title: str, groups: dict) -> list[str]:
        lines = [f"── {title} {'─' * (50 - len(title))}"]
        # Header
        hdr = f"  {'Group':<35}"
        for col in metric_cols:
            short = col.replace("avg_", "").replace("_secs", "s").replace("_", " ")[:10]
            hdr += f" {short:>10}"
        lines.append(hdr)
        lines.append("  " + "-" * (35 + 11 * len(metric_cols)))
        # Rows
        for name, group in sorted(groups.items()):
            row_str = f"  {name:<35}"
            for col in metric_cols:
                val = group_avg(group, col)
                row_str += f" {val:>10.1f}"
            lines.append(row_str)
        lines.append("")
        return lines

    lines = []
    lines.append("=" * 75)
    lines.append("USDA AI Evaluation Engine — Structural Metrics Summary")
    lines.append("=" * 75)
    lines.append(f"Total traces: {len(rows)}")
    lines.append("")

    # ── Overall averages ──────────────────────────────────────────────────────
    lines.append("── Overall averages ───────────────────────────────────────────")
    for col in metric_cols:
        avg = group_avg(rows, col)
        label = col.replace("_", " ").title()
        lines.append(f"  {label:<40} {avg:>8.2f}")
    lines.append("")

    # ── By termination type ───────────────────────────────────────────────────
    by_term = defaultdict(list)
    for r in rows:
        by_term[r["termination_type"]].append(r)
    lines.extend(section("By termination type", by_term))

    # ── By category ───────────────────────────────────────────────────────────
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["template_category"]].append(r)
    lines.extend(section("By category", by_cat))

    # ── Restart rate by category ──────────────────────────────────────────────
    lines.append("── Restart rate by category ───────────────────────────────────")
    lines.append(f"  {'Category':<40} {'Threads':>8} {'Restarts':>9} {'Rate':>7}")
    lines.append("  " + "-" * 68)
    for cat, group in sorted(by_cat.items()):
        n        = len(group)
        restarts = sum(r["restart_flag"] for r in group)
        rate     = restarts / n * 100 if n > 0 else 0
        flag     = " (!)" if n < 20 else ""
        lines.append(f"  {cat:<40} {n:>8} {restarts:>9} {rate:>6.1f}%{flag}")
    lines.append("")
    lines.append("  (!) = fewer than 20 threads — interpret with caution")
    lines.append("")

    # ── Key findings ──────────────────────────────────────────────────────────
    resolved  = [r for r in rows if r["termination_type"] == "resolved"]
    abandoned = [r for r in rows if r["termination_type"] == "abandoned"]
    escalated = [r for r in rows if r["termination_type"] == "escalated"]

    lines.append("── Key findings ────────────────────────────────────────────────")

    res_rate = len(resolved) / len(rows) * 100
    lines.append(f"  Resolution rate          : {res_rate:.1f}% ({len(resolved)} of {len(rows)} conversations)")

    esc_rate = len(escalated) / len(rows) * 100
    lines.append(f"  Escalation rate          : {esc_rate:.1f}% ({len(escalated)} of {len(rows)} conversations)")

    abd_rate = len(abandoned) / len(rows) * 100
    lines.append(f"  Abandonment rate         : {abd_rate:.1f}% ({len(abandoned)} of {len(rows)} conversations)")

    lines.append("")

    if resolved and abandoned:
        res_lat = group_avg(resolved, "avg_response_latency_secs")
        abd_lat = group_avg(abandoned, "avg_response_latency_secs")
        lines.append(f"  Avg latency (resolved)   : {res_lat:.1f}s")
        lines.append(f"  Avg latency (abandoned)  : {abd_lat:.1f}s")

        res_turns = group_avg(resolved, "human_turns")
        abd_turns = group_avg(abandoned, "human_turns")
        lines.append(f"  Avg human turns (resolved)  : {res_turns:.1f}")
        lines.append(f"  Avg human turns (abandoned) : {abd_turns:.1f}")

        res_dur = group_avg(resolved, "session_duration_secs")
        abd_dur = group_avg(abandoned, "session_duration_secs")
        lines.append(f"  Avg session duration (resolved)  : {res_dur:.0f}s ({res_dur/60:.1f} min)")
        lines.append(f"  Avg session duration (abandoned) : {abd_dur:.0f}s ({abd_dur/60:.1f} min)")

        res_fk = group_avg(resolved, "avg_vocab_complexity")
        abd_fk = group_avg(abandoned, "avg_vocab_complexity")
        lines.append(f"  Avg vocab complexity (resolved)  : grade {res_fk:.1f}")
        lines.append(f"  Avg vocab complexity (abandoned) : grade {abd_fk:.1f}")

    lines.append("")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("USDA AI Evaluation Engine — Structural Metrics")
    print("=" * 60)

    # Load traces
    with open(TRACES_FILE, encoding="utf-8") as f:
        traces = json.load(f)
    print(f"\nLoaded {len(traces)} traces")

    # Compute metrics for every trace
    print("Computing metrics...")
    rows = []
    for i, trace in enumerate(traces):
        row = compute_metrics(trace)
        rows.append(row)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(traces)} done")

    print(f"  {len(traces)}/{len(traces)} done")

    # Save CSV
    fieldnames = list(rows[0].keys())
    with open(METRICS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {METRICS_CSV}")

    # Build and save summary
    summary = build_summary(rows)
    SUMMARY_FILE.write_text(summary, encoding="utf-8")
    print(f"Saved: {SUMMARY_FILE}")

    # Print summary
    print()
    print(summary)

    return rows


if __name__ == "__main__":
    main()
