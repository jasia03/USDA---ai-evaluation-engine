"""
USDA AI Evaluation Engine
Step 12: Latency Per Turn + Deflection Rate

Two quick metrics that extend the structural analysis:

PART A — Latency per turn
  Breaks down response latency by turn number within a conversation.
  Answers: does the chatbot get faster or slower as a conversation
  progresses? Does early-turn latency differ from late-turn latency?

PART B — Deflection rate
  Measures how often the chatbot explicitly acknowledged it could not
  help and redirected the user. This is distinct from zero-KBA failure
  (where the chatbot stalled silently) — here the chatbot at least
  communicated its limitation honestly.

Outputs:
  data/latency_per_turn.csv        — latency at each turn number
  data/deflection_rate.csv         — per-trace deflection flags
  data/latency_deflection_summary.txt — combined summary
"""

import json
import re
import csv
import datetime
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
TRACES_FILE   = Path("data/traces.json")
MASTER_FILE   = Path("data/master_metrics.csv")
LATENCY_CSV   = Path("data/latency_per_turn.csv")
DEFLECT_CSV   = Path("data/deflection_rate.csv")
SUMMARY_FILE  = Path("data/latency_deflection_summary.txt")

# ── Deflection phrases ────────────────────────────────────────────────────────
DEFLECTION_PHRASES = [
    "not contain", "do not contain", "does not contain",
    "not available in", "no information", "unable to find",
    "cannot find", "not found in", "not provided in",
    "consult additional", "consult relevant documentation",
    "please consult", "not have specific",
    "don't have specific", "i wasn't able to find",
    "i was unable to find", "no specific guidance",
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_time(time_str: str):
    if not time_str or time_str == "None":
        return None
    try:
        return datetime.time.fromisoformat(time_str)
    except (ValueError, AttributeError):
        return None


def time_to_seconds(t: datetime.time) -> float:
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1_000_000


def time_diff_seconds(t1: datetime.time, t2: datetime.time) -> float:
    s1 = time_to_seconds(t1)
    s2 = time_to_seconds(t2)
    diff = s2 - s1
    if diff < 0:
        diff += 86400
    return diff


def is_real_ai(msg: dict) -> bool:
    if msg["role"] != "ai":
        return False
    content = msg["content"].strip()
    if content.startswith("{") and "buttons" in content:
        return False
    return len(content.split()) >= 5


def has_deflection(trace: dict) -> tuple[bool, int]:
    """
    Check if any AI message contains a deflection phrase.
    Returns (has_deflection, count_of_deflecting_messages).
    """
    count = 0
    for msg in trace["messages"]:
        if msg["role"] != "ai":
            continue
        content = msg["content"].lower()
        if any(phrase in content for phrase in DEFLECTION_PHRASES):
            count += 1
    return count > 0, count


# ── PART A: Latency per turn ──────────────────────────────────────────────────
def compute_latency_per_turn(traces: list) -> list:
    """
    For each conversation, compute the latency for each human→AI exchange.
    Turn 1 = first human message to first AI reply.
    Turn 2 = second human message to second AI reply. Etc.
    """
    turn_latencies = defaultdict(list)  # turn_number → list of latencies
    rows = []

    for trace in traces:
        msgs = sorted(trace["messages"], key=lambda m: m["sequence"])
        term = trace["termination_type"]
        cat  = trace["template_category"] or "Unknown"

        # Find each human→AI pair in sequence
        turn_num = 0
        i = 0
        while i < len(msgs):
            msg = msgs[i]
            if msg["role"] != "human":
                i += 1
                continue

            t_human = parse_time(msg["message_time"])
            if t_human is None:
                i += 1
                continue

            # Find next AI response after this human message
            for j in range(i + 1, len(msgs)):
                next_msg = msgs[j]
                if not is_real_ai(next_msg):
                    continue
                t_ai = parse_time(next_msg["message_time"])
                if t_ai is None:
                    break
                latency = time_diff_seconds(t_human, t_ai)
                if latency <= 1800:  # cap at 30 minutes
                    turn_num += 1
                    turn_latencies[turn_num].append(latency)
                    rows.append({
                        "thread_id":        trace["thread_id"],
                        "termination_type": term,
                        "template_category":cat,
                        "turn_number":      turn_num,
                        "latency_secs":     round(latency, 3),
                    })
                break
            i += 1

    return rows, turn_latencies


# ── PART B: Deflection rate ───────────────────────────────────────────────────
def compute_deflection_rate(traces: list, master: pd.DataFrame) -> list:
    rows = []
    for trace in traces:
        deflects, deflect_count = has_deflection(trace)
        master_row = master[master["thread_id"] == trace["thread_id"]]
        term = trace["termination_type"]
        cat  = trace["template_category"] or "Unknown"
        kbas = int(master_row["kbas_retrieved"].iloc[0]) if len(master_row) > 0 else 0

        rows.append({
            "thread_id":          trace["thread_id"],
            "termination_type":   term,
            "template_category":  cat,
            "kbas_retrieved":     kbas,
            "has_deflection":     int(deflects),
            "deflection_msg_count": deflect_count,
            "deflection_type": (
                "explicit_acknowledgment" if deflects and kbas > 0
                else "silent_stall"       if kbas == 0
                else "no_deflection"
            ),
        })
    return rows


# ── Summary builder ───────────────────────────────────────────────────────────
def build_summary(
    latency_rows: list,
    turn_latencies: dict,
    deflection_rows: list
) -> str:
    lines = []
    lines.append("=" * 65)
    lines.append("USDA AI Evaluation Engine — Latency Per Turn + Deflection Rate")
    lines.append("=" * 65)
    lines.append("")

    # ── Latency per turn ──────────────────────────────────────────────────────
    lines.append("── PART A: Response latency by turn number ───────────────────")
    lines.append("")
    lines.append("  Does the chatbot get faster or slower as the conversation")
    lines.append("  progresses?")
    lines.append("")
    lines.append(f"  {'Turn':>6} {'N':>6} {'Mean (s)':>10} {'Median (s)':>12} {'Std dev':>10}")
    lines.append("  " + "-" * 48)

    for turn in sorted(turn_latencies.keys())[:10]:  # Show first 10 turns
        vals = turn_latencies[turn]
        if len(vals) < 5:
            continue
        mean   = sum(vals) / len(vals)
        sorted_vals = sorted(vals)
        median = sorted_vals[len(sorted_vals) // 2]
        std    = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        lines.append(
            f"  {turn:>6} {len(vals):>6} {mean:>10.2f} {median:>12.2f} {std:>10.2f}"
        )
    lines.append("")

    # Insight
    if turn_latencies.get(1) and turn_latencies.get(3):
        t1_avg = sum(turn_latencies[1]) / len(turn_latencies[1])
        t3_avg = sum(turn_latencies[3]) / len(turn_latencies[3])
        direction = "decreases" if t3_avg < t1_avg else "increases"
        lines.append(
            f"  Turn 1 avg latency: {t1_avg:.2f}s  |  "
            f"Turn 3 avg latency: {t3_avg:.2f}s"
        )
        lines.append(
            f"  → Response latency {direction} as conversations progress."
        )
    lines.append("")

    # ── Deflection rate ───────────────────────────────────────────────────────
    df_defl = pd.DataFrame(deflection_rows)
    total = len(df_defl)

    lines.append("── PART B: Deflection rate ───────────────────────────────────")
    lines.append("")
    lines.append("  Deflection = chatbot explicitly said it could not help.")
    lines.append("  Silent stall = zero KBAs, no response at all.")
    lines.append("")

    defl_counts = df_defl["deflection_type"].value_counts()
    for dtype, count in defl_counts.items():
        lines.append(f"  {dtype:<30} {count:>5}  ({count/total*100:.1f}%)")
    lines.append("")

    # By termination type
    lines.append(f"  {'Outcome':<15} {'Total':>7} {'Deflected':>10} {'Rate':>8}")
    lines.append("  " + "-" * 44)
    by_term = df_defl.groupby("termination_type").agg(
        total=("thread_id", "count"),
        deflected=("has_deflection", "sum")
    )
    for term, row in by_term.iterrows():
        rate = row["deflected"] / row["total"] * 100
        lines.append(
            f"  {term:<15} {int(row['total']):>7} "
            f"{int(row['deflected']):>10} {rate:>7.1f}%"
        )
    lines.append("")

    # By category
    lines.append(f"  {'Category':<42} {'Total':>6} {'Deflected':>10} {'Rate':>8}")
    lines.append("  " + "-" * 70)
    by_cat = df_defl[df_defl["template_category"] != "Unknown"].groupby(
        "template_category"
    ).agg(
        total=("thread_id", "count"),
        deflected=("has_deflection", "sum")
    ).sort_values("deflected", ascending=False)

    for cat, row in by_cat.iterrows():
        rate = row["deflected"] / row["total"] * 100
        lines.append(
            f"  {cat:<42} {int(row['total']):>6} "
            f"{int(row['deflected']):>10} {rate:>7.1f}%"
        )
    lines.append("")

    # Key insight
    total_explicit = int(defl_counts.get("explicit_acknowledgment", 0))
    total_silent   = int(defl_counts.get("silent_stall", 0))
    lines.append("── Key insight ───────────────────────────────────────────────")
    lines.append(
        f"  Explicit deflections (chatbot said 'I don't know'): {total_explicit} "
        f"({total_explicit/total*100:.1f}%)"
    )
    lines.append(
        f"  Silent stalls (zero KBA, no response):              {total_silent} "
        f"({total_silent/total*100:.1f}%)"
    )
    lines.append(
        f"  → Silent stalls are more harmful than explicit deflections"
        f" because users wait without knowing help is unavailable."
    )
    lines.append("")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("USDA AI Evaluation Engine — Latency Per Turn + Deflection Rate")
    print("=" * 60)

    with open(TRACES_FILE, encoding="utf-8") as f:
        traces = json.load(f)
    master = pd.read_csv(MASTER_FILE)

    print(f"\nLoaded {len(traces)} traces")

    # Part A
    print("Computing latency per turn...")
    latency_rows, turn_latencies = compute_latency_per_turn(traces)
    print(f"  {len(latency_rows)} turn-level latency measurements")

    with open(LATENCY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(latency_rows[0].keys()))
        writer.writeheader()
        writer.writerows(latency_rows)
    print(f"  Saved: {LATENCY_CSV}")

    # Part B
    print("Computing deflection rate...")
    deflection_rows = compute_deflection_rate(traces, master)
    explicit = sum(1 for r in deflection_rows if r["deflection_type"] == "explicit_acknowledgment")
    print(f"  Explicit deflections: {explicit}")

    with open(DEFLECT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(deflection_rows[0].keys()))
        writer.writeheader()
        writer.writerows(deflection_rows)
    print(f"  Saved: {DEFLECT_CSV}")

    # Summary
    summary = build_summary(latency_rows, turn_latencies, deflection_rows)
    SUMMARY_FILE.write_text(summary, encoding="utf-8")
    print(f"Saved: {SUMMARY_FILE}")
    print()
    print(summary)


if __name__ == "__main__":
    main()
