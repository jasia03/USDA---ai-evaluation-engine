"""
USDA AI Evaluation Engine
Step 3: North Star Metrics

Computes 4 north star metrics that measure whether users are actually
succeeding — the bridge between structural signals and business outcomes.

Metrics computed:
  1. resolution_rate       — % of conversations fully resolved by chatbot
  2. escalation_rate       — % of conversations handed to a human agent
  3. self_containment_rate — % of conversations handled without escalation
  4. deflection_value_score — estimated proportion of human effort saved

Breakdowns:
  - Overall
  - By category
  - By month
  - By app version

Outputs:
  data/north_star_metrics.csv    — one row per trace with north star flags
  data/north_star_summary.txt    — human-readable summary with all breakdowns
"""

import json
import csv
from pathlib import Path
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
TRACES_FILE  = Path("data/traces.json")
OUTPUT_DIR   = Path("data")
METRICS_CSV  = Path("data/north_star_metrics.csv")
SUMMARY_FILE = Path("data/north_star_summary.txt")


# ── Constants ─────────────────────────────────────────────────────────────────
# Minimum threads in a group to report with full confidence
MIN_RELIABLE_THREADS = 20


# ── Helpers ───────────────────────────────────────────────────────────────────
def pct(numerator: int, denominator: int) -> float:
    """Return percentage rounded to 1 decimal. Returns 0.0 if denominator is 0."""
    if denominator == 0:
        return 0.0
    return round(numerator / denominator * 100, 1)


def reliability_flag(n: int) -> str:
    """Return a caution flag for small sample groups."""
    return " (!)" if n < MIN_RELIABLE_THREADS else ""


def extract_month_label(conversation_date: str) -> str:
    """
    Convert a date string like '2026-01-02' into a readable month label '2026-01'.
    Used for monthly trend breakdowns.
    """
    try:
        return conversation_date[:7]  # YYYY-MM
    except (TypeError, IndexError):
        return "Unknown"


# ── Per-trace metric computation ──────────────────────────────────────────────
def compute_north_star_row(trace: dict) -> dict:
    """
    Compute north star flags for a single trace.
    These are binary signals — each conversation either did or didn't
    achieve each outcome.
    """
    term = trace["termination_type"]

    # ── Metric 1: Resolution ─────────────────────────────────────────────────
    # The chatbot fully answered the user's question
    is_resolved = 1 if term == "resolved" else 0

    # ── Metric 2: Escalation ─────────────────────────────────────────────────
    # The chatbot handed off to a human agent — direct cost to USDA
    is_escalated = 1 if term == "escalated" else 0

    # ── Metric 3: Self-containment ───────────────────────────────────────────
    # The chatbot handled the conversation without escalating
    # (resolved, abandoned, or restarted — but NOT escalated)
    is_self_contained = 1 if term != "escalated" else 0

    # ── Metric 4: Deflection value ───────────────────────────────────────────
    # Did this conversation save meaningful human effort?
    # A resolved conversation = full deflection (1.0)
    # A self-contained but unresolved conversation = partial deflection (0.5)
    # An escalated conversation = no deflection (0.0)
    if term == "resolved":
        deflection_value = 1.0
    elif term != "escalated":
        deflection_value = 0.5
    else:
        deflection_value = 0.0

    return {
        "thread_id":            trace["thread_id"],
        "termination_type":     term,
        "template_category":    trace["template_category"] or "Unknown",
        "month":                extract_month_label(trace["conversation_date"]),
        "app_version":          trace["app_version"],
        "feedback_rating":      trace["feedback_rating"],
        # ── North star flags ──
        "is_resolved":          is_resolved,
        "is_escalated":         is_escalated,
        "is_self_contained":    is_self_contained,
        "deflection_value":     deflection_value,
    }


# ── Group-level aggregation ───────────────────────────────────────────────────
def aggregate_group(rows: list[dict], group_name: str) -> dict:
    """
    Given a list of trace rows, compute aggregate north star rates.
    Returns a dict of rates for one group (e.g. one category or one month).
    """
    n = len(rows)
    resolved = sum(r["is_resolved"] for r in rows)
    escalated = sum(r["is_escalated"] for r in rows)
    contained = sum(r["is_self_contained"] for r in rows)
    defl_total = sum(r["deflection_value"] for r in rows)

    return {
        "group":                    group_name,
        "threads":                  n,
        "resolved":                 resolved,
        "escalated":                escalated,
        "self_contained":           contained,
        "resolution_rate_pct":      pct(resolved,  n),
        "escalation_rate_pct":      pct(escalated, n),
        "self_containment_rate_pct": pct(contained, n),
        "avg_deflection_value":     round(defl_total / n, 3) if n > 0 else 0.0,
        "reliable":                 n >= MIN_RELIABLE_THREADS,
    }


# ── Summary report ────────────────────────────────────────────────────────────
def build_summary(rows: list[dict], traces: list[dict]) -> str:
    """Build human-readable north star summary with all breakdowns."""

    lines = []
    lines.append("=" * 70)
    lines.append("USDA AI Evaluation Engine — North Star Metrics Summary")
    lines.append("=" * 70)
    lines.append(f"Total conversations: {len(rows)}")
    lines.append("")

    # ── Helper to print a breakdown table ────────────────────────────────────
    def print_breakdown(title: str, groups: dict[str, list]):
        lines.append(f"── {title} {'─' * (55 - len(title))}")
        lines.append(
            f"  {'Group':<42} {'N':>5} "
            f"{'Resolved':>9} {'Escalated':>10} "
            f"{'Contained':>10} {'Deflection':>11}"
        )
        lines.append("  " + "-" * 92)
        for name, group_rows in sorted(groups.items()):
            agg = aggregate_group(group_rows, name)
            flag = reliability_flag(agg["threads"])
            lines.append(
                f"  {name:<42} {agg['threads']:>5} "
                f"{agg['resolution_rate_pct']:>8.1f}% "
                f"{agg['escalation_rate_pct']:>9.1f}% "
                f"{agg['self_containment_rate_pct']:>9.1f}% "
                f"{agg['avg_deflection_value']:>10.3f}{flag}"
            )
        lines.append("")

    # ── Overall ───────────────────────────────────────────────────────────────
    overall = aggregate_group(rows, "Overall")
    lines.append(
        "── Overall north star rates ─────────────────────────────────────")
    lines.append(
        f"  Resolution rate        : {overall['resolution_rate_pct']:>6.1f}%  ({overall['resolved']} of {overall['threads']} conversations)")
    lines.append(
        f"  Escalation rate        : {overall['escalation_rate_pct']:>6.1f}%  ({overall['escalated']} of {overall['threads']} conversations)")
    lines.append(
        f"  Self-containment rate  : {overall['self_containment_rate_pct']:>6.1f}%  ({overall['self_contained']} of {overall['threads']} conversations)")
    lines.append(
        f"  Avg deflection value   : {overall['avg_deflection_value']:>6.3f}  (0=no value, 0.5=partial, 1.0=full)")
    lines.append("")

    # ── By category ───────────────────────────────────────────────────────────
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["template_category"]].append(r)
    print_breakdown("By category", by_cat)

    # ── By month ──────────────────────────────────────────────────────────────
    by_month = defaultdict(list)
    for r in rows:
        by_month[r["month"]].append(r)
    print_breakdown("By month", by_month)

    # ── By app version ────────────────────────────────────────────────────────
    by_version = defaultdict(list)
    for r in rows:
        by_version[r["app_version"]].append(r)
    print_breakdown("By app version", by_version)

    # ── Escalation deep dive ──────────────────────────────────────────────────
    lines.append(
        "── Escalation deep dive ─────────────────────────────────────────")
    lines.append(
        "  Which categories send the most conversations to human agents?")
    lines.append("")

    escalation_by_cat = []
    for cat, cat_rows in sorted(by_cat.items()):
        agg = aggregate_group(cat_rows, cat)
        escalation_by_cat.append(agg)

    # Sort by escalation rate descending
    escalation_by_cat.sort(
        key=lambda x: x["escalation_rate_pct"], reverse=True)

    lines.append(
        f"  {'Category':<42} {'Threads':>7} {'Escalations':>12} {'Rate':>7}")
    lines.append("  " + "-" * 72)
    for agg in escalation_by_cat:
        flag = reliability_flag(agg["threads"])
        lines.append(
            f"  {agg['group']:<42} {agg['threads']:>7} "
            f"{agg['escalated']:>12} {agg['escalation_rate_pct']:>6.1f}%{flag}"
        )
    lines.append("")
    lines.append("  (!) = fewer than 20 threads — interpret with caution")
    lines.append("")

    # ── Resolution deep dive ──────────────────────────────────────────────────
    lines.append(
        "── Resolution deep dive ─────────────────────────────────────────")
    lines.append("  Which categories resolve most successfully?")
    lines.append("")

    resolution_by_cat = sorted(
        [aggregate_group(cat_rows, cat) for cat, cat_rows in by_cat.items()],
        key=lambda x: x["resolution_rate_pct"],
        reverse=True
    )

    lines.append(
        f"  {'Category':<42} {'Threads':>7} {'Resolved':>9} {'Rate':>7}")
    lines.append("  " + "-" * 69)
    for agg in resolution_by_cat:
        flag = reliability_flag(agg["threads"])
        lines.append(
            f"  {agg['group']:<42} {agg['threads']:>7} "
            f"{agg['resolved']:>9} {agg['resolution_rate_pct']:>6.1f}%{flag}"
        )
    lines.append("")

    # ── Monthly trend ─────────────────────────────────────────────────────────
    lines.append(
        "── Monthly resolution trend ──────────────────────────────────────")
    lines.append("  Is the chatbot improving over time?")
    lines.append("")
    lines.append(
        f"  {'Month':<12} {'Threads':>8} {'Resolved':>9} {'Res. Rate':>10} {'Esc. Rate':>10}")
    lines.append("  " + "-" * 53)
    for month in sorted(by_month.keys()):
        agg = aggregate_group(by_month[month], month)
        flag = reliability_flag(agg["threads"])
        lines.append(
            f"  {month:<12} {agg['threads']:>8} {agg['resolved']:>9} "
            f"{agg['resolution_rate_pct']:>9.1f}% "
            f"{agg['escalation_rate_pct']:>9.1f}%{flag}"
        )
    lines.append("")

    # ── Key findings ──────────────────────────────────────────────────────────
    lines.append(
        "── Key findings ──────────────────────────────────────────────────")

    # Best and worst category by resolution
    reliable_cats = [a for a in resolution_by_cat if a["reliable"]]
    if reliable_cats:
        best = reliable_cats[0]
        worst = reliable_cats[-1]
        lines.append(
            f"  Highest resolution rate : {best['group']} ({best['resolution_rate_pct']}%)")
        lines.append(
            f"  Lowest resolution rate  : {worst['group']} ({worst['resolution_rate_pct']}%)")

    # Category with highest escalation
    reliable_esc = [a for a in escalation_by_cat if a["reliable"]]
    if reliable_esc:
        highest_esc = reliable_esc[0]
        lines.append(
            f"  Highest escalation rate : {highest_esc['group']} ({highest_esc['escalation_rate_pct']}%)")

    # Self-containment insight
    lines.append(
        f"\n  Self-containment rate of {overall['self_containment_rate_pct']}% means "
        f"{overall['self_contained']} of {overall['threads']} conversations\n"
        f"  were handled without a human agent — but only {overall['resolution_rate_pct']}% "
        f"were actually resolved.\n"
        f"  The gap ({overall['self_containment_rate_pct'] - overall['resolution_rate_pct']:.1f} percentage points) "
        f"represents conversations the chatbot\n"
        f"  attempted but failed to resolve — users who abandoned or restarted."
    )
    lines.append("")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("USDA AI Evaluation Engine — North Star Metrics")
    print("=" * 60)

    # Load traces
    with open(TRACES_FILE, encoding="utf-8") as f:
        traces = json.load(f)
    print(f"\nLoaded {len(traces)} traces")

    # Compute per-trace north star rows
    print("Computing north star metrics...")
    rows = [compute_north_star_row(t) for t in traces]
    print(f"  Done — {len(rows)} rows computed")

    # Save CSV
    fieldnames = list(rows[0].keys())
    with open(METRICS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {METRICS_CSV}")

    # Build and save summary
    summary = build_summary(rows, traces)
    SUMMARY_FILE.write_text(summary, encoding="utf-8")
    print(f"Saved: {SUMMARY_FILE}")

    print()
    print(summary)

    return rows


if __name__ == "__main__":
    main()
