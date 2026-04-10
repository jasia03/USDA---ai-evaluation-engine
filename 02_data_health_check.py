"""
USDA AI Evaluation Engine
Step 1b: Data Health Check

Runs three validation checks on the parsed traces before any metric
computation begins:

  1. Statistical stability  — split traces in half, compare key rates
  2. Category coverage      — are there enough threads per category?
  3. Feedback coverage      — can we rely on feedback as a signal?

Prints a clear PASS / WARN / FAIL verdict for each check, and writes
a summary report to data/health_check_report.txt
"""

import json
import random
from pathlib import Path
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────────
TRACES_FILE  = Path("data/traces.json")
REPORT_FILE  = Path("data/health_check_report.txt")

# Thresholds
RANDOM_SEED = 42
MIN_THREADS_PER_CATEGORY = 20    # fewer than this → warn
MIN_FEEDBACK_COVERAGE = 0.20  # below 20% → feedback unreliable
STABILITY_TOLERANCE = 0.10  # halves must agree within 10 percentage points


# ── Utilities ─────────────────────────────────────────────────────────────────
def resolution_rate(traces: list) -> float:
    resolved = sum(1 for t in traces if t["termination_type"] == "resolved")
    return resolved / len(traces) if traces else 0.0


def escalation_rate(traces: list) -> float:
    escalated = sum(1 for t in traces if t["termination_type"] == "escalated")
    return escalated / len(traces) if traces else 0.0


def abandonment_rate(traces: list) -> float:
    abandoned = sum(1 for t in traces if t["termination_type"] == "abandoned")
    return abandoned / len(traces) if traces else 0.0


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def verdict(status: str) -> str:
    icons = {"PASS": "✓", "WARN": "!", "FAIL": "✗"}
    return f"[{icons.get(status, '?')} {status}]"


# ── Check 1: Statistical Stability ───────────────────────────────────────────
def check_stability(traces: list) -> dict:
    """
    Randomly split traces 50/50. Compute resolution, escalation, and
    abandonment rate on each half. If both halves agree within the
    tolerance threshold, the dataset is stable enough to draw conclusions.
    """
    random.seed(RANDOM_SEED)
    shuffled = traces.copy()
    random.shuffle(shuffled)

    mid = len(shuffled) // 2
    half_a = shuffled[:mid]
    half_b = shuffled[mid:]

    metrics = {
        "resolution_rate":  (resolution_rate,  "Resolution rate"),
        "escalation_rate":  (escalation_rate,  "Escalation rate"),
        "abandonment_rate": (abandonment_rate, "Abandonment rate"),
    }

    results = {}
    all_stable = True

    for key, (fn, label) in metrics.items():
        a = fn(half_a)
        b = fn(half_b)
        diff = abs(a - b)
        stable = diff <= STABILITY_TOLERANCE
        if not stable:
            all_stable = False
        results[key] = {
            "label":  label,
            "half_a": a,
            "half_b": b,
            "diff":   diff,
            "stable": stable,
        }

    return {"status": "PASS" if all_stable else "WARN", "details": results}


# ── Check 2: Category Coverage ────────────────────────────────────────────────
def check_category_coverage(traces: list) -> dict:
    """
    Count threads per template_category. Flag categories with fewer than
    MIN_THREADS_PER_CATEGORY as unreliable for per-category analysis.
    """
    counts = Counter(
        t["template_category"]
        for t in traces
        if t["template_category"]
    )

    categories = {}
    all_pass = True

    for cat, count in sorted(counts.items(), key=lambda x: -x[1]):
        reliable = count >= MIN_THREADS_PER_CATEGORY
        if not reliable:
            all_pass = False
        categories[cat] = {"count": count, "reliable": reliable}

    return {
        "status":     "PASS" if all_pass else "WARN",
        "categories": categories,
        "threshold":  MIN_THREADS_PER_CATEGORY,
    }


# ── Check 3: Feedback Coverage ────────────────────────────────────────────────
def check_feedback_coverage(traces: list) -> dict:
    """
    Count threads that have at least one feedback rating.
    If coverage is below threshold, feedback cannot be used as a primary
    north star metric — we must rely on termination_type instead.
    """
    total = len(traces)
    with_fb = sum(1 for t in traces if t["feedback_rating"] is not None)
    coverage = with_fb / total if total else 0.0

    # Also break down by rating value for those that do have feedback
    rated = [t["feedback_rating"]
             for t in traces if t["feedback_rating"] is not None]
    rating_dist = Counter(int(r) for r in rated)

    return {
        "status":        "PASS" if coverage >= MIN_FEEDBACK_COVERAGE else "WARN",
        "total_threads": total,
        "with_feedback": with_fb,
        "coverage_pct":  coverage,
        "rating_dist":   dict(sorted(rating_dist.items())),
        "recommendation": (
            "Use feedback as PRIMARY signal"
            if coverage >= MIN_FEEDBACK_COVERAGE
            else "Use termination_type as PRIMARY north star; feedback as SECONDARY only"
        ),
    }


# ── Report writer ─────────────────────────────────────────────────────────────
def write_report(stability, coverage, feedback, traces):
    lines = []
    lines.append("=" * 60)
    lines.append("USDA AI Evaluation Engine — Data Health Check Report")
    lines.append("=" * 60)
    lines.append(f"Total traces: {len(traces)}")
    lines.append("")

    # Overall termination breakdown
    tc = Counter(t["termination_type"] for t in traces)
    lines.append("── Termination type breakdown ───────────────────────────")
    for ttype, count in sorted(tc.items(), key=lambda x: -x[1]):
        lines.append(
            f"  {ttype:<12} {count:>4}  ({count/len(traces)*100:.1f}%)")
    lines.append("")

    # Check 1
    lines.append(
        f"── Check 1: Statistical Stability  {verdict(stability['status'])} ──────────")
    lines.append(
        f"  Tolerance: ±{STABILITY_TOLERANCE*100:.0f} percentage points between halves")
    lines.append("")
    for key, r in stability["details"].items():
        s = "stable" if r["stable"] else "UNSTABLE"
        lines.append(
            f"  {r['label']:<20} "
            f"Half A: {pct(r['half_a'])}  "
            f"Half B: {pct(r['half_b'])}  "
            f"Diff: {pct(r['diff'])}  [{s}]"
        )
    lines.append("")

    # Check 2
    lines.append(
        f"── Check 2: Category Coverage  {verdict(coverage['status'])} ──────────────")
    lines.append(
        f"  Minimum threads required for reliable analysis: {coverage['threshold']}")
    lines.append("")
    for cat, info in coverage["categories"].items():
        flag = "OK      " if info["reliable"] else "WARN    "
        lines.append(f"  [{flag}] {cat:<45} {info['count']:>3} threads")
    lines.append("")

    # Check 3
    lines.append(
        f"── Check 3: Feedback Coverage  {verdict(feedback['status'])} ───────────────")
    lines.append(
        f"  Threads with feedback: {feedback['with_feedback']} / "
        f"{feedback['total_threads']}  "
        f"({pct(feedback['coverage_pct'])})"
    )
    lines.append(f"  Minimum required: {pct(MIN_FEEDBACK_COVERAGE)}")
    lines.append("")
    lines.append("  Rating distribution (among threads that have feedback):")
    for rating, count in sorted(feedback["rating_dist"].items()):
        bar = "█" * count
        lines.append(f"    {rating} stars: {count:>3}  {bar}")
    lines.append("")
    lines.append(f"  → Recommendation: {feedback['recommendation']}")
    lines.append("")

    # Summary
    all_statuses = [stability["status"],
                    coverage["status"], feedback["status"]]
    overall = "PASS" if all(s == "PASS" for s in all_statuses) else "WARN"
    lines.append("── Overall verdict ──────────────────────────────────────")
    lines.append(
        f"  {verdict(overall)}  Data is {'ready to proceed' if overall == 'PASS' else 'usable with caveats — see WARNs above'}")
    lines.append("")

    report = "\n".join(lines)
    REPORT_FILE.write_text(report, encoding="utf-8")
    return report


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    with open(TRACES_FILE, encoding="utf-8") as f:
        traces = json.load(f)

    stability = check_stability(traces)
    coverage = check_category_coverage(traces)
    feedback = check_feedback_coverage(traces)

    report = write_report(stability, coverage, feedback, traces)
    print(report)
    print(f"Report saved: {REPORT_FILE}")

    return {"stability": stability, "coverage": coverage, "feedback": feedback}


if __name__ == "__main__":
    main()
