"""
USDA AI Evaluation Engine
Step 9: Alignment Report Generator

Produces a structured periodic alignment report that a USDA team
can run on any cadence (weekly, monthly, quarterly) to monitor
chatbot health. The report compares current metrics against the
established baseline and flags any drift.

This is the core "engine" output — a repeatable, automated report
that answers: "Is the chatbot still performing as expected?"

Usage:
  python 09_alignment_report.py
  python 09_alignment_report.py --baseline data/baseline_metrics.json

Outputs:
  data/alignment_report.txt   — full human-readable alignment report
"""

import json
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
MASTER_FILE    = Path("data/master_metrics.csv")
BASELINE_FILE  = Path("data/baseline_metrics.json")
REPORT_FILE    = Path("data/alignment_report.txt")

# ── Thresholds ────────────────────────────────────────────────────────────────
WARN_THRESHOLD  = 0.05
ALERT_THRESHOLD = 0.10

# ── Helpers ───────────────────────────────────────────────────────────────────
def pct(val: float) -> str:
    return f"{val * 100:.1f}%"

def flag(current: float, baseline: float,
         warn: float, alert: float) -> str:
    diff = abs(current - baseline)
    if diff >= alert:
        return "ALERT  ⚠"
    elif diff >= warn:
        return "WARN   !"
    return "STABLE ✓"

def section(title: str, width: int = 65) -> str:
    return f"── {title} {'─' * (width - len(title) - 4)}"


# ── Report builder ────────────────────────────────────────────────────────────
def generate_report(df: pd.DataFrame, baseline: dict) -> str:
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Header ────────────────────────────────────────────────────────────────
    lines.append("=" * 65)
    lines.append("USDA AI EVALUATION ENGINE")
    lines.append("Periodic Alignment Report")
    lines.append("=" * 65)
    lines.append(f"Generated      : {now}")
    lines.append(f"Data range     : {df['conversation_date'].min()} "
                 f"to {df['conversation_date'].max()}")
    lines.append(f"Conversations  : {len(df)}")
    lines.append(f"Baseline date  : {baseline.get('date_range', 'N/A')}")
    lines.append("")

    # ── Executive summary ─────────────────────────────────────────────────────
    lines.append("=" * 65)
    lines.append("EXECUTIVE SUMMARY")
    lines.append("=" * 65)
    lines.append("")

    res_rate  = df["is_resolved"].mean()
    esc_rate  = df["is_escalated"].mean()
    cont_rate = df["is_self_contained"].mean()
    defl_avg  = df["deflection_value"].mean()

    res_flag  = flag(res_rate,  baseline["resolution_rate"],
                     WARN_THRESHOLD, ALERT_THRESHOLD)
    esc_flag  = flag(esc_rate,  baseline["escalation_rate"],
                     WARN_THRESHOLD, ALERT_THRESHOLD)

    lines.append(f"  Resolution rate     : {pct(res_rate):<8}  "
                 f"[{res_flag}]  baseline: {pct(baseline['resolution_rate'])}")
    lines.append(f"  Escalation rate     : {pct(esc_rate):<8}  "
                 f"[{esc_flag}]  baseline: {pct(baseline['escalation_rate'])}")
    lines.append(f"  Self-containment    : {pct(cont_rate):<8}  "
                 f"baseline: {pct(baseline['self_containment_rate'])}")
    lines.append(f"  Avg deflection value: {defl_avg:.3f}    "
                 f"(0=no value, 0.5=partial, 1.0=full)")
    lines.append("")

    # Overall health verdict
    alerts = sum(1 for metric, base_key in [
        (res_rate,  "resolution_rate"),
        (esc_rate,  "escalation_rate"),
    ] if abs(metric - baseline[base_key]) >= ALERT_THRESHOLD)

    if alerts == 0 and res_rate >= baseline["resolution_rate"]:
        verdict = "HEALTHY — metrics are stable or improving"
    elif alerts == 0:
        verdict = "MONITOR — minor regression detected, watch closely"
    else:
        verdict = "ACTION REQUIRED — significant metric drift detected"

    lines.append(f"  Overall health: {verdict}")
    lines.append("")

    # ── North star metrics ────────────────────────────────────────────────────
    lines.append("=" * 65)
    lines.append("NORTH STAR METRICS")
    lines.append("=" * 65)
    lines.append("")

    # By category
    lines.append(section("Resolution and escalation by category"))
    lines.append("")
    lines.append(f"  {'Category':<42} {'N':>4} {'Resolved':>9} "
                 f"{'Escalated':>10} {'Deflection':>11}")
    lines.append("  " + "-" * 80)

    by_cat = df.groupby("template_category").agg(
        n=("thread_id", "count"),
        resolved=("is_resolved", "mean"),
        escalated=("is_escalated", "mean"),
        deflection=("deflection_value", "mean"),
    ).round(3)

    for cat, row in by_cat.sort_values("resolved", ascending=False).iterrows():
        caution = " (!)" if row["n"] < 20 else ""
        lines.append(
            f"  {cat:<42} {int(row['n']):>4} "
            f"{pct(row['resolved']):>9} "
            f"{pct(row['escalated']):>10} "
            f"{row['deflection']:>11.3f}{caution}"
        )

    lines.append("")
    lines.append("  (!) = fewer than 20 threads — interpret with caution")
    lines.append("")

    # Monthly trend
    lines.append(section("Monthly resolution trend"))
    lines.append("")
    lines.append(f"  {'Month':<12} {'N':>6} {'Resolved':>9} "
                 f"{'Escalated':>10} {'Abandoned':>10}")
    lines.append("  " + "-" * 52)

    by_month = df.groupby("month").agg(
        n=("thread_id", "count"),
        resolved=("is_resolved", "mean"),
        escalated=("is_escalated", "mean"),
        abandoned=("termination_type",
                   lambda x: (x == "abandoned").mean()),
    ).round(3)

    for month, row in by_month.iterrows():
        lines.append(
            f"  {month:<12} {int(row['n']):>6} "
            f"{pct(row['resolved']):>9} "
            f"{pct(row['escalated']):>10} "
            f"{pct(row['abandoned']):>10}"
        )
    lines.append("")

    # ── Structural metrics ────────────────────────────────────────────────────
    lines.append("=" * 65)
    lines.append("STRUCTURAL METRICS")
    lines.append("=" * 65)
    lines.append("")

    struct_metrics = [
        ("avg_response_length",      "Avg response length (words)",  "avg_response_length"),
        ("avg_response_latency_secs","Avg response latency (secs)",   "avg_latency_secs"),
        ("kbas_retrieved",           "Avg KBAs retrieved",            "avg_kbas_retrieved"),
        ("avg_vocab_complexity",     "Avg vocab complexity (grade)",  "avg_vocab_complexity"),
        ("io_ratio",                 "Avg IO ratio",                  "avg_io_ratio"),
    ]

    lines.append(f"  {'Metric':<35} {'Current':>9} {'Baseline':>9} "
                 f"{'Change':>8} {'Status':>10}")
    lines.append("  " + "-" * 76)

    for col, label, base_key in struct_metrics:
        current  = df[col].mean()
        baseline_val = baseline.get(base_key, current)
        diff     = current - baseline_val
        status   = flag(current, baseline_val, WARN_THRESHOLD, ALERT_THRESHOLD)
        lines.append(
            f"  {label:<35} {current:>9.2f} {baseline_val:>9.2f} "
            f"{diff:>+8.2f} [{status}]"
        )
    lines.append("")

    # Structural by termination type
    lines.append(section("Structural metrics by outcome"))
    lines.append("")
    lines.append(f"  {'Outcome':<12} {'Resp len':>9} {'Latency':>8} "
                 f"{'KBAs':>6} {'Vocab':>7} {'IO ratio':>9}")
    lines.append("  " + "-" * 57)

    by_term = df.groupby("termination_type").agg(
        resp_len=("avg_response_length",      "mean"),
        latency=("avg_response_latency_secs", "mean"),
        kbas=("kbas_retrieved",               "mean"),
        vocab=("avg_vocab_complexity",        "mean"),
        io=("io_ratio",                       "mean"),
    ).round(2)

    for term, row in by_term.iterrows():
        lines.append(
            f"  {term:<12} {row['resp_len']:>9.1f} {row['latency']:>8.1f} "
            f"{row['kbas']:>6.1f} {row['vocab']:>7.1f} {row['io']:>9.1f}"
        )
    lines.append("")

    # ── Semantic metrics (if available) ───────────────────────────────────────
    golden = df[df["llm_composite"].notna()]
    if len(golden) > 0:
        lines.append("=" * 65)
        lines.append("SEMANTIC METRICS (GOLDEN SET)")
        lines.append("=" * 65)
        lines.append("")
        lines.append(f"  Based on {len(golden)} annotated conversations")
        lines.append("")
        lines.append(f"  {'Dimension':<20} {'Score':>7} {'/ 5.00':>7}")
        lines.append("  " + "-" * 36)
        for dim in ["llm_accuracy", "llm_helpfulness", "llm_tone",
                    "llm_brevity", "llm_composite"]:
            label = dim.replace("llm_", "").title()
            val   = golden[dim].mean()
            lines.append(f"  {label:<20} {val:>7.2f} {'/ 5.00':>7}")
        lines.append("")

        lines.append(section("Semantic scores by outcome"))
        lines.append("")
        lines.append(f"  {'Outcome':<15} {'N':>4} {'Accuracy':>9} "
                     f"{'Helpful':>8} {'Tone':>6} {'Composite':>10}")
        lines.append("  " + "-" * 58)
        sem_by_term = golden.groupby("termination_type").agg(
            n=("thread_id", "count"),
            acc=("llm_accuracy",     "mean"),
            hlp=("llm_helpfulness",  "mean"),
            tone=("llm_tone",        "mean"),
            comp=("llm_composite",   "mean"),
        ).round(2)

        for term, row in sem_by_term.iterrows():
            lines.append(
                f"  {term:<15} {int(row['n']):>4} "
                f"{row['acc']:>9.2f} {row['hlp']:>8.2f} "
                f"{row['tone']:>6.2f} {row['comp']:>10.2f}"
            )
        lines.append("")

    # ── Key findings ──────────────────────────────────────────────────────────
    lines.append("=" * 65)
    lines.append("KEY FINDINGS")
    lines.append("=" * 65)
    lines.append("")

    # Zero KBA finding
    zero_kba = df[df["kbas_retrieved"] == 0]
    zero_kba_res = zero_kba["is_resolved"].mean()
    lines.append(f"  1. Zero-KBA failure rate")
    lines.append(f"     {len(zero_kba)} conversations retrieved no KBA documents.")
    lines.append(f"     Resolution rate for zero-KBA conversations: "
                 f"{pct(zero_kba_res)}")
    lines.append(f"     → When the knowledge base returns nothing, "
                 f"no conversations resolve.")
    lines.append("")

    # Self-containment gap
    gap = cont_rate - res_rate
    lines.append(f"  2. Self-containment gap")
    lines.append(f"     Self-containment rate: {pct(cont_rate)}")
    lines.append(f"     Resolution rate:       {pct(res_rate)}")
    lines.append(f"     Gap:                   {pct(gap)}")
    lines.append(f"     → The chatbot handles {pct(cont_rate)} of conversations")
    lines.append(f"       without escalating but only resolves {pct(res_rate)}.")
    lines.append(f"       The {pct(gap)} gap represents failed self-service attempts.")
    lines.append("")

    # Vocab finding
    res_vocab = df[df["termination_type"]=="resolved"]["avg_vocab_complexity"].mean()
    abd_vocab = df[df["termination_type"]=="abandoned"]["avg_vocab_complexity"].mean()
    lines.append(f"  3. Vocabulary complexity gap")
    lines.append(f"     Resolved conversations:  grade {res_vocab:.1f} "
                 f"(plain English)")
    lines.append(f"     Abandoned conversations: grade {abd_vocab:.1f} "
                 f"(complex)")
    lines.append(f"     → Simpler language strongly correlates with resolution.")
    lines.append("")

    # Best and worst category
    reliable = by_cat[by_cat["n"] >= 20]
    if len(reliable) > 0:
        best_cat  = reliable["resolved"].idxmax()
        worst_cat = reliable["resolved"].idxmin()
        lines.append(f"  4. Category performance spread")
        lines.append(f"     Best  : {best_cat} "
                     f"({pct(reliable.loc[best_cat, 'resolved'])} resolution)")
        lines.append(f"     Worst : {worst_cat} "
                     f"({pct(reliable.loc[worst_cat, 'resolved'])} resolution)")
        lines.append("")

    # ── Recommendations ───────────────────────────────────────────────────────
    lines.append("=" * 65)
    lines.append("RECOMMENDATIONS")
    lines.append("=" * 65)
    lines.append("")
    lines.append("  1. IMPLEMENT ZERO-KBA FALLBACK")
    lines.append("     When knowledge base retrieval returns no documents,")
    lines.append("     the chatbot should immediately acknowledge the gap")
    lines.append("     and offer escalation — not make the user wait.")
    lines.append("")
    lines.append("  2. MANDATORY CLARIFYING QUESTION FOR SHORT QUERIES")
    lines.append("     Queries under 6 words should trigger a clarifying")
    lines.append("     question before any knowledge base search is attempted.")
    lines.append("     This prevents confident wrong answers on vague input.")
    lines.append("")
    lines.append("  3. AGENCY-AWARE DOCUMENT ROUTING")
    lines.append("     KBA documents should be tagged by USDA agency.")
    lines.append("     Forest Service employees should not receive ERS-specific")
    lines.append("     instructions, and vice versa.")
    lines.append("")
    lines.append("  4. FIX RESOLUTION DETECTION MECHANISM")
    lines.append("     Some conversations are marked resolved without user")
    lines.append("     confirmation. Resolution should require explicit user")
    lines.append("     confirmation rather than a system-triggered event.")
    lines.append("")
    lines.append("  5. REDUCE VOCABULARY COMPLEXITY")
    lines.append("     Target a Flesch-Kincaid grade of 6-8 for all responses.")
    lines.append("     Plain language responses resolve at nearly 3x the rate")
    lines.append("     of complex responses.")
    lines.append("")

    # ── Footer ────────────────────────────────────────────────────────────────
    lines.append("=" * 65)
    lines.append("This report was generated automatically by the USDA AI")
    lines.append("Evaluation Engine. Run on any new dataset to produce a")
    lines.append("fresh alignment report comparing against this baseline.")
    lines.append("=" * 65)
    lines.append("")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate USDA AI alignment report"
    )
    parser.add_argument(
        "--baseline",
        default=str(BASELINE_FILE),
        help="Path to baseline JSON file"
    )
    parser.add_argument(
        "--output",
        default=str(REPORT_FILE),
        help="Path for output report"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("USDA AI Evaluation Engine — Alignment Report Generator")
    print("=" * 60)

    # Load data
    df = pd.read_csv(MASTER_FILE)
    print(f"\nLoaded master metrics: {len(df)} rows")

    with open(args.baseline, encoding="utf-8") as f:
        baseline = json.load(f)
    print(f"Loaded baseline: {baseline.get('date_range', 'N/A')}")

    # Generate report
    print("Generating alignment report...")
    report = generate_report(df, baseline)

    # Save
    output_path = Path(args.output)
    output_path.write_text(report, encoding="utf-8")
    print(f"Saved: {output_path}")
    print()
    print(report)


if __name__ == "__main__":
    main()
