"""
USDA AI Evaluation Engine
Step 8: Cross-Layer Analysis + Drift Detection

Two components in one script:

PART A — Cross-layer correlation analysis
  Connects structural signals to north star outcomes to answer:
  - Does KBA count predict resolution?
  - Does latency predict abandonment?
  - Does vocabulary complexity predict helpfulness scores?
  - Which structural metrics are strongest predictors of success?

PART B — Drift detection
  Establishes a baseline from the full dataset and compares
  January vs February performance to detect metric drift.
  Flags any metric that moved beyond the alert threshold.

Outputs:
  data/correlation_analysis.txt   — cross-layer findings
  data/drift_report.txt           — drift detection results
  data/baseline_metrics.json      — baseline for future runs
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
MASTER_FILE      = Path("data/master_metrics.csv")
CORRELATION_FILE = Path("data/correlation_analysis.txt")
DRIFT_FILE       = Path("data/drift_report.txt")
BASELINE_FILE    = Path("data/baseline_metrics.json")

# ── Drift thresholds ──────────────────────────────────────────────────────────
# How much a metric can move before flagging
WARN_THRESHOLD  = 0.05   # 5 percentage points / 0.5 score points
ALERT_THRESHOLD = 0.10   # 10 percentage points / 1.0 score points


# ── Helpers ───────────────────────────────────────────────────────────────────
def safe_corr(x: pd.Series, y: pd.Series) -> float:
    """Pearson correlation between two series, dropping NaN pairs."""
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 5:
        return float("nan")
    return round(df["x"].corr(df["y"]), 3)


def interpret_corr(r: float) -> str:
    """Plain English interpretation of a correlation coefficient."""
    if pd.isna(r):
        return "insufficient data"
    a = abs(r)
    direction = "positive" if r > 0 else "negative"
    if a >= 0.5:
        strength = "strong"
    elif a >= 0.3:
        strength = "moderate"
    elif a >= 0.1:
        strength = "weak"
    else:
        return "negligible"
    return f"{strength} {direction}"


def group_avg(df: pd.DataFrame, group_col: str, metric_col: str) -> dict:
    """Average of metric_col grouped by group_col."""
    return df.groupby(group_col)[metric_col].mean().round(3).to_dict()


def flag(diff: float, warn: float, alert: float) -> str:
    """Return drift status flag."""
    if abs(diff) >= alert:
        return "ALERT"
    elif abs(diff) >= warn:
        return "WARN"
    else:
        return "STABLE"


# ── PART A: Cross-layer correlation analysis ──────────────────────────────────
def run_correlation_analysis(df: pd.DataFrame) -> str:
    lines = []
    lines.append("=" * 65)
    lines.append("USDA AI Evaluation Engine — Cross-Layer Correlation Analysis")
    lines.append("=" * 65)
    lines.append("")

    # ── 1. Structural metrics vs resolution ───────────────────────────────────
    lines.append("── Structural metrics vs resolution outcome ───────────────────")
    lines.append("  Does each structural metric predict whether a conversation")
    lines.append("  will be resolved? (correlation with is_resolved flag)")
    lines.append("")

    structural_metrics = [
        ("avg_response_length",     "Response length (words)"),
        ("total_turns",             "Total conversation turns"),
        ("human_turns",             "Human turns"),
        ("avg_response_latency_secs","Response latency (secs)"),
        ("kbas_retrieved",          "KBAs retrieved"),
        ("io_ratio",                "IO ratio"),
        ("session_duration_secs",   "Session duration (secs)"),
        ("avg_vocab_complexity",    "Vocabulary complexity"),
    ]

    correlations = []
    for col, label in structural_metrics:
        r = safe_corr(df[col], df["is_resolved"])
        correlations.append((label, r, interpret_corr(r)))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]) if not pd.isna(x[1]) else 0,
                      reverse=True)

    lines.append(f"  {'Metric':<35} {'r':>6}  {'Interpretation'}")
    lines.append("  " + "-" * 62)
    for label, r, interp in correlations:
        r_str = f"{r:>6.3f}" if not pd.isna(r) else "   N/A"
        lines.append(f"  {label:<35} {r_str}  {interp}")
    lines.append("")

    # ── 2. KBA count vs resolution rate ──────────────────────────────────────
    lines.append("── KBA count vs resolution rate ──────────────────────────────")
    lines.append("  Does retrieving more documents lead to better outcomes?")
    lines.append("")

    df["kba_bucket"] = pd.cut(
        df["kbas_retrieved"],
        bins=[-1, 0, 2, 5, 16],
        labels=["0 KBAs", "1-2 KBAs", "3-5 KBAs", "6+ KBAs"]
    )

    kba_analysis = df.groupby("kba_bucket", observed=True).agg(
        threads=("thread_id", "count"),
        resolution_rate=("is_resolved", "mean"),
        escalation_rate=("is_escalated", "mean"),
        avg_latency=("avg_response_latency_secs", "mean")
    ).round(3)

    lines.append(f"  {'KBA bucket':<12} {'Threads':>8} {'Res. rate':>10} "
                 f"{'Esc. rate':>10} {'Avg latency':>12}")
    lines.append("  " + "-" * 56)
    for bucket, row in kba_analysis.iterrows():
        lines.append(
            f"  {str(bucket):<12} {int(row['threads']):>8} "
            f"{row['resolution_rate']*100:>9.1f}% "
            f"{row['escalation_rate']*100:>9.1f}% "
            f"{row['avg_latency']:>11.1f}s"
        )
    lines.append("")

    # ── 3. Latency vs termination type ────────────────────────────────────────
    lines.append("── Response latency vs termination type ──────────────────────")
    lines.append("  Do users abandon faster-responding or slower-responding bots?")
    lines.append("")

    latency_by_term = df.groupby("termination_type")["avg_response_latency_secs"].agg(
        ["mean", "median", "std"]
    ).round(2)

    lines.append(f"  {'Outcome':<15} {'Mean (s)':>9} {'Median (s)':>11} {'Std dev':>9}")
    lines.append("  " + "-" * 48)
    for term, row in latency_by_term.iterrows():
        lines.append(
            f"  {term:<15} {row['mean']:>9.2f} {row['median']:>11.2f} "
            f"{row['std']:>9.2f}"
        )
    lines.append("")

    # ── 4. Vocab complexity vs termination type ───────────────────────────────
    lines.append("── Vocabulary complexity vs termination type ──────────────────")
    lines.append("  Do harder-to-read responses drive abandonment?")
    lines.append("")

    vocab_by_term = df.groupby("termination_type")["avg_vocab_complexity"].mean().round(2)
    lines.append(f"  {'Outcome':<15} {'Avg FK grade':>13} {'Interpretation'}")
    lines.append("  " + "-" * 55)
    for term, grade in vocab_by_term.items():
        if grade <= 6:
            interp = "easy (elementary school)"
        elif grade <= 8:
            interp = "accessible (middle school)"
        elif grade <= 10:
            interp = "moderate (high school)"
        else:
            interp = "complex (college level)"
        lines.append(f"  {term:<15} {grade:>13.2f}  {interp}")
    lines.append("")

    # ── 5. Semantic scores vs structural metrics (golden set) ─────────────────
    golden = df[df["llm_composite"].notna()].copy()
    if len(golden) > 0:
        lines.append("── Semantic scores vs structural metrics (golden set) ─────────")
        lines.append(f"  Based on {len(golden)} annotated conversations")
        lines.append("")

        sem_correlations = []
        for col, label in structural_metrics:
            r = safe_corr(golden[col], golden["llm_composite"])
            sem_correlations.append((label, r, interpret_corr(r)))

        sem_correlations.sort(
            key=lambda x: abs(x[1]) if not pd.isna(x[1]) else 0,
            reverse=True
        )

        lines.append(f"  {'Metric':<35} {'r':>6}  {'Interpretation'}")
        lines.append("  " + "-" * 62)
        for label, r, interp in sem_correlations:
            r_str = f"{r:>6.3f}" if not pd.isna(r) else "   N/A"
            lines.append(f"  {label:<35} {r_str}  {interp}")
        lines.append("")

        # Human score vs LLM composite by category
        lines.append("── Human score vs LLM composite by category ──────────────────")
        lines.append("")
        cat_sem = golden.groupby("template_category")[
            ["human_score", "llm_composite"]
        ].mean().round(2)

        lines.append(f"  {'Category':<45} {'Human':>7} {'LLM':>7} {'Gap':>7}")
        lines.append("  " + "-" * 70)
        for cat, row in cat_sem.iterrows():
            gap = row["llm_composite"] - row["human_score"]
            lines.append(
                f"  {cat:<45} {row['human_score']:>7.2f} "
                f"{row['llm_composite']:>7.2f} {gap:>+7.2f}"
            )
        lines.append("")

    # ── 6. Key insight summary ────────────────────────────────────────────────
    lines.append("── Key analytical insights ────────────────────────────────────")

    # KBA insight
    zero_kba_res = df[df["kbas_retrieved"] == 0]["is_resolved"].mean()
    high_kba_res = df[df["kbas_retrieved"] >= 5]["is_resolved"].mean()
    lines.append(
        f"  KBA impact: conversations with 0 KBAs resolve at "
        f"{zero_kba_res*100:.1f}% vs {high_kba_res*100:.1f}% "
        f"for 5+ KBAs"
    )

    # Latency insight
    res_lat  = df[df["termination_type"]=="resolved"]["avg_response_latency_secs"].mean()
    abd_lat  = df[df["termination_type"]=="abandoned"]["avg_response_latency_secs"].mean()
    lines.append(
        f"  Latency gap: resolved conversations average {res_lat:.1f}s "
        f"vs {abd_lat:.1f}s for abandoned"
    )

    # Vocab insight
    res_fk = df[df["termination_type"]=="resolved"]["avg_vocab_complexity"].mean()
    abd_fk = df[df["termination_type"]=="abandoned"]["avg_vocab_complexity"].mean()
    lines.append(
        f"  Vocab gap: resolved responses average grade {res_fk:.1f} "
        f"vs grade {abd_fk:.1f} for abandoned"
    )

    lines.append("")
    return "\n".join(lines)


# ── PART B: Drift detection ───────────────────────────────────────────────────
def run_drift_detection(df: pd.DataFrame) -> tuple[str, dict]:
    lines = []
    lines.append("=" * 65)
    lines.append("USDA AI Evaluation Engine — Drift Detection Report")
    lines.append("=" * 65)
    lines.append("")
    lines.append(f"  Warn threshold  : ±{WARN_THRESHOLD*100:.0f} percentage points")
    lines.append(f"  Alert threshold : ±{ALERT_THRESHOLD*100:.0f} percentage points")
    lines.append("")

    # ── Establish baseline from full dataset ──────────────────────────────────
    baseline = {
        "resolution_rate":       float(df["is_resolved"].mean()),
        "escalation_rate":       float(df["is_escalated"].mean()),
        "self_containment_rate": float(df["is_self_contained"].mean()),
        "avg_response_length":   float(df["avg_response_length"].mean()),
        "avg_latency_secs":      float(df["avg_response_latency_secs"].mean()),
        "avg_kbas_retrieved":    float(df["kbas_retrieved"].mean()),
        "avg_vocab_complexity":  float(df["avg_vocab_complexity"].mean()),
        "avg_io_ratio":          float(df["io_ratio"].mean()),
        "total_conversations":   int(len(df)),
        "date_range":            f"{df['conversation_date'].min()} to {df['conversation_date'].max()}",
    }

    lines.append("── Baseline metrics (full dataset) ───────────────────────────")
    for key, val in baseline.items():
        if isinstance(val, float):
            lines.append(f"  {key:<35} {val:.4f}")
        else:
            lines.append(f"  {key:<35} {val}")
    lines.append("")

    # ── Compare January vs February ───────────────────────────────────────────
    jan = df[df["month"] == "2026-01"]
    feb = df[df["month"] == "2026-02"]

    lines.append("── Monthly drift analysis: January vs February ───────────────")
    lines.append(f"  January  : {len(jan)} conversations")
    lines.append(f"  February : {len(feb)} conversations")
    lines.append("")

    drift_metrics = [
        ("resolution_rate",       "is_resolved",                  "Resolution rate"),
        ("escalation_rate",       "is_escalated",                 "Escalation rate"),
        ("self_containment_rate", "is_self_contained",            "Self-containment rate"),
        ("avg_response_length",   "avg_response_length",          "Avg response length"),
        ("avg_latency_secs",      "avg_response_latency_secs",    "Avg latency (secs)"),
        ("avg_kbas_retrieved",    "kbas_retrieved",               "Avg KBAs retrieved"),
        ("avg_vocab_complexity",  "avg_vocab_complexity",         "Avg vocab complexity"),
    ]

    lines.append(f"  {'Metric':<35} {'Jan':>8} {'Feb':>8} {'Change':>8} {'Status':>8}")
    lines.append("  " + "-" * 73)

    drift_results = []
    for key, col, label in drift_metrics:
        jan_val = jan[col].mean()
        feb_val = feb[col].mean()
        diff    = feb_val - jan_val
        status  = flag(diff, WARN_THRESHOLD, ALERT_THRESHOLD)

        drift_results.append({
            "metric":  label,
            "january": round(jan_val, 4),
            "february":round(feb_val, 4),
            "change":  round(diff, 4),
            "status":  status,
        })

        lines.append(
            f"  {label:<35} {jan_val:>8.3f} {feb_val:>8.3f} "
            f"{diff:>+8.3f} {status:>8}"
        )
    lines.append("")

    # ── Flag summary ──────────────────────────────────────────────────────────
    alerts = [r for r in drift_results if r["status"] == "ALERT"]
    warns  = [r for r in drift_results if r["status"] == "WARN"]
    stable = [r for r in drift_results if r["status"] == "STABLE"]

    lines.append("── Drift flag summary ────────────────────────────────────────")
    lines.append(f"  ALERT  : {len(alerts)} metric(s)")
    for r in alerts:
        lines.append(f"    → {r['metric']} changed by {r['change']:+.3f}")
    lines.append(f"  WARN   : {len(warns)} metric(s)")
    for r in warns:
        lines.append(f"    → {r['metric']} changed by {r['change']:+.3f}")
    lines.append(f"  STABLE : {len(stable)} metric(s)")
    lines.append("")

    # ── Interpretation ────────────────────────────────────────────────────────
    lines.append("── Interpretation ────────────────────────────────────────────")
    res_change = next(r for r in drift_results if "Resolution" in r["metric"])
    if res_change["change"] > 0:
        lines.append(
            f"  Resolution rate IMPROVED by "
            f"{res_change['change']*100:.1f} percentage points "
            f"from January to February."
        )
    else:
        lines.append(
            f"  Resolution rate DECLINED by "
            f"{abs(res_change['change'])*100:.1f} percentage points "
            f"from January to February."
        )

    lines.append(
        f"  Overall trend: the chatbot appears to be "
        f"{'improving' if res_change['change'] > 0 else 'declining'} "
        f"over the observed period."
    )
    lines.append(
        f"  Recommendation: continue monitoring on a monthly cadence "
        f"using the baseline established above."
    )
    lines.append("")

    return "\n".join(lines), baseline


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("USDA AI Evaluation Engine — Cross-Layer Analysis + Drift")
    print("=" * 60)

    df = pd.read_csv(MASTER_FILE)
    print(f"\nLoaded master metrics: {len(df)} rows × {len(df.columns)} columns")

    # Part A
    print("\nRunning cross-layer correlation analysis...")
    correlation_report = run_correlation_analysis(df)
    CORRELATION_FILE.write_text(correlation_report, encoding="utf-8")
    print(f"Saved: {CORRELATION_FILE}")

    # Part B
    print("Running drift detection...")
    drift_report, baseline = run_drift_detection(df)
    DRIFT_FILE.write_text(drift_report, encoding="utf-8")
    print(f"Saved: {DRIFT_FILE}")

    # Save baseline for future runs
    with open(BASELINE_FILE, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)
    print(f"Saved: {BASELINE_FILE}")

    print()
    print(correlation_report)
    print(drift_report)


if __name__ == "__main__":
    main()