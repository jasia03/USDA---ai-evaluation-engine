"""
USDA AI Evaluation Engine
Step 7: Unified Metrics Merger

Joins all three metric layers into one master metrics table:
  - Structural metrics  (588 traces)
  - North star metrics  (588 traces)
  - Semantic scores     (37 traces — golden set only)

The result is a single CSV with one row per trace and all available
metrics in one place. Semantic columns are NULL for non-golden-set
traces — this is expected and correct.

Output:
  data/master_metrics.csv   — one row per trace, all metrics combined
  data/merge_summary.txt    — validation report confirming merge quality
"""

import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
STRUCTURAL_FILE = Path("data/structural_metrics.csv")
NORTH_STAR_FILE = Path("data/north_star_metrics.csv")
SEMANTIC_FILE   = Path("data/semantic_scores.csv")
MASTER_CSV      = Path("data/master_metrics.csv")
SUMMARY_FILE    = Path("data/merge_summary.txt")


def main():
    print("=" * 60)
    print("USDA AI Evaluation Engine — Unified Metrics Merger")
    print("=" * 60)

    # ── Load all three metric files ───────────────────────────────────────────
    structural = pd.read_csv(STRUCTURAL_FILE)
    north_star = pd.read_csv(NORTH_STAR_FILE)
    semantic   = pd.read_csv(SEMANTIC_FILE)

    print(f"\nLoaded:")
    print(f"  Structural metrics : {len(structural)} rows")
    print(f"  North star metrics : {len(north_star)} rows")
    print(f"  Semantic scores    : {len(semantic)} rows")

    # ── Drop duplicate columns before merging ─────────────────────────────────
    # thread_id is the join key — keep it in structural as base
    # termination_type and template_category exist in all three — drop from right
    north_star_clean = north_star.drop(
        columns=["termination_type", "template_category", "feedback_rating"],
        errors="ignore"
    )
    semantic_clean = semantic.drop(
        columns=["termination_type", "template_category"],
        errors="ignore"
    )

    # ── Merge structural + north star (all 588 traces) ────────────────────────
    master = structural.merge(
        north_star_clean,
        on="thread_id",
        how="left",
        validate="one_to_one"
    )

    print(f"\nAfter structural + north star merge: {len(master)} rows")

    # ── Left join semantic scores (37 traces — NaN for the rest) ─────────────
    master = master.merge(
        semantic_clean,
        on="thread_id",
        how="left",
        validate="one_to_many"
    )

    print(f"After adding semantic scores       : {len(master)} rows")

    # ── Reorder columns logically ─────────────────────────────────────────────
    col_order = [
        # Identity
        "thread_id", "termination_type", "template_category",
        "conversation_date", "month", "app_version",
        "feedback_rating",
        # North star
        "is_resolved", "is_escalated", "is_self_contained", "deflection_value",
        # Structural
        "avg_response_length", "total_turns", "human_turns",
        "avg_response_latency_secs", "kbas_retrieved", "io_ratio",
        "session_duration_secs", "restart_flag", "avg_vocab_complexity",
        # Semantic (golden set only)
        "human_score", "llm_accuracy", "llm_helpfulness",
        "llm_tone", "llm_brevity", "llm_composite", "llm_reasoning",
    ]

    # Only include columns that exist
    col_order = [c for c in col_order if c in master.columns]
    master = master[col_order]

    # ── Save ──────────────────────────────────────────────────────────────────
    master.to_csv(MASTER_CSV, index=False)
    print(f"\nSaved: {MASTER_CSV}")
    print(f"Master table: {len(master)} rows × {len(master.columns)} columns")

    # ── Build validation summary ──────────────────────────────────────────────
    lines = []
    lines.append("=" * 60)
    lines.append("USDA AI Evaluation Engine — Merge Validation Report")
    lines.append("=" * 60)
    lines.append(f"\nTotal rows      : {len(master)}")
    lines.append(f"Total columns   : {len(master.columns)}")
    lines.append(f"\nColumns included:")
    for col in master.columns:
        null_count = master[col].isna().sum()
        coverage   = f"{(1 - null_count/len(master))*100:.0f}%"
        lines.append(f"  {col:<35} {coverage:>8} coverage  ({null_count} nulls)")

    lines.append(f"\nSemantic coverage:")
    lines.append(f"  Golden set threads scored : {master['llm_composite'].notna().sum()}")
    lines.append(f"  Non-golden threads (NULL) : {master['llm_composite'].isna().sum()}")

    lines.append(f"\nKey metric ranges:")
    for col in ["avg_response_length", "avg_response_latency_secs",
                "kbas_retrieved", "llm_composite"]:
        if col in master.columns:
            non_null = master[col].dropna()
            lines.append(
                f"  {col:<35} "
                f"min={non_null.min():.1f}  "
                f"max={non_null.max():.1f}  "
                f"mean={non_null.mean():.1f}"
            )

    lines.append(f"\nTermination type counts:")
    for term, count in master["termination_type"].value_counts().items():
        lines.append(f"  {term:<15} {count}")

    summary = "\n".join(lines)
    SUMMARY_FILE.write_text(summary, encoding="utf-8")
    print(f"Saved: {SUMMARY_FILE}")
    print()
    print(summary)

    return master


if __name__ == "__main__":
    main()