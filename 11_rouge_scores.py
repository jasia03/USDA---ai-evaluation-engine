"""
USDA AI Evaluation Engine
Step 11: ROUGE Scores

Computes ROUGE similarity metrics between the chatbot's actual first
response and the human-annotated ideal response for each of the 37
golden set conversations.

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures
how much of the ideal response content appears in the actual response.

Metrics computed:
  rouge1   — unigram overlap (individual word matches)
  rouge2   — bigram overlap (two-word phrase matches)
  rougeL   — longest common subsequence (structural similarity)

Each metric returns:
  precision — how much of the actual response is in the ideal response
  recall    — how much of the ideal response is covered by the actual
  f1        — harmonic mean of precision and recall (primary score)

Outputs:
  data/rouge_scores.csv         — per-conversation ROUGE scores
  data/rouge_summary.txt        — summary with breakdowns
"""

import re
import json
import csv
import pandas as pd
from pathlib import Path
from rouge_score import rouge_scorer

# ── Paths ─────────────────────────────────────────────────────────────────────
ANNOTATION_FILE = Path("golden_set_for_annotation.txt")
TRACES_FILE     = Path("data/traces.json")
MASTER_FILE     = Path("data/master_metrics.csv")
ROUGE_CSV       = Path("data/rouge_scores.csv")
SUMMARY_FILE    = Path("data/rouge_summary.txt")


# ── Parse annotations ─────────────────────────────────────────────────────────
def parse_annotations(filepath: Path) -> dict:
    """Parse the human annotation file to extract ideal responses."""
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    # Only work with content after the first CONVERSATION header
    first_conv = content.find("CONVERSATION 01")
    if first_conv == -1:
        return {}
    content = content[first_conv:]

    blocks = re.split(
        r"={6,}\nCONVERSATION \d+ of \d+\n={6,}",
        content
    )

    annotations = {}
    for block in blocks:
        thread_match = re.search(r"Thread ID\s*:\s*(\S+)", block)
        if not thread_match:
            continue
        thread_id = thread_match.group(1)

        all_blocks = re.findall(r">>>(.*?)<<<", block, re.DOTALL)
        ideal = all_blocks[0].strip() if all_blocks else ""

        if not ideal or "[Write 2-4 sentences" in ideal:
            continue

        ideal = re.sub(r"^\[CHATBOT\]\s*", "", ideal).strip()
        annotations[thread_id] = ideal

    return annotations

def get_first_real_ai_response(trace: dict) -> str:
    """Get the first substantive AI text response from a trace."""
    for msg in trace["messages"]:
        if msg["role"] != "ai":
            continue
        content = msg["content"].strip()
        if content.startswith("{") and "buttons" in content:
            continue
        if len(content.split()) >= 5:
            return content
    return ""


def clean_for_rouge(text: str) -> str:
    """Clean text for ROUGE computation — remove markdown, KBA refs, extra whitespace."""
    # Remove KBA document references like [KBA00144626...]
    text = re.sub(r"\[KBA\w+[^\]]*\]", "", text)
    # Remove markdown bold/italic markers
    text = re.sub(r"\*+", "", text)
    # Remove numbered list markers
    text = re.sub(r"^\s*\d+\.\s*", "", text, flags=re.MULTILINE)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("USDA AI Evaluation Engine — ROUGE Scores")
    print("=" * 60)

    # Load data
    annotations = parse_annotations(ANNOTATION_FILE)
    print(f"\nAnnotations loaded: {len(annotations)}")

    with open(TRACES_FILE, encoding="utf-8") as f:
        traces = json.load(f)
    traces_by_id = {t["thread_id"]: t for t in traces}

    master = pd.read_csv(MASTER_FILE)
    golden_ids = master[master["llm_composite"].notna()]["thread_id"].tolist()
    print(f"Golden set threads: {len(golden_ids)}")

    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    rows = []
    skipped = 0

    for thread_id in golden_ids:
        ideal = annotations.get(thread_id, "")
        if not ideal:
            skipped += 1
            continue

        trace = traces_by_id.get(thread_id)
        if not trace:
            skipped += 1
            continue

        actual = get_first_real_ai_response(trace)
        if not actual:
            # No AI response — chatbot stalled (zero KBA case)
            # ROUGE scores are 0 for all dimensions
            rows.append({
                "thread_id":        thread_id,
                "termination_type": next(
                    (t["termination_type"] for t in traces if t["thread_id"] == thread_id), ""
                ),
                "template_category": next(
                    (t["template_category"] for t in traces if t["thread_id"] == thread_id), ""
                ),
                "rouge1_precision": 0.0, "rouge1_recall": 0.0, "rouge1_f1": 0.0,
                "rouge2_precision": 0.0, "rouge2_recall": 0.0, "rouge2_f1": 0.0,
                "rougeL_precision": 0.0, "rougeL_recall": 0.0, "rougeL_f1": 0.0,
                "ideal_word_count":  len(ideal.split()),
                "actual_word_count": 0,
                "note": "no_ai_response",
            })
            continue

        # Clean both texts
        ideal_clean  = clean_for_rouge(ideal)
        actual_clean = clean_for_rouge(actual)

        # Compute ROUGE scores
        scores = scorer.score(ideal_clean, actual_clean)

        # Get termination and category from master table
        master_row = master[master["thread_id"] == thread_id].iloc[0]

        rows.append({
            "thread_id":         thread_id,
            "termination_type":  master_row["termination_type"],
            "template_category": master_row["template_category"],
            "rouge1_precision":  round(scores["rouge1"].precision, 4),
            "rouge1_recall":     round(scores["rouge1"].recall, 4),
            "rouge1_f1":         round(scores["rouge1"].fmeasure, 4),
            "rouge2_precision":  round(scores["rouge2"].precision, 4),
            "rouge2_recall":     round(scores["rouge2"].recall, 4),
            "rouge2_f1":         round(scores["rouge2"].fmeasure, 4),
            "rougeL_precision":  round(scores["rougeL"].precision, 4),
            "rougeL_recall":     round(scores["rougeL"].recall, 4),
            "rougeL_f1":         round(scores["rougeL"].fmeasure, 4),
            "ideal_word_count":  len(ideal.split()),
            "actual_word_count": len(actual.split()),
            "note": "",
        })

    print(f"Scored: {len(rows)} conversations  |  Skipped: {skipped}")

    # Save CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(ROUGE_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {ROUGE_CSV}")

    # Build summary
    summary = build_summary(rows)
    SUMMARY_FILE.write_text(summary, encoding="utf-8")
    print(f"Saved: {SUMMARY_FILE}")
    print()
    print(summary)

    return rows


def build_summary(rows: list) -> str:
    if not rows:
        return "No ROUGE scores computed."

    df = pd.DataFrame(rows)

    def avg(col):
        return round(df[col].mean(), 4)

    lines = []
    lines.append("=" * 65)
    lines.append("USDA AI Evaluation Engine — ROUGE Score Summary")
    lines.append("=" * 65)
    lines.append(f"Conversations scored: {len(df)}")
    lines.append("")

    # ── Overall averages ──────────────────────────────────────────────────────
    lines.append("── Overall ROUGE averages ────────────────────────────────────")
    lines.append("")
    lines.append(f"  {'Metric':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    lines.append("  " + "-" * 54)
    for metric in ["rouge1", "rouge2", "rougeL"]:
        lines.append(
            f"  {metric:<20} "
            f"{avg(f'{metric}_precision'):>10.4f} "
            f"{avg(f'{metric}_recall'):>10.4f} "
            f"{avg(f'{metric}_f1'):>10.4f}"
        )
    lines.append("")
    lines.append("  Interpretation:")
    lines.append("  ROUGE-1 F1 measures single-word overlap with the ideal response.")
    lines.append("  ROUGE-2 F1 measures phrase-level similarity.")
    lines.append("  ROUGE-L F1 measures structural/sequential similarity.")
    lines.append("  Score of 0.3+ indicates moderate alignment with ideal response.")
    lines.append("  Score of 0.5+ indicates strong alignment.")
    lines.append("")

    # ── By termination type ───────────────────────────────────────────────────
    lines.append("── ROUGE-L F1 by termination type ────────────────────────────")
    lines.append("")
    by_term = df.groupby("termination_type")["rougeL_f1"].agg(["mean", "count"]).round(4)
    lines.append(f"  {'Outcome':<15} {'N':>5} {'ROUGE-L F1':>12}")
    lines.append("  " + "-" * 35)
    for term, row in by_term.iterrows():
        lines.append(f"  {term:<15} {int(row['count']):>5} {row['mean']:>12.4f}")
    lines.append("")

    # ── By category ───────────────────────────────────────────────────────────
    lines.append("── ROUGE-L F1 by category ────────────────────────────────────")
    lines.append("")
    by_cat = df.groupby("template_category")["rougeL_f1"].agg(["mean", "count"]).round(4)
    by_cat = by_cat.sort_values("mean", ascending=False)
    lines.append(f"  {'Category':<45} {'N':>4} {'ROUGE-L F1':>12}")
    lines.append("  " + "-" * 65)
    for cat, row in by_cat.iterrows():
        lines.append(f"  {cat:<45} {int(row['count']):>4} {row['mean']:>12.4f}")
    lines.append("")

    # ── Top and bottom conversations ──────────────────────────────────────────
    lines.append("── Highest ROUGE-L F1 (most similar to ideal) ────────────────")
    top = df.nlargest(5, "rougeL_f1")[["thread_id", "termination_type", "rougeL_f1"]]
    for _, row in top.iterrows():
        lines.append(f"  {row['thread_id'][:8]}  {row['termination_type']:<12}  rougeL={row['rougeL_f1']:.4f}")
    lines.append("")

    lines.append("── Lowest ROUGE-L F1 (least similar to ideal) ────────────────")
    bot = df.nsmallest(5, "rougeL_f1")[["thread_id", "termination_type", "rougeL_f1"]]
    for _, row in bot.iterrows():
        lines.append(f"  {row['thread_id'][:8]}  {row['termination_type']:<12}  rougeL={row['rougeL_f1']:.4f}")
    lines.append("")

    # ── Key insight ───────────────────────────────────────────────────────────
    res_avg = df[df["termination_type"] == "resolved"]["rougeL_f1"].mean()
    esc_avg = df[df["termination_type"] == "escalated"]["rougeL_f1"].mean()
    lines.append("── Key insight ───────────────────────────────────────────────")
    lines.append(
        f"  Resolved conversations average ROUGE-L F1: {res_avg:.4f}"
    )
    if not pd.isna(esc_avg):
        lines.append(
            f"  Escalated conversations average ROUGE-L F1: {esc_avg:.4f}"
        )
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
