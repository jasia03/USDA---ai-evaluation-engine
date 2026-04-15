"""
USDA AI Evaluation Engine
Entry Point: run_evaluation.py

Runs the complete evaluation pipeline in one command.
Point it at any USDA chatbot conversation dataset and it will
produce the full suite of metrics, analysis, and alignment report.

Usage:
  python run_evaluation.py --data your_data.xlsx
  python run_evaluation.py --data your_data.xlsx --baseline data/baseline_metrics.json
  python run_evaluation.py --help

Pipeline steps:
  1. Parse raw Excel data into structured traces
  2. Run data health check
  3. Compute structural metrics
  4. Compute north star metrics
  5. Select golden set for annotation (if not already done)
  6. Run LLM judge (if API key is set and annotation file exists)
  7. Merge all metrics into master table
  8. Run cross-layer analysis and drift detection
  9. Generate alignment report
# Additional analysis scripts (run independently):
#   python 10_failure_classifier.py   → data/failure_patterns.csv
#   python 11_rouge_scores.py         → data/rouge_scores.csv
#   python 12_latency_and_deflection.py → data/latency_per_turn.csv, deflection_rate.csv

Outputs (all saved to data/ folder):
  traces.json
  health_check_report.txt
  structural_metrics.csv + structural_summary.txt
  north_star_metrics.csv + north_star_summary.txt
  golden_set_ids.json + golden_set_for_annotation.txt
  semantic_scores.csv         (if LLM judge runs)
  master_metrics.csv
  correlation_analysis.txt
  drift_report.txt
  baseline_metrics.json
  alignment_report.txt        ← primary deliverable
"""

import argparse
import sys
import os
import time
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────
def header(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def step(number: int, title: str):
    print(f"\n[Step {number}/9] {title}")
    print("-" * 50)


def success(msg: str):
    print(f"  ✓ {msg}")


def warn(msg: str):
    print(f"  ! {msg}")


def error(msg: str):
    print(f"  ✗ {msg}")


# ── Pipeline steps ────────────────────────────────────────────────────────────
def run_step_1(data_file: str):
    """Parse raw Excel data into traces."""
    from pathlib import Path
    import importlib.util, sys

    # Dynamically update INPUT_FILE in trace_parser and run
    spec = importlib.util.spec_from_file_location(
        "trace_parser", "01_trace_parser.py"
    )
    mod = importlib.util.module_from_spec(spec)

    # Override INPUT_FILE before executing
    import pandas as pd
    import json, re
    sys.modules["trace_parser"] = mod

    # Import and run with custom input file
    import subprocess
    result = subprocess.run(
        [sys.executable, "01_trace_parser.py", "--input", data_file],
        capture_output=True, text=True
    )

    # Fall back to running with default path if --input not supported
    if result.returncode != 0:
        result = subprocess.run(
            [sys.executable, "01_trace_parser.py"],
            capture_output=True, text=True
        )

    if result.returncode == 0:
        success("Traces parsed successfully")
        # Print key line from output
        for line in result.stdout.split("\n"):
            if "Match:" in line or "Traces built:" in line:
                print(f"    {line.strip()}")
    else:
        error("Trace parser failed")
        print(result.stderr[-500:])
        sys.exit(1)


def run_script(script: str, step_num: int, step_name: str,
               extra_args: list = None) -> bool:
    """Run a pipeline script as a subprocess."""
    import subprocess
    step(step_num, step_name)

    cmd = [sys.executable, script]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        success(f"{script} completed")
        # Print summary lines from output
        for line in result.stdout.split("\n"):
            if any(kw in line for kw in [
                "Saved:", "Traces built:", "Match:", "Done",
                "resolution rate", "Resolution rate", "PASS", "WARN", "ALERT"
            ]):
                if line.strip():
                    print(f"    {line.strip()}")
        return True
    else:
        error(f"{script} failed")
        print(result.stderr[-800:])
        return False


# ── Main pipeline ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="USDA AI Evaluation Engine — Full Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_evaluation.py
  python run_evaluation.py --data my_chatbot_data.xlsx
  python run_evaluation.py --skip-llm
  python run_evaluation.py --baseline data/baseline_metrics.json
        """
    )
    parser.add_argument(
        "--data",
        default="messages_data_student_extract.xlsx",
        help="Path to input Excel file (default: messages_data_student_extract.xlsx)"
    )
    parser.add_argument(
        "--baseline",
        default="data/baseline_metrics.json",
        help="Path to baseline metrics JSON for drift comparison"
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM judge step (use if no API key or annotation file)"
    )
    parser.add_argument(
        "--annotation",
        default="golden_set_for_annotation.txt",
        help="Path to completed annotation file for LLM judge"
    )
    args = parser.parse_args()

    # ── Startup ───────────────────────────────────────────────────────────────
    header("USDA AI EVALUATION ENGINE")
    print(f"  Data file  : {args.data}")
    print(f"  Baseline   : {args.baseline}")
    print(f"  LLM judge  : {'SKIP' if args.skip_llm else 'RUN if annotation file exists'}")
    print(f"  Started    : {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Check data file exists
    if not Path(args.data).exists():
        error(f"Data file not found: {args.data}")
        print("  Please provide the path to your Excel data file with --data")
        sys.exit(1)

    # Ensure data directory exists
    Path("data").mkdir(exist_ok=True)

    start_time = time.time()
    failed_steps = []

    # ── Step 1: Trace parser ──────────────────────────────────────────────────
    step(1, "Parsing raw conversation data into traces")
    ok = run_script("01_trace_parser.py", 1, "Trace parser")
    if not ok:
        failed_steps.append("01_trace_parser.py")

    # ── Step 2: Data health check ─────────────────────────────────────────────
    ok = run_script("02_data_health_check.py", 2, "Data health check")
    if not ok:
        failed_steps.append("02_data_health_check.py")

    # ── Step 3: Structural metrics ────────────────────────────────────────────
    ok = run_script("03_structural_metrics.py", 3, "Structural metrics")
    if not ok:
        failed_steps.append("03_structural_metrics.py")

    # ── Step 4: North star metrics ────────────────────────────────────────────
    ok = run_script("04_north_star_metrics.py", 4, "North star metrics")
    if not ok:
        failed_steps.append("04_north_star_metrics.py")

    # ── Step 5: Golden set selector ───────────────────────────────────────────
    ok = run_script("05_golden_set_selector.py", 5, "Golden set selection")
    if not ok:
        warn("Golden set selector failed — continuing without it")

    # ── Step 6: LLM judge ─────────────────────────────────────────────────────
    step(6, "LLM judge (semantic scoring)")
    annotation_exists = Path(args.annotation).exists()
    api_key_set = bool(os.environ.get("ANTHROPIC_API_KEY", ""))

    if args.skip_llm:
        warn("LLM judge skipped (--skip-llm flag set)")
    elif not annotation_exists:
        warn(f"LLM judge skipped — annotation file not found: {args.annotation}")
        warn("Complete the annotation file then re-run with the annotation path")
    elif not api_key_set:
        warn("LLM judge skipped — ANTHROPIC_API_KEY not set")
        warn("Set the key and re-run to add semantic scores")
    else:
        ok = run_script(
            "06_llm_judge.py", 6, "LLM judge",
        )
        if not ok:
            warn("LLM judge failed — continuing without semantic scores")

    # ── Step 7: Unified merger ────────────────────────────────────────────────
    ok = run_script("07_unified_merger.py", 7, "Unified metrics merger")
    if not ok:
        failed_steps.append("07_unified_merger.py")

    # ── Step 8: Analysis and drift ────────────────────────────────────────────
    ok = run_script("08_analysis_and_drift.py", 8, "Cross-layer analysis + drift detection")
    if not ok:
        failed_steps.append("08_analysis_and_drift.py")

    # ── Step 9: Alignment report ──────────────────────────────────────────────
    extra = ["--baseline", args.baseline] if Path(args.baseline).exists() else []
    ok = run_script(
        "09_alignment_report.py", 9, "Alignment report generation", extra
    )
    if not ok:
        failed_steps.append("09_alignment_report.py")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = round(time.time() - start_time, 1)

    header("PIPELINE COMPLETE")
    print(f"  Completed in : {elapsed}s")
    print(f"  Failed steps : {len(failed_steps)}")

    if failed_steps:
        for s in failed_steps:
            error(f"  {s}")
    else:
        success("All steps completed successfully")

    print()
    print("  Output files saved to data/:")
    outputs = [
        "traces.json",
        "health_check_report.txt",
        "structural_metrics.csv",
        "north_star_metrics.csv",
        "master_metrics.csv",
        "correlation_analysis.txt",
        "drift_report.txt",
        "baseline_metrics.json",
        "alignment_report.txt",
    ]
    for f in outputs:
        path = Path("data") / f
        if path.exists():
            size = path.stat().st_size
            success(f"data/{f}  ({size:,} bytes)")
        else:
            warn(f"data/{f}  (not generated)")

    print()
    print("  Primary deliverable: data/alignment_report.txt")
    print()


if __name__ == "__main__":
    main()
